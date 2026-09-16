"""CPU toy models only: sampled spans, gradient ownership and adapter lock."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from agent.hf_backend import HuggingFaceBackend,HuggingFaceConfig
from grpo.policy import GRPOPolicy
from grpo.rollouts import (collect_group,collect_group_batched,collect_group_serial,
                           derived_seed,variance_report)
from grpo.schemas import GRPOConfig,TurnTrace,RolloutGroup,TaskCoordinate
from grpo.tasks import build_task_manifest,coordinates,training_schedule,expression_key
from grpo.trainer import update_groups,weights_digest
from tasks.generator import TaskGenerator


@pytest.fixture
def toy_policy():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Embedding(16,16)
            self.proj = torch.nn.Module()
            self.proj.lora_A = torch.nn.ModuleDict({n:torch.nn.Linear(16,2,bias=False) for n in ("default","sft_reference")})
            self.proj.lora_B = torch.nn.ModuleDict({n:torch.nn.Linear(2,16,bias=False) for n in ("default","sft_reference")})
            for layers in (self.proj.lora_A,self.proj.lora_B):
                layers["sft_reference"].load_state_dict(layers["default"].state_dict())
            self.config = SimpleNamespace(use_cache=True)
            self.adapter = "default"
            self.calls = []

        def set_adapter(self, name):
            self.adapter = name
            self.calls.append(("switch",name))

        def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
            self.calls.append(("forward",self.adapter,torch.is_grad_enabled()))
            self.last_input = input_ids.clone()
            x = self.base(input_ids)
            logits = x+self.proj.lora_B[self.adapter](self.proj.lora_A[self.adapter](x))
            return SimpleNamespace(logits=logits[:,-logits_to_keep:,:])

    p = GRPOPolicy.__new__(GRPOPolicy)
    p.model = Model()
    p.config = GRPOConfig("fixture","fixture","fixture",device="cpu",dtype="float32",group_size=2)
    p.policy_adapter,p.reference_adapter,p._locked = "default","sft_reference",False
    p.switch("policy")
    p._assert_equal_initial_adapters()
    return p


def traced_turn(policy,prompt=None,generated=None):
    turn = TurnTrace([{"role":"user","content":"Task"}],prompt or [1,3,4],generated or [5,6,2],[-1.]*3,"raw",True,False)
    turn.old_logprobs = [-1.]*len(turn.generated_ids)
    turn.old_logprobs = policy.score(turn).detach().tolist()
    return turn


def test_raw_ids_shift_eos_and_context_mask(toy_policy):
    p = toy_policy
    turn = traced_turn(p)
    assert p.model.last_input.tolist() == [[1,3,4,5,6]]
    assert turn.policy_mask == [0,0,0,1,1,1]
    assert len(p.score(turn)) == 3  # EOS also supervised; no PAD insertion.
    members = [SimpleNamespace(turns=[turn])]
    assert p.validate_alignment(members)["max_logprob_error"] == 0


def test_multiturn_tokens_and_frozen_reference_policy_only_update(toy_policy):
    torch = pytest.importorskip("torch")
    p = toy_policy
    members = []
    for reward,ids in [(0,[5,6,2]),(1,[8,9,2])]:
        turns = [traced_turn(p,generated=ids),traced_turn(p,prompt=[1,3,4,5,6,2,12,13],generated=ids)]
        members.append(SimpleNamespace(turns=turns,reward=SimpleNamespace(total=reward),reference_logprobs=[]))
    frozen = weights_digest(p,frozen=True)
    previous = weights_digest(p,frozen=False)
    optimizer = torch.optim.AdamW(p.trainable_parameters(),lr=.01)
    p.model.calls.clear()
    stats = update_groups(p,[RolloutGroup("toy",members)],optimizer)
    assert stats["sampled_token_count"] == 12 and stats["gradient_norm"] > 0
    assert weights_digest(p,frozen=True) == frozen
    assert weights_digest(p,frozen=False) != previous
    calls = p.model.calls
    assert calls[0] == ("switch","sft_reference")
    assert all(c == ("forward","sft_reference",False) for c in calls[1:5])
    assert calls[5] == ("switch","default")
    assert all(c == ("forward","default",True) for c in calls[6:])
    assert all(not param.requires_grad and param.grad is None for name,param in p.model.named_parameters() if "sft_reference" in name or name.startswith("base."))


def test_switch_locked_until_optimizer_step(toy_policy):
    torch = pytest.importorskip("torch")
    p = toy_policy
    turn = traced_turn(p)
    optimizer = torch.optim.SGD(p.trainable_parameters(),lr=.01)
    with p.update_phase():
        loss = -p.score(turn).sum()
        with pytest.raises(RuntimeError,match="switching prohibited"):
            p.switch("reference")
        loss.backward()
        with pytest.raises(RuntimeError): p.switch("reference")
        optimizer.step()
        with pytest.raises(RuntimeError): p.switch("reference")
    p.switch("reference")


def test_actual_peft_reference_frozen_policy_update_and_adapter_roundtrip(toy_policy,tmp_path):
    """Random tiny Qwen architecture from config; NEVER from_pretrained a model."""
    from copy import deepcopy
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(vocab_size=16,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
        attention_dropout=0.,eos_token_id=2,pad_token_id=0)
    base = transformers.Qwen2ForCausalLM(config)
    original_base = deepcopy(base)
    lora = peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
    p = toy_policy
    p.model = peft.get_peft_model(base,lora)
    p.model.add_adapter("sft_reference",lora)
    params = dict(p.model.named_parameters())
    with torch.no_grad():
        for name,value in params.items():
            if ".default." in name:
                params[name.replace(".default.",".sft_reference.")].copy_(value)
    p.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    p.switch("policy")
    p._assert_equal_initial_adapters()
    members = [SimpleNamespace(turns=[traced_turn(p,generated=ids)],reward=SimpleNamespace(total=r),
        reference_logprobs=[]) for ids,r in (([5,6,2],0),([8,9,2],1))]
    frozen = weights_digest(p,frozen=True)
    before = weights_digest(p,frozen=False)
    optimizer = torch.optim.AdamW(p.trainable_parameters(),lr=.01)
    stats = update_groups(p,[RolloutGroup("peft",members)],optimizer)
    assert stats["gradient_norm"] > 0 and weights_digest(p,frozen=False) != before
    assert weights_digest(p,frozen=True) == frozen
    assert all(value.grad is None for name,value in p.model.named_parameters() if ".default." not in name)
    p.model.save_pretrained(tmp_path/"adapter",selected_adapters=["default"],save_embedding_layers=False)
    assert not (tmp_path/"adapter"/"sft_reference").exists()
    loaded = peft.PeftModel.from_pretrained(original_base,str(tmp_path/"adapter"),is_trainable=False)
    turn = members[0].turns[0]
    expected = p.score(turn).detach()
    p.model = loaded
    p.model.eval()
    assert torch.allclose(expected,p.score(turn).detach(),atol=1e-6)


def test_alignment_and_overlength_hard_gates(toy_policy):
    p = toy_policy
    turn = traced_turn(p)
    turn.old_logprobs[0] += 1
    with pytest.raises(ValueError,match="alignment gate"):
        p.validate_alignment([SimpleNamespace(turns=[turn])])
    turn.prompt_ids = [1]*8192
    with pytest.raises(ValueError,match="no truncation"):
        p.score(turn)


def test_generated_scores_capture_exact_raw_tokens():
    torch = pytest.importorskip("torch")
    b = HuggingFaceBackend.__new__(HuggingFaceBackend)
    b.config = HuggingFaceConfig("toy",do_sample=True,device="cpu",temperature=.8,top_p=1,max_new_tokens=3)
    b._torch,b._template = torch,"own-template"
    b._generation_kwargs = {"do_sample":True,"temperature":.8,"top_p":1.,"eos_token_id":[2],"pad_token_id":0,"max_new_tokens":3}
    b._transformers = SimpleNamespace(GenerationConfig=lambda **k:SimpleNamespace(**k))
    logits = [torch.tensor([[0.,1.,2.,3.]]),torch.tensor([[1.,3.,2.,0.]])]
    def generate(**kwargs):
        assert kwargs["top_k"] == 0 and kwargs["output_scores"] is True
        return SimpleNamespace(sequences=torch.tensor([[9,8,3,2]]),scores=logits)
    b._model = SimpleNamespace(generate=generate)
    b._tokenizer = SimpleNamespace(decode=lambda ids,**kw:"malformed JSON" if list(ids) == [3,2] else "bad slice")
    messages = [{"role":"user","content":"Normal task"}]
    b._prepare_inputs = lambda m:({"input_ids":torch.tensor([[9,8]]),"attention_mask":torch.ones(1,2)},2,m)
    trace = TurnTrace(**b.generate_with_trace(messages))
    trace.validate()
    assert trace.raw_output == "malformed JSON" and trace.generated_ids == [3,2]
    assert trace.ended_with_eos and not trace.token_limit
    assert trace.old_logprobs == pytest.approx([torch.log_softmax(logits[i],-1)[0,t].item() for i,t in enumerate([3,2])])


def test_batched_trace_matches_independent_rows_with_variable_prompt_lengths():
    """A random tiny Qwen is constructed locally; no model weights are downloaded."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
        attention_dropout=0.,eos_token_id=63,pad_token_id=0,
    )).eval()

    class Tokenizer:
        model_max_length = 128
        def apply_chat_template(self,messages,**kwargs):
            return messages[-1]["content"]
        def __call__(self,prompt,**kwargs):
            return {"input_ids":[int(value) for value in prompt.split()]}
        def decode(self,ids,**kwargs):
            return " ".join(map(str,ids))

    backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
    backend.config = HuggingFaceConfig("tiny",device="cpu",do_sample=True,temperature=.8,
                                       top_p=1.,max_new_tokens=4)
    backend._torch,backend._model,backend._tokenizer = torch,model,Tokenizer()
    backend._template = "fixture-template"
    backend._generation_kwargs = {"eos_token_id":[63],"pad_token_id":0}
    rows = [[{"role":"user","content":"4 5"}],
            [{"role":"user","content":"7 8 9 10 11"}],
            [{"role":"user","content":"12 13 14"}]]
    seeds = [101,202,303]
    batched = backend.generate_batch_with_trace(rows,seeds=seeds)
    serial = [backend.generate_batch_with_trace([row],seeds=[seed])[0]
              for row,seed in zip(rows,seeds,strict=True)]
    for left,right in zip(batched,serial,strict=True):
        assert left["prompt_ids"] == right["prompt_ids"]
        assert left["generated_ids"] == right["generated_ids"]
        assert left["old_logprobs"] == pytest.approx(right["old_logprobs"],abs=2e-6)
        assert left["raw_output"] == right["raw_output"]


def test_rollouts_preserve_parse_failures_and_prompt_boundary(tmp_path):
    pytest.importorskip("torch")
    task = TaskGenerator(40000).generate_task(4)
    config = GRPOConfig("fixture","fixture","fixture",device="cpu",group_size=2,max_steps=2)
    class Backend:
        count = 0
        _copy_messages = staticmethod(HuggingFaceBackend._copy_messages)
        def generate_with_trace(self,messages):
            self.count += 1
            text = "not JSON" if self.count % 2 else '{"type":"final","answer":0}'
            return asdict(TurnTrace(self._copy_messages(messages),[1,2,3],[4,5],[-.5,-.5],text,False,False))
    p = SimpleNamespace(config=config,backend=Backend(),switch=lambda _:None)
    group = collect_group(p,TaskCoordinate(40000,4,0,task.task_id),0,tmp_path)
    report = variance_report([group],.1)
    assert report["parse_error_count"] == 2 and report["equal_reward_group_count"] == 1
    assert report["invalid_action_count"] == 2 and report["invalid_tool_call_count"] == 0
    assert report["backend_error_count"] == report["execution_error_count"] == 0
    assert len((tmp_path/"rollouts.jsonl").read_text().splitlines()) == 2
    for member in group.members:
        text = json.dumps(member.turns[0].messages)
        assert all(k not in text for k in ("ground_truth","verification_spec","metadata","task_id","environment_id"))
        assert member.turns[1].messages[-1]["role"] == "user"


def test_batched_rollout_matches_serial_for_deterministic_backend(tmp_path):
    pytest.importorskip("torch")
    task = TaskGenerator(40000).generate_task(1,0)
    company,field = task.metadata["company"],task.metadata["field"]
    answer = task.ground_truth
    config = GRPOConfig("fixture","fixture","fixture",device="cpu",group_size=2,max_steps=3)

    class DeterministicBackend:
        _copy_messages = staticmethod(HuggingFaceBackend._copy_messages)
        def row(self,messages):
            if len(messages) == 2:
                text = json.dumps({"type":"tool_call","tool":"lookup_company",
                                   "arguments":{"company":company,"field":field}},separators=(",",":"))
            else:
                text = json.dumps({"type":"final","answer":answer},separators=(",",":"))
            return asdict(TurnTrace(self._copy_messages(messages),[1,2,3],[4,2],[-.5,-.5],text,True,False))
        def generate_with_trace(self,messages):
            return self.row(messages)
        def generate_batch_with_trace(self,messages_batch,**kwargs):
            return [self.row(messages) for messages in messages_batch]

    coordinate = TaskCoordinate(40000,1,0,task.task_id)
    serial_policy = SimpleNamespace(config=config,backend=DeterministicBackend(),switch=lambda _:None)
    batch_policy = SimpleNamespace(config=config,backend=DeterministicBackend(),switch=lambda _:None)
    (tmp_path/"serial").mkdir()
    (tmp_path/"batched").mkdir()
    serial = collect_group_serial(serial_policy,coordinate,0,tmp_path/"serial")
    batched = collect_group_batched(batch_policy,coordinate,0,tmp_path/"batched")
    assert [asdict(member.run) for member in batched.members] == [asdict(member.run) for member in serial.members]
    assert [[asdict(turn) for turn in member.turns] for member in batched.members] == [
        [asdict(turn) for turn in member.turns] for member in serial.members]
    assert [asdict(member.reward) for member in batched.members] == [asdict(member.reward) for member in serial.members]


def test_batched_rollout_only_generates_for_still_active_members(tmp_path):
    pytest.importorskip("torch")
    task = TaskGenerator(40000).generate_task(1,0)
    config = GRPOConfig("fixture","fixture","fixture",device="cpu",group_size=2,max_steps=2)

    class Backend:
        _copy_messages = staticmethod(HuggingFaceBackend._copy_messages)
        def __init__(self): self.batch_sizes = []
        def generate_batch_with_trace(self,messages_batch,**kwargs):
            self.batch_sizes.append(len(messages_batch))
            rows = []
            for index,messages in enumerate(messages_batch):
                final = len(self.batch_sizes) == 1 and index == 0 or len(self.batch_sizes) == 2
                text = '{"type":"final","answer":0}' if final else "not JSON"
                rows.append(asdict(TurnTrace(self._copy_messages(messages),[1,2],[3,2],[-.5,-.5],text,True,False)))
            return rows

    backend = Backend()
    policy = SimpleNamespace(config=config,backend=backend,switch=lambda _:None)
    group = collect_group_batched(policy,TaskCoordinate(40000,1,0,task.task_id),0,tmp_path)
    assert backend.batch_sizes == [2,1]
    assert [member.run.agent_steps for member in group.members] == [1,2]
    assert group.members[1].run.events[0].parse_error is not None


def test_batched_backend_failure_is_retained_for_every_active_member(tmp_path):
    pytest.importorskip("torch")
    task = TaskGenerator(40000).generate_task(1,0)
    config = GRPOConfig("fixture","fixture","fixture",device="cpu",group_size=2,max_steps=2)
    class Backend:
        _copy_messages = staticmethod(HuggingFaceBackend._copy_messages)
        def generate_batch_with_trace(self,*args,**kwargs):
            raise RuntimeError("fixture failure")
    policy = SimpleNamespace(config=config,backend=Backend(),switch=lambda _:None)
    with pytest.raises(RuntimeError,match="abort whole batch"):
        collect_group_batched(policy,TaskCoordinate(40000,1,0,task.task_id),0,tmp_path)
    errors = [json.loads(line) for line in (tmp_path/"rollout_errors.jsonl").read_text().splitlines()]
    assert len(errors) == 2
    assert {row["member_index"] for row in errors} == {0,1}
    assert all(row["run"]["termination_reason"] == "backend_error" for row in errors)


@pytest.fixture(scope="module")
def source_manifest(tmp_path_factory):
    path = tmp_path_factory.mktemp("grpo-source")/"manifest.json"
    splits = {}
    for split,seed in (("train",10000),("dev",20000),("test",30000)):
        rows = [asdict(TaskCoordinate(seed,l,1,TaskGenerator(seed).generate_task(l,1).task_id)) for l in (1,2,3,4)]
        splits[split] = {"coordinates":rows}
    path.write_text(json.dumps({"schema_version":"sft-manifest-v1","splits":splits}),encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def rl_manifest(source_manifest):
    return build_task_manifest(source_manifest)


def test_manifest_deterministic_no_sft_or_test_overlap(rl_manifest,source_manifest):
    assert rl_manifest == build_task_manifest(source_manifest)
    train,dev = coordinates(rl_manifest,"train"),coordinates(rl_manifest,"dev")
    assert len(train) == 2560 and len(dev) == 256
    assert len({c.task_id for c in train+dev}) == 2816
    assert not {c.seed for c in train}&{c.seed for c in dev}
    expressions = [expression_key(TaskGenerator(c.seed).generate_task(2,c.task_index).metadata["expression"]) for c in train+dev if c.difficulty == 2]
    assert len(expressions) == len(set(expressions))
    assert all(c.task_index % 2 for c in train+dev if c.difficulty == 2)


def test_manifest_privileged_extras_are_not_read_or_copied(rl_manifest,source_manifest,tmp_path,monkeypatch):
    from pathlib import Path
    source = json.loads(source_manifest.read_text())
    for split in source["splits"].values():
        for row in split["coordinates"]:
            row.update(messages=[{"role":"assistant","content":"oracle-secret"}],
                       ground_truth=987654321,metadata={"hidden":987654321})
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(source))
    read_text = Path.read_text
    def guarded_read(self,*args,**kwargs):
        assert self == path,"Must never open any SFT chat/oracle export"
        return read_text(self,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",guarded_read)
    assert build_task_manifest(path) == rl_manifest
    assert "oracle-secret" not in json.dumps(rl_manifest)


def test_train_schedule_deterministic_l4_heavy(rl_manifest):
    schedule = training_schedule(rl_manifest,42,20)
    assert schedule == training_schedule(rl_manifest,42,20)
    assert [sum(c.difficulty == level for c in schedule) for level in (1,2,3,4)] == [1,1,2,16]
    assert all(40000 <= c.seed < 40256 for c in schedule)
    assert derived_seed(42,0,"task",0) != derived_seed(42,0,"task",1)
