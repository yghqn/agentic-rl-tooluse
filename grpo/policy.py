"""Shared frozen base with independent policy/reference LoRA adapters."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from agent.hf_backend import HuggingFaceBackend, HuggingFaceConfig
from grpo.schemas import GRPOConfig, TurnTrace


class AlignmentError(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Generation/teacher-force alignment gate failed: {result}")


def adapter_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json","adapter_model.safetensors"):
        digest.update(name.encode())
        digest.update((path/name).read_bytes())
    return digest.hexdigest()


class GRPOPolicy:
    def __init__(self, config: GRPOConfig):
        import torch
        self.config = config
        self.backend = HuggingFaceBackend(HuggingFaceConfig(
            config.model,revision=config.revision,device=config.device,dtype=config.dtype,
            adapter_path=config.sft_adapter,tokenizer_path=config.tokenizer_path,
            cache_dir=config.cache_dir,local_files_only=config.local_files_only,
            do_sample=True,temperature=config.temperature,top_p=1.0,max_new_tokens=config.max_new_tokens))
        self.model = self.backend._model
        self.tokenizer = self.backend._tokenizer
        self.source_record = json.loads((Path(config.sft_adapter).parent/"run_config.json").read_text(encoding="utf-8"))
        if self.source_record["schema_version"] != "lora-sft-run-v1":
            raise ValueError("GRPO starting point must be the SFT adapter, not another RL experiment")
        if hashlib.sha256(Path(config.sft_manifest).read_bytes()).hexdigest() != self.source_record["dataset"]["manifest_sha256"]:
            raise ValueError("SFT coordinate manifest differs from the starting adapter's training provenance")
        self.source_hash = adapter_hash(Path(config.sft_adapter))
        self.policy_adapter = "default"
        self.reference_adapter = "sft_reference"
        self.model.load_adapter(config.sft_adapter,adapter_name=self.reference_adapter,is_trainable=False,local_files_only=True)
        # HF evaluation casts the first adapter with the base dtype; PEFT loading
        # a second adapter promotes LoRA tensors to FP32. Clone the effective,
        # already-loaded SFT values, not a differently rounded second disk load.
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name,p in params.items():
                if ".default." in name and (".lora_A." in name or ".lora_B." in name):
                    params[name.replace(".default.",".sft_reference.")].copy_(p)
        self._locked = False
        for module in self.model.modules():
            if isinstance(module,torch.nn.Dropout):
                module.p = 0.0
            if getattr(module,"attention_dropout",0) != 0:
                raise ValueError("V1 requires zero attention dropout for sampling/scoring consistency")
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.switch("policy")
        self._assert_equal_initial_adapters()

    def _assert_equal_initial_adapters(self):
        import torch
        params = dict(self.model.named_parameters())
        for name,p in params.items():
            if f".{self.policy_adapter}." in name and (".lora_A." in name or ".lora_B." in name):
                reference = name.replace(f".{self.policy_adapter}.",f".{self.reference_adapter}.")
                if reference not in params or not torch.equal(p,params[reference]):
                    raise ValueError("Policy is not an exact copy of SFT reference")

    def switch(self, role: str):
        if self._locked:
            raise RuntimeError("Adapter switching prohibited throughout policy forward/backward/step")
        if role not in {"policy","reference"}:
            raise ValueError("Unknown adapter role")
        self.model.set_adapter(self.policy_adapter if role == "policy" else self.reference_adapter)
        for name,p in self.model.named_parameters():
            p.requires_grad = role == "policy" and f".{self.policy_adapter}." in name and (
                ".lora_A." in name or ".lora_B." in name)
        self.model.eval()

    @contextmanager
    def update_phase(self):
        self.switch("policy")
        self._locked = True
        self.model.train()  # Enable gradient checkpointing; ALL dropout remains zero.
        self.model.config.use_cache = False
        try:
            yield
        finally:
            self._locked = False
            self.model.eval()

    def score(self, turn: TurnTrace):
        import torch
        turn.validate()
        if len(turn.prompt_ids) + len(turn.generated_ids) > self.config.max_scoring_length:
            raise ValueError("Scoring length exceeds hard resource budget; no truncation or selective dropping")
        device = next(self.model.parameters()).device
        # Last generated ID is a target, not an extra input. Last N logits predict all N samples.
        ids = torch.tensor([turn.prompt_ids+turn.generated_ids[:-1]],device=device,dtype=torch.long)
        output = self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                            logits_to_keep=len(turn.generated_ids))
        logits = output.logits[0].float()/self.config.temperature
        targets = torch.tensor(turn.generated_ids,device=device,dtype=torch.long)
        if logits.shape[0] != len(targets):
            raise ValueError("Assistant logits/targets alignment failure")
        return logits.gather(1,targets[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)

    def validate_alignment(self, members) -> dict:
        import torch
        self.switch("policy")
        worst,count = 0.0,0
        with torch.no_grad():
            for member in members:
                for turn in member.turns:
                    scored = self.score(turn).cpu()
                    old = torch.tensor(turn.old_logprobs)
                    if not torch.isfinite(scored).all():
                        raise ValueError("Non-finite logprobs")
                    worst = max(worst,(scored-old).abs().max().item())
                    count += len(old)
        result = {"sampled_token_count":count,"max_logprob_error":worst,
                  "tolerance":self.config.alignment_tolerance,"passed":worst <= self.config.alignment_tolerance}
        if not result["passed"]:
            raise AlignmentError(result)
        return result

    def cache_reference(self, members):
        import torch
        self.switch("reference")
        with torch.no_grad():
            for member in members:
                member.reference_logprobs = [self.score(t).cpu().tolist() for t in member.turns]
        # No graphs exist here. update_phase switches ONCE and locks until step finishes.

    def trainable_parameters(self):
        return [p for name,p in self.model.named_parameters() if f".{self.policy_adapter}." in name
                and (".lora_A." in name or ".lora_B." in name)]
