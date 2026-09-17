# Agentic RL for Tool-Using LLMs

An end-to-end project for studying post-training methods for tool-using language model agents.

This project builds a deterministic multi-tool environment, evaluates a prompt-only baseline, performs LoRA supervised fine-tuning on oracle-generated trajectories, and then applies multi-turn GRPO to improve difficult agentic reasoning tasks.

The main result is that SFT successfully teaches reliable tool use and multi-turn interaction, while GRPO further improves high-level decision making on difficult multi-step tasks without degrading previously learned tool-use capabilities.

---

## Overview

Modern LLM agents need to do more than generate text. They must decide:

- which tool to call,
- what arguments to provide,
- how to use tool observations,
- whether more actions are needed,
- and when to produce a final answer.

The full pipeline is:

```text
Task Generation
      ↓
Deterministic Environment
      ↓
Prompt-only Baseline
      ↓
Trajectory Collection
      ↓
Verifier / Evaluation
      ↓
Oracle SFT Data
      ↓
LoRA SFT
      ↓
Multi-turn GRPO
      ↓
Reward Ablation
      ↓
Frozen Test Evaluation
      ↓
Failure Analysis
```

The project focuses on three questions:

1. Can supervised fine-tuning teach a small model reliable tool-use behavior?
2. Can reinforcement learning improve difficult multi-step agent tasks after SFT?
3. Does shaped reward outperform simple outcome-only reward in this setting?

---

## Why This Project

Tool-using LLM agents are different from fixed software pipelines.

In a fixed workflow, the sequence of operations is predefined by the programmer. In an agentic setting, the model must dynamically decide which tool to call, what arguments to use, how to interpret observations, whether more actions are needed, and when to stop.

A pretrained language model may understand a task semantically, but still fail to follow a structured action protocol, maintain a reliable multi-step interaction, or make the correct final decision after several tool calls.

This project therefore focuses on training and evaluating the policy that controls tool use, rather than only checking whether the model can produce the correct final text answer.

---

## Task Environment

The project uses a deterministic synthetic company database containing companies A-J.

Each company has several structured attributes, including:

- revenue,
- profit,
- number of employees,
- growth rate.

The environment is generated from fixed random seeds, so the same seed always produces the same underlying data. This makes experiments reproducible and allows training, development, and test environments to be separated cleanly.

The agent cannot directly access the hidden database. It must interact with the environment through tools.

### Tools

The agent can use three tools:

- `lookup_company`: retrieve one attribute of a company
- `calculator`: safely evaluate arithmetic expressions
- `list_companies`: retrieve available company names

All tool calls follow a strict structured JSON action protocol.

The calculator uses restricted AST-based evaluation rather than arbitrary Python execution.

---

## Task Difficulty

Tasks are divided into four levels of increasing difficulty.

### L1 — Single-step retrieval

Retrieve one attribute from one company.

Example:

```text
What is Company B's revenue?
```

### L2 — Simple composition

Combine retrieved values with simple arithmetic or list operations.

### L3 — Multi-step arithmetic reasoning

Retrieve a company's profit and revenue, then compute:

```text
profit_margin = profit / revenue
```

This requires multiple tool calls followed by calculator use.

### L4 — Multi-candidate aggregation

Compute the profit margins of multiple companies and return the company with the highest margin.

A typical L4 trajectory requires:

```text
3 profit lookups
3 revenue lookups
3 calculator calls
1 final comparison
```

L4 stresses longer-horizon tool use, intermediate-result tracking, and final aggregation.

---

## Agent Interaction Loop

The model only receives the public task description and tool schema. Hidden answers remain inside the environment and are used only by the verifier.

At each step, the policy generates a structured action. The parser converts the model output into an executable action.

If the action is a tool call, the environment executes the tool and returns an observation to the model. This process repeats until the model emits a final answer.

```text
Task
  ↓
Public Task View
  ↓
LLM Policy
  ↓
Structured Action Parser
  ↓
Tool Call
  ↓
Environment
  ↓
Observation
  ↓
LLM Policy
  ↓
...
  ↓
Final Answer
  ↓
Verifier
```

The verifier evaluates the final outcome instead of requiring an exact reference trajectory.

This allows different valid tool-use paths to receive credit.

---

## Prompt-only Baseline

The first baseline uses the pretrained instruction model directly with a carefully specified tool-use protocol.

Base model:

```text
Qwen2.5-0.5B-Instruct
```

The model receives:

- the task,
- tool descriptions,
- strict action formatting rules,
- previous tool observations.

No parameter updates are performed.

### Result

The prompt-only model achieves:

```text
0 / 128
```

on the frozen evaluation set.

The main problem is unreliable adherence to the structured agent protocol rather than simply a lack of task understanding.

This motivates supervised post-training.

---

## Supervised Fine-Tuning

An oracle policy generates valid multi-turn trajectories for all task types.

The oracle follows the known task rules to construct correct tool-use traces, but does not expose hidden chain-of-thought reasoning.

Training examples contain only observable agent interactions:

```text
system
user
assistant tool call
tool observation
assistant tool call
tool observation
...
assistant final answer
```

### Dataset

The SFT dataset uses disjoint deterministic environment seeds:

```text
Train: 1024 tasks
Dev:    128 tasks
Test:   128 tasks
```

The frozen test split is never used for training or checkpoint selection.

### Training

SFT uses LoRA with assistant-only loss.

Non-assistant tokens are masked with:

```text
label = -100
```

LoRA is applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Configuration:

```text
LoRA rank:        8
LoRA alpha:       16
LoRA dropout:     0
Trainable params: ~4.4M
```

SFT teaches the model the structured action protocol, tool selection, argument generation, multi-turn interaction, and correct stopping behavior.

---

## GRPO

After SFT, the project applies a custom multi-turn Group Relative Policy Optimization implementation.

The RL policy starts from the SFT LoRA checkpoint, while a frozen copy of the SFT policy is used as the reference policy.

For each task, multiple complete Agent trajectories are sampled.

Configuration:

```text
Group size:                8
Prompts per optimizer step: 4
Optimizer steps:           200
Total rollouts:            6400
Learning rate:             5e-6
Sampling temperature:      0.8
Top-p:                     1.0
KL coefficient:            0.02
Clipping epsilon:          0.2
```

For each prompt:

```text
1 prompt
→ 8 rollouts
→ rewards
→ group-relative advantages
```

For each optimizer step:

```text
4 prompts
× 8 rollouts
= 32 trajectories
```

Over 200 optimizer steps:

```text
4 × 8 × 200 = 6400 rollouts
```

### Group-relative Advantages

Rewards are normalized relative to other trajectories generated for the same task.

Conceptually:

```text
advantage =
(reward - group_mean)
/
max(group_std, std_floor)
```

Trajectories with positive advantages are encouraged, while trajectories with negative advantages are suppressed.

The KL penalty keeps the RL policy from drifting too far from the SFT reference policy.

---

## Reward Variants

Two GRPO reward functions are evaluated.

### Outcome-only Reward

The simplest reward depends only on final task success:

```text
correct final answer → reward = 1
incorrect final answer → reward = 0
```

### Shaped Reward

The shaped reward additionally considers:

- final correctness,
- evidence completeness,
- valid completion,
- invalid actions,
- redundant actions,
- missing final answers,
- tool-use efficiency.

The reward is:

```text
R =
    1.00 * success
  + 0.15 * evidence_completeness
  + 0.02 * valid_completion
  - 0.10 * min(invalid_actions, 3)
  - 0.02 * min(redundant_actions, 3)
  - 0.05 * missing_final_answer
  - 0.02 * success * normalized_tool_cost
```

---

## Multi-turn Rollout Batching

The initial rollout implementation generated trajectories serially.

To improve throughput, active trajectories at the same interaction turn are dynamically batched.

The batched rollout engine uses:

- left padding,
- KV cache,
- independent trajectory RNG,
- preserved token-level log probabilities.

On an RTX 4090, one L4 group with 8 trajectories takes approximately:

```text
~10.5 seconds
```

This optimization improves GPU utilization while preserving rollout semantics.

---

## Numerical Alignment

The custom GRPO implementation needs rollout-time token log probabilities to align closely with training-time recomputation.

BF16 introduced alignment mismatches under the project's strict numerical checks, so formal GRPO training was performed in FP32.

The FP32 alignment probe achieved a maximum error of approximately:

```text
4.22e-05
```

This provided a more reliable foundation for the custom GRPO loss.

---

## Development Curve

The GRPO development-set performance improves steadily during training.

```text
Before GRPO:
214 / 256 = 83.59%
L4: 22 / 64 = 34.38%

Step 50:
223 / 256 = 87.11%
L4: 31 / 64 = 48.44%

Step 100:
231 / 256 = 90.23%
L4: 39 / 64 = 60.94%

Step 150:
237 / 256 = 92.58%
L4: 45 / 64 = 70.31%

Step 200:
237 / 256 = 92.58%
L4: 45 / 64 = 70.31%
```

L1-L3 remain at 100% throughout the evaluated checkpoints.

The final checkpoint is selected using development performance only, without using the frozen test set.

---

## Results

All methods are evaluated on the same frozen 128-task test set, with 32 tasks per difficulty level.

| Method | Overall | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| Prompt-only | 0 / 128 (0.0%) | - | - | - | - |
| SFT | 111 / 128 (86.7%) | 32 / 32 (100%) | 32 / 32 (100%) | 32 / 32 (100%) | 15 / 32 (46.9%) |
| GRPO — outcome-only | 121 / 128 (94.5%) | 32 / 32 (100%) | 32 / 32 (100%) | 32 / 32 (100%) | 25 / 32 (78.1%) |
| GRPO — shaped reward | 121 / 128 (94.5%) | 32 / 32 (100%) | 32 / 32 (100%) | 32 / 32 (100%) | 25 / 32 (78.1%) |

### Main Findings

SFT successfully teaches reliable structured tool use.

After SFT:

```text
L1: 100%
L2: 100%
L3: 100%
L4: 46.9%
```

The remaining weakness is concentrated in the long-horizon L4 tasks.

GRPO improves:

```text
Overall:
86.7% → 94.5%

L4:
46.9% → 78.1%
```

while preserving 100% accuracy on L1-L3.

The 10 additional correct frozen-test tasks after GRPO all come from L4.

This suggests that GRPO primarily improves the difficult high-level decision behavior rather than relearning simpler tool-use skills.

---

## Tool-use Reliability

The final GRPO policy preserves clean tool-use behavior on the frozen test set:

```text
Tool selections:    448 / 448 correct
Tool arguments:     448 / 448 correct
Parse errors:       0
Invalid tool calls: 0
Execution errors:   0
Redundant calls:    0
Valid trajectories: 128 / 128
```

The improvement therefore does not come from noisier or more aggressive tool use.

---

## Reward Ablation

Outcome-only and shaped-reward GRPO reach exactly the same frozen-test performance:

```text
Overall: 121 / 128
L4:      25 / 32
```

The two final LoRA adapters have different parameter hashes, indicating different optimization trajectories.

However, their greedy frozen-test trajectories are byte-identical across all 128 tasks.

Therefore:

> Under the current benchmark, model size, training budget, and decoding strategy, reward shaping does not provide an observable final-performance improvement over the simpler outcome-only reward.

This negative ablation result shows that additional reward complexity is not automatically beneficial.

---

## Failure Analysis

After GRPO, only 7 frozen-test tasks remain incorrect.

All 7 failures are L4 tasks.

Importantly, these failures are not caused by tool-use errors.

In every failed trajectory, the agent successfully:

1. selects the correct tools,
2. supplies correct arguments,
3. retrieves all required company values,
4. computes all candidate profit margins correctly,
5. completes the full multi-step trajectory.

The error occurs only during final aggregation.

For example:

```text
Company D margin = 0.3083
Company F margin = 0.0772
Company J margin = 0.1015
```

The model correctly computes all three values but still returns Company J instead of Company D.

The remaining bottleneck can therefore be summarized as:

```text
Evidence acquisition   ✓
Tool selection         ✓
Tool arguments         ✓
Arithmetic             ✓
Multi-step execution   ✓
Final aggregation      ✗
```

A recurring qualitative pattern is over-selection of recently processed candidates, suggesting a possible recency or last-candidate bias during final aggregation.

Compared with SFT, GRPO reduces the number of L4 failures from:

```text
17 → 7
```

but does not fully eliminate the final aggregation bottleneck.

---

## Training Diagnostics

During each formal GRPO training run, one malformed sampled action appears among 6400 rollouts.

An example is:

```text
"type": "tool Call"
```

instead of:

```text
"type": "tool_call"
```

Because the training health gate requires completely valid generated protocol behavior, the formal training report is marked:

```text
passed = False
```

This does not mean that optimization failed.

The strict gate is intentionally preserved rather than weakened after observing the result.

The selected models still show strong development and frozen-test performance, and frozen greedy evaluation contains zero protocol errors.

---

## Reproducibility

The environment is deterministic and seed-based.

The project uses separate seeds for:

```text
SFT train
SFT dev
Frozen test
RL train
RL dev
```

The frozen test set is never used for:

- model training,
- reward tuning,
- checkpoint selection.

GRPO checkpoints are selected using development performance only.

This design reduces evaluation leakage and makes the experimental comparison reproducible.

---

## Repository Structure

```text
agent/             Agent loop, action parsing, prompts, and Hugging Face backend
environment/       Deterministic company database and executable tools
tasks/             Task schemas, generation, and validation
evaluation/        Answer verification, metrics, and failure analysis
sft/               Oracle trajectories, datasets, preprocessing, and SFT training
grpo/              Rollouts, rewards, policy, loss, and GRPO training
scripts/           Entry points for data generation, training, evaluation, and benchmarks
configs/           Base configuration
tests/             Automated tests
PROJECT_SPEC.md    Project specification
requirements*.txt  Python dependencies
```

---

## Key Takeaways

This project demonstrates a complete post-training pipeline for tool-using LLM agents.

The main findings are:

1. Prompting alone is insufficient for reliable structured tool use with the 0.5B model.
2. SFT effectively teaches the tool protocol, argument generation, and multi-turn execution.
3. Difficult long-horizon aggregation remains weak after SFT.
4. GRPO substantially improves L4 performance while preserving simpler capabilities.
5. Reward shaping does not outperform outcome-only reward in this experiment.
6. The remaining failures are concentrated in final aggregation rather than tool execution.

---

## Future Work

Possible extensions include:

- explicit aggregation tools,
- larger candidate sets,
- longer-horizon tasks,
- stochastic environments,
- noisy or partially observable tools,
- larger base models,
- curriculum-based RL,
- alternative group-based policy optimization methods,
- targeted training for final aggregation,
- further analysis of possible recency bias.