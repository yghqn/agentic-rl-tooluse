\# Agentic RL for Multi-Step Tool Use



\## 1. Project Goal



Build a reproducible environment for training and evaluating LLM agents on multi-step tool-use tasks.



The project focuses on:



\- synthetic agent task generation

\- multi-step tool interaction

\- trajectory collection

\- deterministic verification

\- agent evaluation

\- tool-use supervised fine-tuning

\- Agentic RL with GRPO

\- reward design and reward hacking analysis

\- rollout and training efficiency



Main pipeline:



Task → Environment → Agent → Trajectory → Verifier → SFT → GRPO → Independent Evaluation





\## 2. Environment



V1 uses a deterministic synthetic company database.



Each company contains hidden attributes such as:



\- revenue

\- profit

\- employees

\- growth\_rate



These values are not directly exposed in the user prompt.



The agent must obtain required information through tools.





\## 3. Available Tools



\### lookup\_company



Inputs:



\- company

\- field



Output:



\- corresponding value from the environment





\### calculator



Input:



\- mathematical expression



Output:



\- numerical result





\### list\_companies



Input:



None



Output:



\- companies available in the current environment





\## 4. Task Difficulty



\### Level 1 — Single Tool



Example:



What is the revenue of company A?



Target capability:



\- single information retrieval





\### Level 2 — Tool Selection



Examples:



Calculate 183 \* 27.



What is the profit of company B?



Target capability:



\- select the appropriate tool





\### Level 3 — Multi-Step Tool Composition



Example:



What is the revenue difference between company A and company B?



Target capabilities:



\- multiple information retrieval steps

\- tool composition

\- calculation





\### Level 4 — Planning and Comparison



Example:



Among companies A, B, and C, which has the highest profit margin,

and how many percentage points higher is it than the lowest?



Target capabilities:



\- multi-step planning

\- information gathering

\- calculation

\- comparison

\- final answer generation





\## 5. Task Representation



Each task contains:



\- task\_id

\- question

\- difficulty

\- environment\_id

\- available\_tools

\- ground\_truth

\- verification\_spec

\- metadata



Information required to solve the task must not be directly exposed in the prompt.





\## 6. Trajectory



A trajectory records the complete interaction:



User Task

→ Agent Action

→ Tool Observation

→ Agent Action

→ Tool Observation

→ ...

→ Final Answer



Each tool interaction records:



\- tool name

\- arguments

\- observation

\- execution status





\## 7. Verification



Verification should be deterministic whenever possible.



The evaluator separately measures:



\- final task correctness

\- tool selection correctness

\- argument correctness

\- invalid tool calls

\- redundant tool calls

\- number of interaction steps

\- trajectory validity



The evaluator must not require one unique reference trajectory when multiple valid solutions exist.





\## 8. Main Metrics



Primary metric:



\- Task Success Rate



Secondary metrics:



\- Tool Selection Accuracy

\- Argument Accuracy

\- Invalid Tool Call Rate

\- Redundant Tool Call Rate

\- Average Tool Calls

\- Average Agent Steps



Metrics should also be reported by difficulty level.





\## 9. Dataset Split



Tasks are divided into:



\- train

\- dev

\- test



The test set must remain isolated from training.



The project should also contain compositional held-out tasks to evaluate generalization to unseen task structures.





\## 10. Baselines



V1:



\- Direct LLM

\- Prompt-based Tool Agent



V2:



\- Tool-use SFT Agent



V3:



\- SFT + GRPO Agent





\## 11. Experiment Matrix



Compare:



Direct LLM

vs

Prompt Agent

vs

SFT Agent

vs

GRPO Agent



Main comparison dimensions:



\- task success

\- tool-use accuracy

\- argument accuracy

\- trajectory efficiency

\- performance by difficulty





\## 12. Reward Experiments



\### Experiment A — Outcome Reward



Reward primarily depends on final task success.





\### Experiment B — Shaped Reward



Possible components:



\- valid tool call

\- correct tool selection

\- correct arguments

\- final task success

\- efficiency penalty



Compare:



\- learning speed

\- final task success

\- trajectory efficiency

\- reward hacking behavior





\## 13. Failure Taxonomy



Failures are classified into:



\- wrong tool

\- wrong argument

\- invalid tool call

\- missing information

\- planning failure

\- calculation failure

\- redundant tool use

\- loop

\- premature termination

\- final answer error





\## 14. Development Order



1\. Environment

2\. Tools

3\. Task Generator

4\. Task Validator

5\. Verifier

6\. Unit Tests

7\. Evaluation Harness

8\. Prompt Agent Baseline

9\. Trajectory Collection

10\. SFT Dataset Construction

11\. Tool-use SFT

12\. GRPO

13\. Reward Analysis

14\. Training / Rollout Optimization





\## 15. Core Principles



Do not train before the benchmark and verifier are validated.



Training reward and independent evaluation metrics must remain conceptually separate.



Do not expose hidden environment state in task prompts.



Prefer deterministic verification over subjective evaluation whenever possible.

