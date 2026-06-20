# AdaptiveThink-RL

- **Problem Statement Number** - PS06
- **Problem Statement Title** - Enhancing Reasoning in Small Language Models (SLMs, ≤7B) using Reinforcement Learning — Samsung EnnovateX AX Hackathon 2026
- **Team name** - StateZero
- **Team members (Names)** - Jeeth Bhavesh Kataria, Ojasvi Poonia
- **Institute/College Name** - Ramaiah Institute of Technology, Bengaluru, Karnataka – 560054
- **Final Presentation Google Drive Link** - <!-- TODO: add Google Drive / Slides link -->
- **Full Submission Demo Video Link** - <!-- TODO: add YouTube/Drive demo video link -->
- **Setup & Result Reproducibility Video Link** - <!-- TODO: add YouTube/Drive reproducibility video link -->

---

## Solution Summary

Small language models can reason, but they reason *wastefully* — burning a full chain-of-thought on a question they could answer in one step, and sometimes still getting hard questions wrong. **AdaptiveThink-RL** makes a sub-2B SLM reason both **more accurately** and **more efficiently** with **reinforcement learning only — no SFT distillation, and no LLM API anywhere** (a hard hackathon constraint).

The system has two coupled layers built on **`Qwen/Qwen2.5-1.5B-Instruct`** (Apache-2.0, chosen for genuine +5% headroom, single-GPU speed, and an open license):

1. **Accuracy core — Dr.GRPO + RLVR.** We train the base model with **Dr.GRPO** (constant length-normalization + no per-group std division, killing GRPO's length/difficulty biases) using a **rule-based, verifiable reward** (RLVR): exact-match on GSM8K numerics, AQuA letters, and StrategyQA booleans, plus a light `<think>/<answer>` format reward. No teacher, no preference model, no API — just a robust answer-matcher in the reward loop.

2. **Efficiency / adaptive-compute layer — CCDD.** Our novelty, **Competence-Calibrated Difficulty Self-Distillation (CCDD)**, defines each question's difficulty as `1 − (the base model's OWN empirical solve-rate over K rollouts)`. This is **fully API-free and teacher-free** — the difficulty signal is distilled from the model's own competence, not guessed by an external LLM. CCDD is used (a) as a **GRPO curriculum filter** that drops zero-gradient groups (items solved 0/K or K/K), and (b) to train a **tiny difficulty verifier** that gates `think` / `no_think` adaptive compute on-device.

The RL-trained adapter is QLoRA 4-bit, merged, and exported to **GGUF Q4_K_M** (llama.cpp) for on-device inference on a Samsung Galaxy. The whole thing trains in a single ~24 h run on **one RTX A6000 (48 GB)**.

> **Honest positioning vs prior work.** Unlike **AdaptThink** (internal model confidence) or **CODA** (group-rollout pass-rate consumed online), CCDD's difficulty is a *pre-computed, self-distilled* signal reused as both a curriculum filter and an on-device routing gate. We build on **TRL/Unsloth/vLLM/llama.cpp**; the new contribution is CCDD and its dual use, not a new RL algorithm.

### KPI Results

**Goal (PS06):** improve **≥ 2 of 3** benchmarks by **≥ +5%** over the **same base model** baseline, while keeping latency/efficiency competitive. Targets: lift **GSM8K** + **StrategyQA**; **maintain MMLU** (MMLU is eval-only and is *never* trained on).

> All numbers below are reported as Pass@1, mean over eval seeds `{0,1,2}` with the **identical** harness for baseline and trained model (same few-shot, same prompt template). The training run has **not been executed yet**, so measured cells are placeholders — see `results/figures/kpi_table.md` (written by `scripts/05_eval.sh`) after the run.

| Benchmark        | Role     | Baseline (Qwen2.5-1.5B-Instruct) | Ours (Dr.GRPO + CCDD) | Δ        | Target  | Met? |
|------------------|----------|----------------------------------|-----------------------|----------|---------|------|
| GSM8K            | Improve  | <!-- TODO -->                    | <!-- TODO -->         | <!--TODO-->| ≥ +5%  | <!--TODO--> |
| StrategyQA       | Improve  | <!-- TODO -->                    | <!-- TODO -->         | <!--TODO-->| ≥ +5%  | <!--TODO--> |
| MMLU             | Maintain | <!-- TODO -->                    | <!-- TODO -->         | <!--TODO-->| ≥ floor| <!--TODO--> |
| Avg tokens / Q   | Efficiency | <!-- TODO -->                  | <!-- TODO -->         | <!--TODO-->| ↓ lower | <!--TODO--> |

<!-- TODO: fill the table above from results/figures/kpi_table.md after the 24h run. Do NOT fabricate. -->

---

### Project Artefacts

- **Technical Documentation** - [`docs/technical.md`](docs/technical.md) (stack, architecture, OSS libraries, design decisions), [`RUNBOOK.md`](RUNBOOK.md) (staged copy-paste commands), and [`WHERE_TO_RUN.md`](WHERE_TO_RUN.md) (environment-specific run instructions).
- **[Important]** [`docs/ax.md`](docs/ax.md) — agentic AI usage: open-weight model usage, agentic development workflow, tool chaining, reasoning/planning pipeline, memory/context handling, and what worked vs what did not.
- **Source Code** - [`src/adaptivethink/`](src/adaptivethink/) — all training, evaluation, and deployment code (see the stage map below).
- **Models Used** (open-weight only)
  - [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — base reasoning SLM, Apache-2.0 (the trainee; 3B / 7B are a one-flag `--model` switch — all Qwen2.5 share ChatML + the same 7 LoRA target modules).
  - *No teacher model and no LLM API are used anywhere* — the difficulty signal (CCDD) and the reward (RLVR) are both computed from the base model itself.
- **Models Published**
  - `statezero/adaptivethink-rl-1.5b-grpo-lora` — RL-trained QLoRA adapter <!-- TODO: publish + link after training -->
  - `statezero/adaptivethink-rl-1.5b-Q4_K_M-gguf` — merged on-device GGUF <!-- TODO: publish + link after training -->
- **Datasets Used** (links + licenses)
  - [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) — grade-school math, **MIT** (train + eval).
  - [`ChilleD/StrategyQA`](https://huggingface.co/datasets/ChilleD/StrategyQA) (train labels) and [`wics/strategy-qa`](https://huggingface.co/datasets/wics/strategy-qa) (held-out eval) — multi-hop yes/no reasoning, **Apache-2.0 / MIT** (kept train/eval disjoint to prevent leakage).
  - [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) — algebraic word problems (MC), **Apache-2.0** (train pool).
  - [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) — knowledge MC, **MIT** — **eval-only / maintain** (NEVER trained on).
- **Datasets Published**
  - `statezero/ccdd-self-difficulty` — teacher-free CCDD self-difficulty labels (`{question, answer, difficulty, solve_rate, source}`) over the train pool, Apache-2.0 <!-- TODO: publish + link after training -->

---

### Architecture

```
                         Question
                            │
        ┌───────────────────┴────────────────────┐
        │                                         │
   TRAIN-TIME (RL)                          INFERENCE (on-device)
        │                                         │
        ▼                                         ▼
┌──────────────────────────┐         ┌──────────────────────────┐
│ CCDD self-difficulty pass │        │  Tiny difficulty verifier │
│ d = 1 − base solve-rate@K │        │  (trained on CCDD labels) │
│ (API-free, no teacher)    │        │      outputs d ∈ [0,1]    │
└────────────┬─────────────┘         └────────────┬─────────────┘
             │ curriculum filter                  │ d
             │ (drop 0/K and K/K groups)          ▼
             ▼                          ┌──────────────────────────┐
┌──────────────────────────┐           │  Adaptive-compute gate    │
│   Dr.GRPO + RLVR core     │           │  think  vs  no_think      │
│ loss=dr_grpo, KL off,     │           └────────────┬─────────────┘
│ reward = exact-match      │                  ┌──────┴──────┐
│   + light <think>/<answer>│              <think>       <no_think>
└────────────┬─────────────┘             Full CoT      Direct answer
             │                                  └──────┬──────┘
             ▼                                         ▼
   QLoRA 4-bit adapter ──merge──► GGUF Q4_K_M ──► Samsung Galaxy (llama.cpp)
```

**Stage map (code ↔ config):**

| Stage | Code | Config |
|---|---|---|
| CCDD self-difficulty (novelty, API-free) | [`src/adaptivethink/rl/self_difficulty.py`](src/adaptivethink/rl/self_difficulty.py) | `configs/strategy1.yaml` (`self_difficulty_file`) |
| Dr.GRPO RLVR (accuracy core) | [`src/adaptivethink/rl/drgrpo_train.py`](src/adaptivethink/rl/drgrpo_train.py), [`src/adaptivethink/rl/rewards.py`](src/adaptivethink/rl/rewards.py) | [`configs/strategy1.yaml`](configs/strategy1.yaml) |
| Robust answer matcher (shared train + eval) | [`src/adaptivethink/router/reward.py`](src/adaptivethink/router/reward.py) | — |
| Adaptive-compute router / verifier | [`src/adaptivethink/router/`](src/adaptivethink/router/), [`src/adaptivethink/verifier/`](src/adaptivethink/verifier/) | `configs/grpo_router.yaml`, `configs/verifier_distill.yaml` |
| Quantize + on-device inference | [`src/adaptivethink/quantize/`](src/adaptivethink/quantize/), [`src/adaptivethink/inference/`](src/adaptivethink/inference/) | — |
| Eval harness (multi-seed + CIs, KPI/Pareto) | [`eval/run_benchmarks.py`](eval/run_benchmarks.py), [`eval/plots.py`](eval/plots.py) | — |
| Optional — Test-Time RL ablation | [`src/adaptivethink/ttrl/`](src/adaptivethink/ttrl/) | `configs/ttrl_ablation.yaml` |

### Quick Start

```bash
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink-RL

# Day 0 — one-time setup (pinned, verified-compatible stack):
#   vLLM 0.19.1 / torch 2.10 / TRL 0.24 / Unsloth / transformers 4.57.6; lm-eval for benchmarks.
./run.sh setup
source .venv/bin/activate

# Smoke test the whole wiring (2 GRPO steps, ~5–7 GB VRAM):
./run.sh smoke
```

The full single-run pipeline (≈24 h on one RTX A6000, ~5–6 h of which is training):

```bash
bash scripts/run_all_24h.sh
# Stages: baseline eval → CCDD self-difficulty → Dr.GRPO RLVR → (optional verifier)
#         → multi-seed eval (seeds 0,1,2) + KPI table → GGUF Q4_K_M export.
# Every stage is idempotent/resumable; re-running continues from the first incomplete stage.
```

`scripts/05_eval.sh` runs the **identical** `lm-eval`/harness config for baseline and trained model and writes the KPI delta table to **`results/figures/kpi_table.md`** (flagging whether ≥ +5% on ≥ 2 of GSM8K / StrategyQA / MMLU is met) plus Pareto charts (accuracy vs compute). All hyperparameters live in [`configs/strategy1.yaml`](configs/strategy1.yaml). See [`RUNBOOK.md`](RUNBOOK.md) for the staged, copy-paste day-by-day commands and per-flag tuning.

---

#### Final Presentation

The Google Drive link above points to the slide deck. The deck leads with the headline KPI ("+X% GSM8K, +Y% StrategyQA, MMLU maintained, ~Z% fewer tokens"), then an ablation table isolating each reward / stability / CCDD component, a Pareto chart (accuracy vs tokens/latency), multi-seed runs with 95% confidence intervals, reward & entropy curves proving training stability, one clean qualitative `<think>/<answer>` example, and on-device latency from the quantized GGUF.

#### Full Submission Demo Video

The demo video walks through the problem, the two-layer architecture (Dr.GRPO + RLVR accuracy core and the CCDD adaptive-compute layer), the headline KPI improvement over the same base model, and a live on-device inference clip of the GGUF Q4_K_M model on a Samsung Galaxy answering both an easy (`no_think`) and a hard (`think`) question.

#### Setup & Result Reproducibility Video

The reproducibility video shows a clean clone running `./run.sh setup` → `./run.sh smoke` → `bash scripts/run_all_24h.sh` end to end on a single RTX A6000, then opens `results/figures/kpi_table.md` to confirm the reported deltas against the same-base-model baseline using the identical eval harness.

---

### Attribution

AdaptiveThink-RL is built on open-source foundations. Our **new contribution is CCDD** (Competence-Calibrated Difficulty Self-Distillation) — an API-free, teacher-free difficulty signal distilled from the model's own solve-rate, reused as both a GRPO curriculum filter and an on-device think/no_think gate — and the system integration around it. We do **not** claim a new RL algorithm.

| Project | Link | Our use / what is new |
|---|---|---|
| Dr.GRPO | [arxiv:2503.20783](https://arxiv.org/abs/2503.20783) | RL loss (constant length-norm, no per-group std div); we apply it to a sub-2B dense SLM with RLVR rewards |
| DeepSeek-R1 (RL recipe) | [arxiv:2501.12948](https://arxiv.org/abs/2501.12948) | Rule-based RLVR + format reward inspiration |
| HuggingFace TRL | [github.com/huggingface/trl](https://github.com/huggingface/trl) | `GRPOTrainer` / `GRPOConfig` |
| Unsloth | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | Memory-efficient QLoRA GRPO on a single GPU |
| vLLM | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | Fast colocated rollout generation (CCDD + GRPO) |
| llama.cpp | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF Q4_K_M quantization + on-device inference |
| AdaptThink | [arxiv:2505.13417](https://arxiv.org/abs/2505.13417) | Adaptive-thinking baseline; CCDD differs — difficulty is self-distilled and pre-computed, not internal confidence |
| CODA | [arxiv:2603.08659](https://arxiv.org/abs/2603.08659) | Difficulty-gated baseline; CCDD's `d` is a reusable self-distilled label, not an online group pass-rate |
| TTRL | [arxiv:2504.16084](https://arxiv.org/abs/2504.16084) | Optional, clearly-labeled test-time RL ablation only — never the headline |
| Qwen2.5 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Open-weight base model (Apache-2.0) |

**License:** Apache-2.0 (see [`LICENSE`](LICENSE)). Copyright 2026 StateZero (Jeeth Bhavesh Kataria, Ojasvi Poonia).
