# Technical Documentation — AdaptiveThink-RL

> **Team StateZero** · PS06 *Enhancing Reasoning in SLMs (≤7B) using Reinforcement Learning* (Samsung EnnovateX AX Hackathon 2026)
> Jeeth Bhavesh Kataria · Ojasvi Poonia — Ramaiah Institute of Technology, Bengaluru

This document describes **what the code in this repository actually does**: the
technical stack, the OSS we build on, the three-stage architecture, the exact
reward and curriculum mechanisms, how to install the pinned dependency stack,
and how to run every stage. Numbers from the training run do not exist yet —
KPI cells are marked `<!-- TODO: fill from results/ after the run -->` and the
text below states **targets and methodology**, never fabricated measurements.

---

## 1. Technical Stack

The single source of truth for versions is [`requirements.lock`](../requirements.lock),
reproduced exactly by [`./run.sh setup`](../run.sh). The conflict this stack
resolves: latest vLLM (≥0.20) pins `torch==2.11`, but `unsloth==2026.6.7`
requires `torch<2.11`, so a naive `pip install unsloth vllm` is unsolvable.
vLLM `0.19.1` (torch 2.10.0) is the highest vLLM that Unsloth allows, so vLLM is
pinned first and everything else is resolved into its window.

| Component | Pin | One-line role |
|---|---|---|
| **Python** | `>=3.10,<3.14` (3.12 preferred) | Runtime; `pyproject.toml` `requires-python>=3.10`. |
| **PyTorch** | `torch==2.10.0` (+ `torchvision==0.25.0`, `torchaudio==2.10.0`) | Tensor/autograd backend; CUDA 12.x wheels. |
| **vLLM** | `vllm==0.19.1` | Fast colocated GRPO rollouts + CCDD self-difficulty sampling; **the strictest torch pinner**, installed first. |
| **TRL** | `trl==0.24.0` | `GRPOTrainer` / `GRPOConfig` — the Dr.GRPO RL loop. |
| **Unsloth** | `unsloth==2026.6.7` (+ `unsloth_zoo>=2026.6.5`) | Memory-efficient 4-bit QLoRA + fused kernels so sub-3B GRPO fits one GPU. |
| **Transformers** | `transformers==4.57.6` | Model/tokenizer loading; HF generation fallback. |
| **PEFT** | `peft==0.19.1` | LoRA adapters (rank 16) on the 7 Qwen2.5 projection modules. |
| **Accelerate** | `accelerate==1.14.0` | Device placement / training loop plumbing for TRL. |
| **bitsandbytes** | `bitsandbytes==0.49.2` | NF4 4-bit quantisation (QLoRA) for the training fallback path. |
| **datasets** | `datasets==3.6.0` | GSM8K / StrategyQA / AQuA-RAT / MMLU loading (lazy-imported). |
| **lm-eval** | `lm-eval==0.4.12` | Standardised benchmark harness; installed **last with `--no-deps`**. |
| **llama.cpp** | upstream (cloned/built on demand) | HF→GGUF conversion + `llama-quantize` Q4_K_M for on-device. |
| **llama-cpp-python** | upstream | GGUF inference backend (`GGUFBackend`) for the on-device latency path. |
| **W&B** | `wandb` (optional) | Experiment tracking; auto-offline when `WANDB_API_KEY` is unset. |
| **matplotlib / scipy / numpy** | loose | Pareto / length-histogram plots and stats in `eval/plots.py`. |

> **flash-attn is intentionally skipped** — there is no official wheel for torch
> 2.10, and Unsloth uses xformers + Triton kernels instead. See §5.

---

## 2. OSS Libraries / Projects Used (with links)

**Frameworks & runtimes**
- HuggingFace TRL — https://github.com/huggingface/trl
- Unsloth — https://github.com/unslothai/unsloth
- HuggingFace Transformers — https://github.com/huggingface/transformers
- PEFT — https://github.com/huggingface/peft
- Accelerate — https://github.com/huggingface/accelerate
- vLLM — https://github.com/vllm-project/vllm
- bitsandbytes — https://github.com/bitsandbytes-foundation/bitsandbytes
- HuggingFace `datasets` — https://github.com/huggingface/datasets
- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- llama.cpp — https://github.com/ggml-org/llama.cpp
- llama-cpp-python — https://github.com/abetlen/llama-cpp-python
- Weights & Biases — https://github.com/wandb/wandb

**Models** (open-weight; no proprietary API in the product)
- Qwen2.5-1.5B-Instruct (Apache-2.0) — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct *(base policy)*
- Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct — one `--model` flag switch (same ChatML + 7 LoRA modules)
- Qwen2.5-0.5B-Instruct — https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct *(encoder base for the optional difficulty verifier)*

**Datasets**
- GSM8K (MIT) — https://huggingface.co/datasets/openai/gsm8k *(train: improve)*
- StrategyQA — https://huggingface.co/datasets/ChilleD/StrategyQA *(train: improve; fallback https://huggingface.co/datasets/wics/strategy-qa for eval)*
- AQuA-RAT — https://huggingface.co/datasets/deepmind/aqua_rat *(train: improve)*
- MMLU (MIT) — https://huggingface.co/datasets/cais/mmlu *(eval-only: maintain; never trained on)*

**Method references** (we position honestly against these; we do not depend on them)
- Dr.GRPO — https://arxiv.org/abs/2503.20783
- DAPO — https://arxiv.org/abs/2503.14476
- AdaptThink — https://arxiv.org/abs/2505.13417
- Thinkless — https://arxiv.org/abs/2505.13379
- CODA — https://arxiv.org/abs/2603.08659
- Weaver — https://arxiv.org/abs/2506.18203
- TTRL — https://arxiv.org/abs/2504.16084

---

## 3. Technical Architecture

AdaptiveThink-RL has three coupled stages. The headline KPI gain comes from
**Stage 1** (accuracy core); Stages 2–3 are the efficiency + on-device layer.

```
                ┌──────────────────────── STAGE 1 — RLVR accuracy core ────────────────────────┐
  GSM8K ┐       │                                                                                │
StrategyQA ─────┼─► rl/data.py ──► mixed train pool (45/30/25) ──► Dr.GRPO (TRL GRPOTrainer) ───┼─► LoRA adapter
  AQuA  ┘       │        ▲                                  rule-based RLVR reward               │   outputs/grpo-seed*
                │        │ curriculum filter (drop solve_rate∈{0,1})                              │
                │  ┌─────┴───────── STAGE "CCDD" — self-difficulty (API-free novelty) ──────┐    │
                │  │ self_difficulty.py: sample base K× per item, score with the SAME        │    │
                │  │ verifier the reward uses ⇒ solve_rate; difficulty = 1 − solve_rate      │    │
                │  └────────────────────────────────────────────────────────────────────────┘    │
                └─────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                          (self_difficulty.jsonl also trains a tiny 0.5B difficulty verifier)
                                                  ▼
                ┌──────────────────── STAGE 2 — CCDD adaptive compute (efficiency) ─────────────┐
   question ───►│ verifier d∈[0,1] ─► RL router self-routes <think>/<no_think> on its 1st token │─► answer
                │ verifier-aware override: if model says no_think but d≥τ ⇒ force a think pass    │   (\boxed / <answer>)
                │ reward = correctness − λ_tok·tokens·(1−d) + λ_obey·honoured·correct            │
                └────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                ┌──────────────────── STAGE 3 — on-device GGUF ─────────────────────────────────┐
                │ merge LoRA → FP16 HF → llama.cpp convert → llama-quantize Q4_K_M → llama.cpp   │─► Samsung Galaxy
                └────────────────────────────────────────────────────────────────────────────────┘
```

### Data flow
1. **Pool build** ([`src/adaptivethink/rl/data.py`](../src/adaptivethink/rl/data.py)).
   GSM8K / StrategyQA / AQuA-RAT train splits are mapped to a uniform
   `{prompt, answer, dataset, question}` schema with a shared ChatML
   `<think>…</think><answer>…</answer>` system prompt, mixed ~45/30/25 into a
   ~6.1k-row pool. **HARD RULE:** only train splits are touched — MMLU and all
   test splits are eval-only.
2. **CCDD self-difficulty** ([`src/adaptivethink/rl/self_difficulty.py`](../src/adaptivethink/rl/self_difficulty.py)).
   The frozen base model is sampled `K` times per question; each rollout is
   scored against gold with the **same** matcher the reward uses, giving an
   empirical `solve_rate`. Output rows
   `{question, answer, difficulty, solve_rate, source}` go to
   `data/self_difficulty.jsonl`.
3. **Dr.GRPO RL** ([`src/adaptivethink/rl/drgrpo_train.py`](../src/adaptivethink/rl/drgrpo_train.py)).
   The pool is curriculum-filtered with the CCDD file (drop `solve_rate` 0.0 and
   1.0 — zero-advantage groups), then trained with TRL `GRPOTrainer` under a
   Dr.GRPO config and the RLVR reward. Output: a LoRA adapter per seed.
4. **Eval** ([`eval/run_benchmarks.py`](../eval/run_benchmarks.py), [`eval/plots.py`](../eval/plots.py), `lm-eval`).
   The *identical* harness config is run for baseline and trained model, over
   eval seeds `{0,1,2}` for confidence intervals, producing the KPI delta table
   and Pareto charts.
5. **Route + quantize** ([`inference/pipeline.py`](../src/adaptivethink/inference/pipeline.py), [`quantize/export_gguf.py`](../src/adaptivethink/quantize/export_gguf.py)).
   The trained adapter drives the adaptive-compute router; the merged model is
   exported to GGUF Q4_K_M for on-device latency.

### The exact reward formulas

**Stage 1 — RLVR correctness + format**
([`src/adaptivethink/rl/rewards.py`](../src/adaptivethink/rl/rewards.py)). TRL
sums weighted reward functions; weights are `[1.0, 0.2]`:

```
reward = 1.0 · correctness  +  0.2 · format
  correctness = 1 if matcher(extracted_answer, gold) else 0      (binary exact-match RLVR)
  format      = 1 if completion matches ^\s*<think>…</think>\s*<answer>…</answer>\s*$ else 0
```

`correctness` reuses the robust matcher in
[`router/reward.py`](../src/adaptivethink/router/reward.py) (numeric / fraction /
boolean / LaTeX tolerant). Format is kept small (0.2) so it can never be gamed
over correctness. **No teacher API anywhere.**

**Stage 2 — adaptive-compute router** (`compute_rewards` in
[`router/reward.py`](../src/adaptivethink/router/reward.py)):

```
reward = correct − λ_tok · n_tokens · (1 − d)  +  λ_obey · honoured · correct
  λ_tok  = 3e-3   (length penalty)         d        = external difficulty ∈ [0,1]
  λ_obey = 0.05   (route-honoured bonus)   honoured = 1 if the chosen <think>/<no_think> tag is respected
```

The **`(1 − d)` gate is the efficiency-layer novelty**: easy questions pay the
full length penalty (pushing the router to `no_think`), hard questions pay
near-zero (so the router keeps full CoT where it matters and routing does not
collapse). Crucially `d` is **external** (a separate signal), not the policy's
own confidence.

---

## 4. Implementation Details

### 4.1 Dr.GRPO configuration
Assembled in `_build_grpo_config` ([`drgrpo_train.py`](../src/adaptivethink/rl/drgrpo_train.py));
knobs map verbatim to `trl.GRPOConfig`. Defaults from
[`configs/strategy1.yaml`](../configs/strategy1.yaml):

| GRPOConfig field | Value | Why |
|---|---|---|
| `loss_type` | `"dr_grpo"` | Constant `1/(L·G)` length normalization — kills the GRPO length bias. |
| `scale_rewards` | `False` (hard-set) | Drops per-group std division — kills the difficulty bias (the second half of the Dr.GRPO fix). |
| `beta` (KL) | `0.0` | No KL ⇒ no reference model loaded (saves memory); the cited SLM ablation favours no-KL for tiny models. |
| `epsilon` / `epsilon_high` | `0.2` / `None` | Symmetric clip. `--clip-higher` sets `0.28` but is **off** (hurts sub-3B and is inert at on-policy `μ=1`). |
| `top_entropy_quantile` | `1.0` | Entropy **mask** (TRL has no additive bonus); 1.0 = train on all tokens, kept high for tiny models. |
| `mask_truncated_completions` | `True` | DAPO overlong filtering — the one DAPO trick that *helps* small models. |
| `num_generations` (G) | `8` | Group size; effective batch must divide by it. |
| `num_iterations` | `1` | On-policy (μ=1) ⇒ the clip term is a no-op. |
| `max_prompt_length` / `max_completion_length` | `512` / `1024` | 24h budget; `1024` is also the Dr.GRPO `L` constant + truncation cap. |
| `learning_rate` | `1e-6` | Small LR for RL stability. |
| LoRA | `r=16, α=32`, 7 modules | `q/k/v/o/gate/up/down_proj` — shared by all Qwen2.5 dense sizes. |

Loading is **Unsloth 4-bit QLoRA first**, with a transformers + PEFT +
bitsandbytes NF4 fallback. vLLM colocated rollouts auto-enable at ≥22 GB VRAM.
Training is idempotent/resumable — `_find_resume_checkpoint` resumes from the
latest `checkpoint-*` (`save_steps=50`).

### 4.2 CCDD — Competence-Calibrated Difficulty Self-Distillation (the novelty)
`difficulty(q) = 1 − solve_rate(q)`, where `solve_rate` is the fraction of `K`
base-model rollouts whose extracted answer matches gold, scored with the exact
GRPO matcher ([`self_difficulty.py`](../src/adaptivethink/rl/self_difficulty.py)).
This is **API-free and teacher-free** — difficulty is distilled from the
*trainee's own competence*, replacing the legacy teacher-distilled verifier that
needed an LLM API the team deliberately avoids. Two downstream consumers:
1. **Curriculum filter for GRPO** (below).
2. **Self-labels for a tiny 0.5B difficulty verifier** (same `{question,
   answer, difficulty}` schema), which gates `think`/`no_think` on-device.

### 4.3 Curriculum filter
`apply_ccdd_filter` ([`rl/data.py`](../src/adaptivethink/rl/data.py)) drops rows
whose precomputed `solve_rate` is `0.0` (unsolvable) or `1.0` (trivial) via
`keep_item` (`0 < p < 1`) — both extremes produce zero-advantage GRPO groups
(no gradient). Unlabeled rows are kept (conservative). Resolution in the trainer
is guarded by `_resolve_self_difficulty_file`: the filter is applied only when
the file is configured **and** exists, else training falls back to the full pool
with a clear log line. An equivalent **online** filter
(`difficulty_filter_rows`, base-model pass-rate via injected callables) is also
available behind `--difficulty-filter`. Watch the trainer's
`frac_reward_zero_std` to confirm few dead groups remain.

### 4.4 The answer matcher
[`router/reward.py`](../src/adaptivethink/router/reward.py) is the single,
stdlib-only matcher reused across **training reward, CCDD scoring, and eval** —
so self-difficulty, RL reward, and the headline metric are all calibrated to the
same notion of "correct."
- `extract_answer`: prefers the **last** brace-balanced `\boxed{…}`, then the
  last labelled form (`final answer` / `answer is` / `answer:`), then a bare
  `=` fallback — picking the final answer out of a CoT trace, not a scratch
  value.
- `_answers_match`: normalises (strips `$`, commas, `\text{}`, LaTeX spacing),
  then boolean mapping (`yes/true/correct` vs `no/false/incorrect`), numeric
  equivalence (`72`==`72.0`, `1/2`==`0.5`, `\frac{1}{2}`==`0.5`, `$1,000`==`1000`,
  percent-tolerant, exact for integer-valued answers), then normalised string
  equality. Every parse is guarded so non-numeric input never throws.

---

## 5. Installation

One command resolves the entire stack into a project `.venv`:

```bash
./run.sh setup
source .venv/bin/activate   # for any manual module calls
```

**Dependency-resolution story** (enforced by `cmd_setup` in
[`run.sh`](../run.sh); pins in [`requirements.lock`](../requirements.lock)):

1. **vLLM first** — `vllm==0.19.1`, the strictest torch pinner, brings
   `torch==2.10.0` (uv `--torch-backend=auto`, or pip `--extra-index-url`
   CUDA wheels).
2. **Re-pin the torch trio** defensively (`torch/torchvision/torchaudio`).
3. **Training stack into the resolved window** — Unsloth + transformers 4.57.6
   + TRL 0.24 + PEFT + accelerate + bitsandbytes + datasets.
4. **Repo runtime deps** (loose — must not move torch).
5. **lm-eval last with `--no-deps`** so its loose floors cannot silently upgrade
   torch/transformers/vllm; its runtime deps are added unpinned afterward.
6. **flash-attn SKIPPED** — no official torch-2.10 wheel exists; Unsloth uses
   xformers + Triton. Opt back in only via `FLASH_ATTN_WHEEL=<url>`.

Setup also installs the repo editable (`adaptivethink.*`), verifies imports +
asserts `torch<2.11`, and runs a 1-step GRPO smoke if a GPU is present. uv is
preferred (Unsloth's recommended resolver) with an automatic plain-pip fallback.
GRPO runtime gotcha handled by the trainers: **`import unsloth` precedes
`trl`/`transformers`**.

> **Target hardware:** one Linux GPU, CUDA 12.x, ~50 GB VRAM (developed against
> a single RTX A6000 48 GB). QLoRA 4-bit + gradient checkpointing keep sub-3B
> comfortably in budget; the full ~24h run is `scripts/run_all_24h.sh`.

---

## 6. User Guide

### `run.sh` subcommands
`./run.sh <subcommand>` (each activates `.venv`; stages are idempotent/resumable;
logs to `logs/<stage>.log`):

| Subcommand | What it does |
|---|---|
| `setup` | Build `.venv` + resolve the pinned stack (§5), verify, 1-step smoke. |
| `smoke` | Tiny end-to-end wiring check (2 GRPO steps, no vLLM, ~5–7 GB). |
| `baseline` | `lm-eval` on the **base** model (GSM8K/MMLU/StrategyQA) — the fixed comparison harness. |
| `sft` | *(optional)* short `<think>/<answer>` cold-start on R1 traces (off unless `--sft`/`RUN_SFT=1`). |
| `grpo` | Multi-seed Dr.GRPO RL → `outputs/grpo-seed*` (`SEED=<n>` for one seed). |
| `eval` | `lm-eval` on the trained adapter with the **identical** baseline config + efficiency Pareto. |
| `router` | Verifier + adaptive routing Pareto points + plots. |
| `quantize` | Merge adapter → GGUF Q4_K_M for on-device. |
| `all` | `baseline → (sft) → grpo → eval → router → quantize`. |
| `help` | Usage. |

Common direct calls:
```bash
./run.sh smoke
./run.sh grpo --model Qwen/Qwen2.5-1.5B-Instruct --seeds 0,1,2 --loss dr_grpo --kl 0.0
# CCDD self-difficulty pass (writes data/self_difficulty.jsonl):
python -m adaptivethink.rl.self_difficulty --model Qwen/Qwen2.5-1.5B-Instruct \
  --datasets gsm8k,strategyqa,aqua --k 8 --n 2000 --out data/self_difficulty.jsonl
# Full 24h single run:
bash scripts/run_all_24h.sh
```

> **Full staged command reference:** see [`RUNBOOK.md`](../RUNBOOK.md) — it lists
> the exact copy-paste commands per stage (Day 0 setup → Day 12–14 ablations),
> the stability A/B levers, and the multi-seed eval/CI workflow.

### Switching the base model with one flag
All Qwen2.5 dense sizes share ChatML and the same 7 LoRA modules, so scaling up
is a single `--model` change — no code edits:
```bash
./run.sh grpo --model Qwen/Qwen2.5-3B-Instruct    # or Qwen/Qwen2.5-7B-Instruct
# equivalently: MODEL=Qwen/Qwen2.5-7B-Instruct ./run.sh all
```
The default base is `Qwen/Qwen2.5-1.5B-Instruct` (chosen for headroom — see §7).
The same flag flows through `self_difficulty`, `drgrpo_train`, eval, and quantize.

---

## 7. Salient Features

- **API-free novelty (CCDD).** Difficulty is the trainee's own
  `1 − solve_rate` over K rollouts — no teacher, no LLM API (a hard PS06
  constraint). It serves double duty: a GRPO curriculum filter *and* the gating
  signal for adaptive compute.
- **+5% headroom model choice.** `Qwen2.5-1.5B-Instruct` (Apache-2.0) is
  deliberately non-saturated (base GSM8K ≈73%), leaving room for the required
  ≥+5% lift while running fast on one A6000; 3B/7B are one-flag switches if more
  capacity is wanted.
- **Accuracy core + efficiency layer, cleanly separated.** Dr.GRPO RLVR drives
  the KPI gain (GSM8K + StrategyQA improve; MMLU maintained, never trained); the
  `(1−d)`-gated adaptive-compute router is bolted on for tokens/latency wins and
  ships as an optional inference mode if it ever dips below the +5% bar.
- **On-device.** QLoRA 4-bit train → merge → GGUF **Q4_K_M** (llama.cpp) for
  Samsung Galaxy inference, with s1-style budget-forced think extension and an
  exact-tokenizer context-headroom guard.
- **Robust, shared evaluation.** One tolerant matcher across train + CCDD +
  eval; the **identical** `lm-eval` config for baseline and trained model;
  multi-seed (`0,1,2`) Pass@1 with std/CIs; unbiased Pass@k (Chen et al.); an
  automatic KPI table that flags ≥+5% on ≥2 of GSM8K/MMLU/StrategyQA.

---

## 8. Results (to be filled after the run)

> Numbers do not exist yet. After `./run.sh all` / `scripts/run_all_24h.sh`,
> populate from `results/` (`results/figures/kpi_table.md` is generated by
> `eval/plots.py`).

**KPI delta table** — *target: ≥+5% on ≥2 of GSM8K/MMLU/StrategyQA over the
SAME base baseline.*

| Benchmark | Baseline (mean±CI) | Trained (mean±CI) | Δ | Target | KPI met? |
|---|---|---|---|---|---|
| GSM8K | <!-- TODO: fill from results/ after the run --> | <!-- TODO --> | <!-- TODO --> | ≥0.50 & +5% | <!-- TODO --> |
| StrategyQA | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | ≥0.65 & +5% | <!-- TODO --> |
| MMLU (maintain) | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | ≥0.45 (hold) | <!-- TODO --> |

**Figures to insert** (generated under `results/figures/`):
- <!-- TODO: insert Pareto chart — accuracy vs avg output tokens (pareto_gsm8k.png) -->
- <!-- TODO: insert reward curve from the GRPO run (W&B export / log scrape) -->
- <!-- TODO: insert entropy / frac_reward_zero_std curve (stability evidence) -->
- <!-- TODO: insert length-distribution histogram per routing policy (len_*_gsm8k.png) -->
- <!-- TODO: insert one clean qualitative <think>/<answer> example -->
- <!-- TODO: insert on-device latency table from the quantized GGUF (Galaxy) -->

---

*See also: [`docs/ax.md`](ax.md) (agentic AI usage), [`README.md`](../README.md)
(overview), [`RUNBOOK.md`](../RUNBOOK.md) (staged commands),
[`WHERE_TO_RUN.md`](../WHERE_TO_RUN.md) (environment notes).*
