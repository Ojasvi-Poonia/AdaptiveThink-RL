# docs/ax.md — Agentic AI & Open-Weight Model Usage **[Important]**

> Required by the Samsung EnnovateX AX Hackathon 2026 (PS06) submission guidelines.
> This document is the canonical account of (a) the open-weight models AdaptiveThink
> ships, and (b) how the system was **built** and how it **runs** using an agentic
> coding harness. It is written to be judged for both **Technical (30%)** and
> **Innovation (25%)**, so it is deliberately concrete and candid — including a
> "what worked / what did NOT work" section.
>
> **Honesty note on numbers.** The 24h training run produces the headline KPI
> deltas; those numbers do not exist at the time of writing. Every measured KPI in
> this repo is a clearly-marked placeholder
> `<!-- TODO: fill from results/ after the run -->`. Below we state **targets** and
> **methodology**, never fabricated measurements.

---

## 0. TL;DR

- **Trained reasoner (open-weight):** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0),
  RL-tuned with **Dr.GRPO + rule-based verifiable rewards (RLVR)** — no SFT, and
  **no LLM API anywhere** in the pipeline (a hard project constraint).
- **Novelty (API-free):** **CCDD — Competence-Calibrated Difficulty
  Self-Distillation.** Difficulty is `1 − (the base model's own empirical
  solve-rate over K rollouts)` — no teacher model, no API. CCDD does double duty:
  a **GRPO curriculum filter** (drops zero-gradient groups) and the label source
  for a **tiny 0.5B difficulty verifier** that acts as an on-device adaptive-compute
  gate.
- **How it was built:** with **Claude Code** as the coding harness, driven by a
  **multi-agent Workflow orchestrator** in a strict **research → parallel build →
  adversarial verify** loop. Adversarial reviewer agents caught **real integration
  bugs before any GPU time was spent** (a `build_grpo_pool` signature mismatch
  between parallel agents, a double-`<think>` token bug, and a false-PASS in the
  KPI table).
- **The agentic angle in the product itself:** at inference, the verifier is an
  amortized "tool call" that gates a `think` / `no_think` decision — an agentic
  adaptive-compute controller running fully on-device after a GGUF Q4_K_M export.

---

## 1. Open-Weight Models Used

| Model | Role in the system | Why chosen | In shipped artefact? |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) | The **trained reasoner** (Dr.GRPO RLVR target) | Non-saturated baseline ⇒ real **+5% headroom** (GSM8K base ≈73%); fastest ⇒ most GRPO steps in a 24h budget on one A6000; open license; 3B/7B are a one-flag `--model` switch (all Qwen2.5 share ChatML + the same 7 LoRA modules) | **Yes** |
| `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) | Encoder base of the **400M difficulty verifier** (mean-pooled encoder + small regression head) — the on-device adaptive-compute gate | Compact, strong text representations; fits trivially alongside the reasoner on-device | **Yes** |

**Everything shipped is open-weight (Apache-2.0).** Critically, there is **no
teacher model and no external LLM API** at any stage — neither for reward, nor for
difficulty labelling, nor for evaluation. This is enforced as an interface contract
in `configs/strategy1.yaml` ("No teacher API anywhere") and re-stated in the trainer
docstring (`src/adaptivethink/rl/drgrpo_train.py`).

> Historical note (honest): an earlier design distilled difficulty from a teacher
> LLM (DeepSeek-V3 via API) into the verifier. **That path was abandoned** when it
> became clear the team had no usable distillation API. It was replaced by CCDD
> (§3), which is strictly better for this problem because it calibrates difficulty
> to *the exact model being trained*, not a foreign teacher.

**Datasets** (all open, eval never touches a train split):
`openai/gsm8k` (MIT), `ChilleD/StrategyQA`, `deepmind/aqua_rat` (training mix
45/30/25); `cais/mmlu` is **eval-only and never trained on**.

---

## 2. The System Is Itself an Agentic Controller

AdaptiveThink is not just *built* with agents — at inference it **is** a small
agentic loop. The 400M verifier is an external, independently-callable tool whose
scalar output steers a planning decision:

```
Observe (question)
  → Tool call   : verifier.score(question) → difficulty d ∈ [0,1]
  → Plan        : route think / no_think
  → Act         : reasoner generates answer (full CoT, or direct)
  → Verify      : RLVR reward (exact-match correctness + light format)
```

This is implemented in `src/adaptivethink/inference/pipeline.py`
(`AdaptivePipeline`). Two design points make `d` *causal* rather than decorative:

1. **Verifier-aware override.** In the headline `model` route mode the RL router
   self-routes on its first emitted token. If it picks `no_think` but the verifier
   says the question is hard (`d ≥ override_threshold`), the pipeline **forces a
   think pass**. Without this the external difficulty signal never enters the loop.
   (This is "bug #1" in §6 — caught and fixed before any GPU run.)
2. **s1-style budget forcing.** A think pass starts on a *small* initial budget so
   there is real headroom to top up with a "Wait, let me re-check…" continuation if
   no boxed answer has appeared yet — capped against the context window so
   prompt+completion can never overflow. (This is "bug #2".)

The amortization argument (the Innovation pitch): CCDD spends rollout compute
**once, offline**, to compute per-item solve-rates; that signal is compressed into
a 0.5B verifier, so at inference the adaptive-compute decision costs a single cheap
forward pass instead of K live rollouts.

---

## 3. The Open-Weight Innovation: CCDD (API-Free Self-Distillation)

`src/adaptivethink/rl/self_difficulty.py` implements CCDD:

```
solve_rate(q) = (# of K base-model rollouts whose final answer matches gold) / K
difficulty(q) = 1 − solve_rate(q)
```

The base model is sampled `K` times per question (temperature > 0), each rollout's
final answer is extracted and checked against gold using **the exact same verifier
the GRPO reward uses** (`adaptivethink.router.reward.extract_answer` +
`_answers_match`, numeric/fraction/boolean/letter tolerant). No second opinion, no
teacher — difficulty is *the model's own competence*, measured against the metric it
is actually trained on. Two downstream consumers, both API-free:

1. **GRPO curriculum filter.** GRPO advantage is zero when every rollout in a group
   earns the same reward, so items at `solve_rate ∈ {0, 1}` carry no gradient.
   Dropping those extremes (`is_learnable: 0 < solve_rate < 1`) keeps only the
   learnable band. Wired in `rl/data.py::apply_ccdd_filter` and gated by
   `--self-difficulty-file` in the trainer.
2. **Verifier labels.** CCDD rows use the same `{question, answer, difficulty}`
   schema `verifier/train.py` consumes, so the 0.5B verifier is trained on
   *self-labels* (`scripts/03_train_verifier.sh`) with no teacher.

Position vs prior work (stated honestly): AdaptThink routes on internal model
confidence; CODA gates on group-rollout pass-rate; Thinkless/Weaver use other
controllers. CCDD's contribution is the **system-level** combination — using the
*trainee's own* offline solve-rate both as a GRPO curriculum signal **and** as the
distillation target for an amortized on-device gate, with **zero API dependence**.

---

## 4. Agentic Development Setup — Claude Code as the Coding Harness

The entire codebase was engineered with **Claude Code** (Anthropic's CLI) as the
coding harness, driven by a **multi-agent Workflow orchestrator**. The orchestration
pattern was deliberately **research → build → verify**, with a hard rule: *adversarial
verification happens before any GPU time is spent*, because GPU hours on the single
A6000 are the scarcest resource in a 24h budget.

### 4.1 The research → parallel-build → adversarial-verify loop

```
┌─────────────┐   structured    ┌──────────────────────┐   disjoint-file   ┌──────────────────────┐
│  RESEARCH   │  briefs (ALGO,  │   PARALLEL BUILD      │   contracts       │  ADVERSARIAL VERIFY  │
│  SCOUTS     ├────DATA, MODEL)→│   AGENTS              ├──────────────────→│  / CODE REVIEW       │
│ (web + HF)  │                 │ (1 agent / module,    │                   │  (red-team the diff, │
│             │                 │  non-overlapping files)│                  │   run pure-logic     │
└─────────────┘                 └──────────────────────┘                   │   tests, KPI sanity) │
       ▲                                                                     └──────────┬───────────┘
       │                              re-scope / re-contract on a caught defect         │
       └─────────────────────────────────────────────────────────────────────────────┘
```

- **Research scouts** ran web + Hugging Face research and emitted **structured-output
  briefs** ("ALGO research" for the Dr.GRPO/TRL knob mapping, "DATA research" for the
  exact dataset ids/fields, "MODEL research" for base-model headroom). These briefs
  are still visible as the design-rationale comments throughout
  `rl/drgrpo_train.py`, `rl/data.py`, and `configs/strategy1.yaml` — e.g. the
  verbatim TRL `GRPOConfig` mapping (`loss_type="dr_grpo"`, `scale_rewards=False`,
  `mask_truncated_completions=True`, `top_entropy_quantile=1.0`) and the note that
  TRL's default `loss_type="dapo"` is harmful for sub-3B models.
- **Parallel build agents** each owned a **disjoint set of files** (reward,
  data/loaders, trainer, verifier, inference, eval) so they could run concurrently
  without stepping on each other.
- **Adversarial verify agents** red-teamed the merged diff, ran the pure-logic test
  suite, and sanity-checked the KPI logic *before* any GPU run.

### 4.2 Structured-output schemas as the inter-agent contract

The lesson that made parallelism safe: **parallel agents drift on shared
interfaces** unless the interface is a written contract. We made several contracts
explicit and machine-checkable:

- **Reward matcher is the single source of truth.** `router/reward.py`'s
  `extract_answer` + `_answers_match` are reused — never reimplemented — by the RLVR
  reward (`rl/rewards.py`), the CCDD scorer (`rl/self_difficulty.py`), and eval
  (`eval/run_benchmarks.py`). The docstrings literally say "do NOT reimplement
  matching." This prevents three agents inventing three subtly-different "is this
  answer correct?" functions.
- **Uniform dataset row schema** `{prompt, answer, dataset, question}` across every
  loader.
- **CLI/arg surface mirrors the YAML** (`configs/strategy1.yaml` maps 1:1 to
  `drgrpo_train` flags), so the orchestrator script and a human override agree.

### 4.3 Tool use / tool chaining (edit → compile → test loops)

Within Claude Code, the tool chain per build step was:

1. `Read` existing code to learn the contract before modifying it.
2. `web_search` / Hugging Face lookups → confirm dataset ids, TRL API, model
   licenses (the registry-first research step).
3. `Edit` / `Write` → implement on the owned files only.
4. `Bash` → `python -m py_compile` every module (heavy deps like torch/trl/unsloth
   are imported **inside functions** specifically so modules compile and unit-test
   on a machine with no GPU), then run the **pure-logic test suite**
   (`tests/test_reward.py`, `test_eval.py`, `test_rl_data.py`, `test_loaders.py`)
   without the ML stack installed.

This "edit → compile → pure-logic test" inner loop is what let the whole pipeline be
validated **off-GPU**, reserving the A6000 exclusively for the real training run.

### 4.4 Memory / context handling across sessions

Claude Code's context does not persist by default, so continuity was supplied by:

- **Persistent project memory** (the harness's cross-session memory): three notes —
  *project* (KPIs, hardware, top-50 standing), *strategy* (the "invert to an
  accuracy core" decision and the pivot to CCDD), and *codebase-fixes* (the exact
  accuracy-lever bugs already fixed and verified by 51 passing tests). Every session
  reads these first, so the strategy and the list of already-fixed bugs never get
  re-litigated or regressed.
- **In-repo durable docs** as a retrieval surface: `README.md`, `RUNBOOK.md`
  (the day-by-day command plan), `configs/strategy1.yaml` (the interface contract
  with `HARD RULES`), and `scripts/run_all_24h.sh` (the executable plan). New agents
  orient from these instead of guessing.

### 4.5 MCP servers & skills

- **MCP / tool servers** were used for the research and version-pinning steps —
  documentation lookup (library/API behaviour) and code/web search to confirm the
  exact pinned stack (`vllm 0.19.1` → `torch 2.10` → `TRL 0.24` / `transformers
  4.57.6`, `lm-eval` installed last with `--no-deps`). The pinning rationale is
  preserved in `RUNBOOK.md` Day-0.
- **Skills / sub-agent roles**: planner (the staged RUNBOOK), language-specific
  Python review, and a security/secrets pass (the project keeps secrets in
  `.env.template`, never hard-coded). The orchestrator dispatched these as the
  build/verify roles in §4.1.

---

## 5. The Training Pipeline as an Executable Agentic Workflow

`scripts/run_all_24h.sh` is the single, resumable orchestration of the whole run —
itself a workflow with explicit stages, idempotent outputs, and a documented
**time-budget cut order** (drop quantize → drop multi-seed → drop CCDD → stop GRPO
at the last ≥250 checkpoint) so a human orchestrator can degrade gracefully if the
24h wall-clock gets tight:

```
0  smoke (50-step GRPO sanity)                       [optional]
1  baseline eval (never_think + always_think)        gsm8k / strategyqa / mmlu
2  CCDD self-difficulty  → data/self_difficulty.jsonl
3  Dr.GRPO RLVR training → outputs/grpo-seed0
4  self-distilled 400M verifier on CCDD labels       [optional]
5  final eval (--seeds 0,1,2) + KPI table            eval/plots.py
6  quantize merged adapter → GGUF Q4_K_M             on-device path
```

Every stage skips itself if its primary output exists (re-running resumes from the
first incomplete stage). The KPI table (`eval/plots.py`) flags PASS only when
`delta ≥ +5%` **and** `router ≥ target` on the required number of benchmarks — see
§6 for the false-PASS bug this logic was hardened against.

---

## 6. What Worked / What Did NOT Work (candid)

This section is the point of the document for judges: the honest, specific record.

### Real integration bugs the adversarial-verify phase caught *before* GPU time

| Bug | How it arose | Fix | Where |
|---|---|---|---|
| **`build_grpo_pool` signature mismatch** | Two parallel agents disagreed on the pool builder's return schema/args — one emitted `{prompt, answer, source}`, a consumer expected `question` too. A shared interface drifted because there was no written contract. | The CCDD pool loader (`load_pool`) now defensively prefers `load_rows` (which carries `question`) and falls back across `build_dataset`/`build_grpo_pool`; the row schema was made uniform. | `rl/self_difficulty.py::load_pool`, `rl/data.py` |
| **Double-`<think>` token** | Budget-force continuations re-prepended `<think>` to a completion whose prompt already ended with the prefilled `<think>`, producing `<think><think>`. | `_budget_extend` strips the duplicate routing token from the body when `prefilled`. | `inference/pipeline.py::_budget_extend` |
| **False-PASS in the KPI table** | Early KPI logic hardcoded a "2 of 3" pass rule, so a run that only had **1 of 2** comparable benchmarks could read as PASS. | Denominator derived as `ceil(n_comparable * 2/3)` (n=1→1, n=2→2, n=3→2); PASS requires `delta ≥ +5%` **and** `router ≥ target`. | `eval/plots.py::kpi_table` |
| **`d` was decorative** (verifier ignored) | In `model` route mode the RL router self-routed and the external difficulty never affected inference. | Verifier-aware override forces a think pass when `d ≥ override_threshold`. | `inference/pipeline.py` (bug #1) |
| **Budget-forcing never fired** | Giving the first think pass the *full* budget saturated the headroom check, so the "Wait" top-up never triggered. | First pass capped at a small `initial_think_tokens`, leaving real headroom. | `inference/pipeline.py` (bug #2) |
| **`extract_answer` grabbed the *first* scratch value** | A CoT trace ends with the real answer, but the matcher took the first `\boxed{}`/`=`. | Take the **last** boxed/labelled answer; equivalence-aware matching (numeric/`\frac`/`$`/comma, yes/no↔true/false). This is the single biggest accuracy lever and it also cleans the GRPO reward signal. | `router/reward.py` |
| **`lambda_tok` too weak** | An early length penalty `5e-4` was too small to discourage long answers on easy items; a unit test flagged it. | Raised to `3e-3`; `lambda_obey` kept small so a wrong-but-honoured answer stays negative. | `router/reward.py::compute_rewards` |
| **`num_workers>0` deadlock** | DataLoader workers can deadlock in forked processes on the verifier trainer. | `NUM_WORKERS = 0` with a comment. | `verifier/train.py` |

The throughline: **most of these are integration defects between independently-built
parts** — exactly the failure mode of parallel agents — and they were caught by
red-team review + pure-logic tests, *not* by the original author agent. That is the
case for keeping an adversarial verify phase distinct from the build phase.

### What worked

- **Research → build → verify with disjoint-file ownership** genuinely parallelised
  the build, and the verify phase paid for itself by catching the bugs above before
  a single GPU hour.
- **One reused reward matcher** as a contract eliminated a whole class of
  "three different correctness functions" bugs and made CCDD, RLVR, and eval
  mutually consistent by construction.
- **Heavy-imports-inside-functions** let the entire pipeline be unit-tested and
  `py_compile`d off-GPU, so the A6000 was reserved for the real run.
- **Persistent project memory** stopped the strategy (and the fixed-bug list) from
  being re-litigated every session.

### What did NOT work (and the pivots)

- **Parallel agents drift on shared interfaces.** Without an explicit, written
  contract, two agents disagreed on `build_grpo_pool` — see the bug table. The fix
  was process: *contracts first, then a mandatory verify phase.*
- **The teacher-API distillation path was a dead end.** The original verifier was to
  be distilled from a teacher LLM via API; the team had no usable distillation API,
  so it was **abandoned and replaced by API-free CCDD**. (A stale docstring in
  `verifier/model.py` still says "teacher soft labels" — the live training default
  in `verifier/train.py` is the CCDD `self_difficulty.jsonl`.)
- **7B was rejected for lack of headroom.** The KPI is *+5% over the same base
  model's baseline*, and baseline accuracy is anti-correlated with headroom (7B
  GSM8K ≈91%, 3B ≈87% leave almost no room). 1.5B (≈73%) makes +5% reachable *and*
  trains fastest — so the "bigger model" instinct was the wrong move here. 3B/7B
  remain a one-flag `--model` switch.
- **flash-attn build failures.** No official wheel exists for the pinned
  `torch 2.10`; flash-attn was **skipped** (Unsloth uses xformers/Triton), per the
  Day-0 pinning notes in `RUNBOOK.md`.
- **Dependency resolution was the single biggest non-GPU pain.** The fix was a
  strict install order (vLLM first as the strictest torch pinner, training stack
  pinned into the resolved window, `lm-eval` last with `--no-deps`), captured in
  `requirements.lock` / `RUNBOOK.md` so it is reproducible rather than rediscovered.

---

## 7. Measured Results (placeholders — to be filled after the run)

<!-- TODO: fill from results/ after the 24h run. -->
- KPI delta table (baseline vs router, +5% / target gate): `results/figures/kpi_table.md`
- Accuracy-vs-compute Pareto + length histograms: `results/figures/`
- Multi-seed Pass@1 (mean ± std over seeds 0/1/2) and on-device GGUF latency.

**Targets (not measurements):** improve **GSM8K** and **StrategyQA** by **≥ +5%**
over the `Qwen2.5-1.5B-Instruct` baseline (meeting ≥2 of the 3 KPI benchmarks),
**maintain MMLU** above floor, while the verifier-gated router cuts average tokens on
easy items — reported with confidence intervals on the identical lm-eval harness used
for the baseline.

---

## 8. Reproduce

```bash
./run.sh setup                 # pinned stack (vLLM→torch 2.10→TRL 0.24; flash-attn skipped)
bash scripts/run_all_24h.sh    # the ONE resumable 24h pipeline (stages 0–6 above)
# 3B/7B is a one-flag switch:  MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_all_24h.sh
```

See `RUNBOOK.md` for the staged day-by-day commands and per-flag overrides, and
`docs/technical.md` for stack/architecture detail.
