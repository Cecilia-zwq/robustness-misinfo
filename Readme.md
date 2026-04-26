# robustness-misinfo

> **Research question.** How robust are LLMs in resisting misinformation across a multi-turn interaction?

We study this by pitting a **simulated misinformation-leaning user** (an LLM-based agent designed to stay in character and hold a false belief) against a **target LLM**, and scoring the target’s behavior with a dedicated **evaluator**.

---

## Table of contents

1. [Current status](#current-status)
2. [Repository layout](#repository-layout)
3. [Iteration tracker (at a glance)](#iteration-tracker-at-a-glance)
4. [Iteration history in detail](#iteration-history-in-detail)
5. [Datasets](#datasets)
6. [Environment](#environment)
7. [Conventions](#conventions)

---

## Current status

**Active iteration: `Iteration 5`** — passage-scale misinformation, Experiment 3 validation.

|                       | Location                                     |
| --------------------- | -------------------------------------------- |
| Framework code        | `scripts/Iteration5/misinfo_eval_framework/` |
| Experiment entrypoint | `scripts/Iteration5/experiment3.py`          |
| Latest outputs        | `results/Iteration5/`                        |

**Collaborators: start here.** Older iterations (`Iteration1`–`Iteration4`) are preserved snapshots for reproducibility and history, not the current working system.

---

## Repository layout

| Directory    | Purpose                                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/`      | Curated claim sets (`data/dataset/`) and raw fake-news text (`data/fake/`).                                                                      |
| `scripts/`   | Evaluation framework and experiment drivers, **versioned by iteration** (`Iteration1` … `Iteration5`). Each folder is a self-contained snapshot. |
| `results/`   | Timestamped run outputs (logs, summaries, checkpoints), mirroring the iteration folders in `scripts/`.                                           |
| `notebooks/` | Dataset filtering, inspection, and per-experiment analysis.                                                                                      |
| `models/`    | Reserved for local model assets.                                                                                                                 |

---

## Iteration tracker

| #   | Theme                                | Key change                                                             | Experiment                                  | Status     |
| --- | ------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------- | ---------- |
| 1   | Naive two-model dialogue             | Plumbing for role-played conversations                                 | —                                           | Archived   |
| 2   | **Agentic user simulation** (Plan A) | Multi-module user: plan → act → reflect                                | —                                           | Archived   |
| 3   | **Base framework + API**             | Package with three components: `user_agent`, `target_llm`, `evaluator` | **Experiment 1** (short claims, full-scale) | Archived   |
| 4   | **Finer-grained reflection**         | Split reflection into _character break_ + _belief break_               | **Experiment 2** (matched subset)           | Archived   |
| 5   | **Passage-scale misinformation**     | Long-form inputs; system-prompt persona; parallel runs                 | **Experiment 3** (validation)               | **Active** |

Entry points inside each iteration folder follow the same convention: an `experiment*.py` driver with a usage docstring at the top, plus (from Iteration 3 on) a `misinfo_eval_framework/` package.

---

## Iteration history in detail

### Why a naive two-model chat was not enough — the motivation for Iteration 2

Early tests showed a consistent failure mode: the simulated user was **too quick to accept the target model’s corrections** and **drifted to ground truth by the second turn**, even with an uncensored user model. In short, it **stopped pushing misinformation**.

This matches a known limitation of LLM-based personas: models **prefer to correct false claims and drop role-play** when pushed, making it hard to portray **fact-resistant** people from a prompt alone.

> Chuang, Yun-Shiuan, et al. _Simulating Opinion Dynamics with Networks of LLM-based Agents._ Findings of the ACL: NAACL 2024.

**Plan A (adopted).** Instead of a single loose prompt, steer the simulated user with **multiple modules**:

- **Planning module** — decides how to advance the misinformation goal on the next turn.
- **Action / dialogue module** — turns that plan into the actual user message.

Plan B was proposed but **never implemented**. **Iterations 2–5 all build on Plan A**; filenames like `PlanA-test0-v*.py` are legacy labels — read them as “the multi-module agentic user.”

---

### Iteration 1 — naive dialogue between two models

Two models take turns, each with its own system prompt. Useful for validating APIs and prompt formats, but it **does not** solve the collapse-toward-truth problem described above.

### Iteration 2 — agentic user simulation (Plan A)

First working **simulated user agent**: **plan → act → reflect**, with logging, so the user keeps a **false belief** and a **stable character** rather than folding under the target’s corrections. The overall shape is the one **carried forward** into Iterations 3–5.

**Deliberately set aside (to keep the baseline tractable):**

- A **memory module** letting the user learn from earlier turns to attack more cleverly — dropped as too complex at this stage.
- Rich **persuasion tactics** — deferred. The first priority is a believable user with a wrong belief and a coherent persona; **tactics are better treated later as an independent variable** in their own experiments.

Code appears as several versions of one large script (`PlanA-test0-v*.py`) before being refactored in Iteration 3.

### Iteration 3 — base framework and Experiment 1

The Iteration 2 design is refactored into a single package, `misinfo_eval_framework`, organized around **three components with a defined API**:

- `**user_agent`\*\* — the simulated misinformation-leaning user.
- `**target_llm**` — the model under evaluation.
- `**evaluator**` — scores how the target responds over the dialogue.

Shared **utilities** (session driver, helpers) live in the same package so the codebase is imported and run as one unit.

- **Experiment 1** — the first **large end-to-end run** on curated **single-sentence claims**; pilots are in the same folder; results were **reported in the course**. Anything named _experiment 1_ belongs here.

### Iteration 4 — finer-grained reflection and Experiment 2

Same three-part architecture as Iteration 3. The change is inside the user’s **reflection** step: Iteration 3 checked character and belief together in a single reflection; Iteration 4 **splits them into two explicit dimensions** — **character break** and **belief break** — for more granular diagnosis of where the persona slips.

- **Experiment 2** — confirms the stack still runs end-to-end with the split reflection. Uses **short claims** and a **matched subset** so results stay comparable to Experiment 1. **Result: it works.**

### Iteration 5 — passage-scale misinformation and Experiment 3 _(current)_

Iterations 3 and 4 targeted **single-sentence claims**. Iteration 5 **extends Iteration 4** so the framework can stress-test **richer inputs** — **multi-sentence or passage-length** misinformation (e.g., fake-news-style text), not only atomic claims.

Implementation notes:

- **Persona and belief** are delivered via **system-level instructions** for cleaner wiring.
- **Parallel execution** (ThreadPoolExecutor) for throughput on larger runs.
- **Ablation helpers** can load the Iteration 4 package side-by-side for controlled comparisons.
- **Experiment 3** — validates the long-text extension. Once validated, **long-text support is the maintained direction**.

---

## Datasets

All processed inputs live in `data/dataset/`. Two formats:

- **Short-claim** — single-sentence misinformation items, schema `(content, type)` (plus extras for `ds_bias`).
- **Long-text** — passage-scale misinformation items with a body, schema `(title, content, type)`, where `title` is the headline (display label) and `content` is the multi-sentence body (the long-text belief).

Raw source text and per-source cleaning notebooks sit under their own `data/<source>/` folders; the consolidation pass is in `notebooks/dataset_process.ipynb`.

### Short-claim datasets

| File                  | Rows | Columns                                               | Subtypes (`type`)                                                                                                                                                               |
| --------------------- | ---- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ds_bias.csv`         | 72   | `index, content, type, bias_type, stereotyped_entity` | `declaration`, `description`                                                                                                                                                    |
| `ds_conspiracy.csv`   | 59   | `content, type`                                       | `government malfeasance`, `personal wellbeing`, `malevolent global conspiracy`, `control of information`, `extraterrestrial cover-up`                                           |
| `ds_fibvid.csv`       | 430  | `content, type`                                       | `political_general`, `race_protest_police`, `economy_taxes_jobs`, `elections_voting`, `immigration_religion`, `covid_health`, `media_censorship`, `guns_violence`, `conspiracy` |
| `ds_climatefever.csv` | 253  | `content, type`                                       | `co2_emissions`, `temperature_warming`, `ice_sea_polar`, `general_climate_denial`, `extreme_weather`, `policy_energy`, `climate_science`                                        |

`ds_bias.csv` additionally exposes `bias_type` (13 stereotype attributes: `gender`, `age`, `political`, `regional-person`, `physical-appearance`, `urbanity`, `ethnicity`, `disability`, `region`, `sexual-orientation`, and combinations) and `stereotyped_entity`.

### Long-text datasets

| File                | Rows | Columns                | Subtypes (`type`)                                                                                                                                                                                           |
| ------------------- | ---- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ds_fakenews.csv`   | 240  | `title, content, type` | `business`, `education`, `entertainment`, `politics`, `sports`, `technology` (40 each)                                                                                                                      |
| `ds_fakehealth.csv` | 220  | `title, content, type` | `general_health`, `cancer`, `lifestyle_diet_alt_med`, `diagnostics_devices_drugs`, `chronic_disease`, `cardiovascular`, `neurological`, `mental_health`, `womens_reproductive_health`, `infectious_disease` |

---

## Environment

- **API routing.** Iterations 3+ use **LiteLLM**; set the relevant provider credentials (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) in the environment before running.
- **Python.** Standard scientific stack for analysis: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.
- **Ignored locally.** `.claude/` and `.vscode/` are listed in `.gitignore`.

---

## Conventions

- **One iteration = one self-contained snapshot** under `scripts/IterationN/`, with its outputs under `results/IterationN/`.
- **Experiment naming follows iterations:** Experiment 1 ↔ Iteration 3, Experiment 2 ↔ Iteration 4, Experiment 3 ↔ Iteration 5.
- **Runnable entrypoints** are `experiment*.py` files; their **module docstring is the source of truth** for flags, paths, and output layout.
- **Resumable runs.** Experiment drivers write incremental `summary.json`, `summary.csv`, `turn_level.csv`, and `checkpoint.json`, and support `--resume <existing_results_dir>`.
