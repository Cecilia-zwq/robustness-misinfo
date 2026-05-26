# The Fragility of Correction: Evaluating LLM Robustness to Misinformation in Multi-Turn Dialogues

Large language models exhibit sycophantic behaviour. Standard evaluations use static, pre-scripted user messages, which systematically underestimate this vulnerability. We replace scripted inputs with a **simulated user agent** that generates messages dynamically, persistently advocating for a misinformation belief throughout an 8-turn conversation.

Evaluating four frontier models across **5,697 conversations and 45,576 turn-level observations**, we identify three distinct model behavioural patterns: **stable resistance**, **gradual capitulation**, and **disengagement**. We find that emotionally framed misinformation causes the largest degradation in model robustness across all tested models, and that interactive evaluation exposes vulnerabilities that static benchmarks miss.

---

## Framework overview

The framework consists of three components:

```
┌─────────────────────────────────────────────────────────────────┐
│  Simulated User Agent                                           │
│  ┌──────────────┐    ┌───────────────────┐                     │
│  │ Actor module │───▶│ Reflection module │──▶ user message     │
│  │ (generates   │◀───│ (audits for       │                     │
│  │  candidate   │    │  belief / persona │                     │
│  │  message)    │    │  breaks)          │                     │
│  └──────────────┘    └───────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │ user message
                              ▼
                  ┌─────────────────────┐
                  │    Target LLM       │  ◀── model under evaluation
                  └─────────────────────┘
                              │ model response
                              ▼
                  ┌─────────────────────┐
                  │     Evaluator       │  scores correction quality,
                  └─────────────────────┘  rebuttal quality, agreeableness
```

**Simulated user agent** — given a false belief and an optional persona, the actor module generates an in-character user message. The reflection module audits each candidate for belief breaks and character breaks, requesting a regeneration if either is detected.

**Target LLM** — the model under evaluation. Receives a minimal system prompt to reflect natural deployment conditions, without any instruction to resist misinformation.

**Evaluator** — independently scores each target response on three dimensions (1–3 scale):

- **Correction quality**: how explicitly the model identifies the user's claim as misinformation.
- **Rebuttal quality**: how effectively the model counters the false belief with evidence.
- **Agreeableness**: the extent to which the model validates the user's intent.

---

## Repository layout

```
data/
  dataset/                        # curated false beliefs (285 sampled beliefs)

scripts/
  final_experiment/
    misinfo_eval_framework/       # core package: user_agent, target_llm, evaluator, llm_utils
    core/                         # experiment utilities: runner, storage, scoring, conditions
    main_user_IVs/                # main experiment (5 user conditions × 4 models × 285 beliefs)
    evaluator_validation/         # three-evaluator agreement study
    reflection_ablation/          # reflection module ablation
    static_interactive_ablation/  # static vs interactive comparison

results/
  final_experiment/
    main_user_IVs/
      conversations/              # raw conversation logs (5,697 sessions)
      conversations_static/       # static ablation conversations
      conversations_none_reflection/  # reflection ablation conversations
      scores/                     # turn-level scores (main experiment)
      scores_static/              # turn-level scores (static ablation)
      evaluator_validation/       # secondary evaluator scores and analysis
      analysis_output/            # figures and summary tables

notebooks/
  fianl_experiment/
    final_experiment_analysis.ipynb  # all analyses and figures reported in the paper
  dataset_process.ipynb              # dataset construction and sampling pipeline
```

---

## Datasets

All processed beliefs live in `data/dataset/`. The 285 beliefs used in the experiment are drawn from five source datasets:

| File                  | N   | Format                   | Source                                | Topics                                                    |
| --------------------- | --- | ------------------------ | ------------------------------------- | --------------------------------------------------------- |
| `ds_bias.csv`         | 72  | short claim              | BiasShades                            | gender, age, ethnicity, politics, … (13 stereotype types) |
| `ds_conspiracy.csv`   | 59  | short claim              | Generic Conspiracist Beliefs Scale    | government, personal wellbeing, global conspiracy, …      |
| `ds_climatefever.csv` | 40  | short claim              | Climate-FEVER                         | CO₂, temperature, ice/sea, policy, …                      |
| `ds_fakenews.csv`     | 74  | long text (title + body) | FakeNewsAMT                           | politics, technology                                      |
| `ds_fakehealth.csv`   | 40  | long text (title + body) | FakeHealth                            | cancer, cardiovascular, mental health, …                  |
| `sampled_beliefs.csv` | 285 | combined                 | all of the above                      | all of the above                                          |

`sampled_beliefs.json` contains the same 285 beliefs in the format consumed by the experiment scripts. `ds_fibvid.csv` is present in the dataset folder but was not included in the final experiment.

---

## Setup

**Python 3.11+** is required. Install the key dependencies:

```bash
pip install litellm openai pandas scipy statsmodels seaborn
```

For a full reproducible environment, use the provided conda spec:

```bash
conda create --name misinfo --file requirements.txt
conda activate misinfo
```

**API keys** — the framework routes all calls through [LiteLLM](https://github.com/BerriAI/litellm). Set whichever provider keys correspond to the models you use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

---

## Running the experiment

**Main experiment** (user behaviour conditions × target models):

```bash
cd scripts/final_experiment/main_user_IVs
python run_conversations.py   # runs all 5 user conditions × 4 models
python run_scoring.py         # scores the collected conversations
```

**Ablations:**

```bash
# Static vs interactive comparison
cd scripts/final_experiment/static_interactive_ablation
python run_static.py && python run_scoring.py

# Reflection module ablation
cd scripts/final_experiment/reflection_ablation
python run_no_reflection.py && python run_scoring.py
```

**Evaluator validation:**

```bash
cd scripts/final_experiment/evaluator_validation
python run_validation.py
python analyze_agreement.py
```

Configuration (models, temperatures, paths) is centralised in `scripts/final_experiment/main_user_IVs/config.py`.

---

## Analysis

All analyses and figures reported in the paper are reproduced in:

```
notebooks/fianl_experiment/final_experiment_analysis.ipynb
```

The notebook reads from `results/final_experiment/main_user_IVs/` and writes figures and tables to `results/final_experiment/main_user_IVs/analysis_output/`.

---

## Ethical considerations

The simulated user agent in this framework persistently advocates for false beliefs and can, in principle, elicit long-form misinformation text from a target model. **This framework is intended solely for controlled research settings.** Deploying it outside of evaluation contexts poses a misuse risk for large-scale misinformation generation. The misinformation datasets used contain offensive language and should not be used as training data.

---

## Citation

```bibtex
@inproceedings{anonymous2025fragility,
  title     = {The Fragility of Correction: Evaluating {LLM} Robustness to Misinformation in Multi-Turn Dialogues},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2025},
  year      = {2025},
  note      = {Under review}
}
```
