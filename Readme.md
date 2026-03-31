# robustness-misinfo

Repository for building misinformation-focused datasets and evaluating LLM robustness in multi-turn conversations with simulated adversarial users.

## Current project status

Iteration 3 is the active framework and experiment stage in this repo. It introduces a modular pipeline with three configurable components:

- simulated user agent,
- target LLM under test,
- evaluator for correction/rebuttal quality.

## Repository layout

- `data/dataset/`: core experiment datasets and curated claim sets.
- `data/fake/`: raw fake-news source text files (`*.fake.txt`) by domain.
- `scripts/Iteration1/`, `scripts/Iteration2/`: earlier iteration scripts.
- `scripts/Iteration3/`: active experiment scripts and reusable evaluation framework.
- `scripts/Iteration3/misinfo_eval_framework/`: modular framework package (user agent, target model wrapper, evaluator, session runner).
- `notebooks/filter_data.ipynb`: dataset filtering and earlier scoring work.
- `notebooks/Iteration1-2: testing/`: notebooks used in early-stage testing.
- `notebooks/Iteration3: Stage 2/`: Iteration 3 planning + analysis notebook.
- `results/Iteration2/`: archived Iteration 2 outputs.
- `results/Iteration3/`: pilot and full Iteration 3 run outputs.

## Datasets used in experiments

Primary files in `data/dataset/`:

- `ds_bias.csv`
  - Bias-focused misinformation statements.
  - Typical columns: `index`, `content`, `type`, `bias_type`, `stereotyped_entity`.

- `ds_conspiracy.csv`
  - Conspiracy-themed statements.
  - Typical columns: `content`, `type`.

- `ds_fakenews.csv`
  - Fake-news examples with domain labels.
  - Typical columns: `content`, `details`, `type`.

- `sampled_claims.json`
  - Curated claim subset used by Iteration 3 Experiment 1.
  - Includes category/subtype metadata for balanced runs.

## Iteration 3 workflow

Main scripts are in `scripts/Iteration3/`.

1. `test_personas.py`
   - Smoke test for persona behavior on a single claim.
   - Useful for validating prompts/character-break logic before larger runs.

2. `pilot_test1.py`
   - Small pilot over sampled claims and personas.
   - Produces timestamped pilot outputs in `results/Iteration3/`.

3. `experiment1.py`
   - Full run: 30 claims × 2 personas × 3 reps = 180 sessions.
   - Writes session logs plus `summary.json`, `summary.csv`, `turn_level.csv`, and `checkpoint.json`.
   - Supports resume with `--resume <existing_results_dir>`.

4. `experiment1_analysis.py`
   - Statistical summary and figure generation from `summary.csv` and `turn_level.csv`.
   - Generates trajectory/distribution/interaction plots for reporting.

## Iteration 3 output structure

Typical output directory:

`results/Iteration3/experiment1_YYYYMMDD_HHMMSS/`

- `sessions/`: one text log per session (claim x persona x rep).
- `summary.json`: full structured run output.
- `summary.csv`: session-level metrics.
- `turn_level.csv`: turn-level metrics.
- `checkpoint.json`: progress state for interrupted run recovery.
- `analysis/` (when generated): saved figures from analysis scripts/notebooks.

## Notebooks currently in use

- `notebooks/filter_data.ipynb`
- `notebooks/Iteration1-2: testing/test_dataset.ipynb`
- `notebooks/Iteration1-2: testing/test_StrongREJECT.ipynb`
- `notebooks/Iteration1-2: testing/test-llama2-uncensored.ipynb`
- `notebooks/Iteration1-2: testing/test_user_simulation.ipynb`
- `notebooks/Iteration3: Stage 2/experiment1_analysis.ipynb`
- `notebooks/Iteration3: Stage 2/experiment_plan.md`
- `notebooks/Iteration3: Stage 2/Stage 2.md`

## Environment notes

- Iteration 3 uses LiteLLM-based API calling for model interaction.
- Configure provider credentials (for example, OpenAI keys) in your environment before running scripts.
- Analysis scripts depend on common scientific Python packages (for example: pandas, numpy, scipy, matplotlib, seaborn).
