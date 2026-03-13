# robustness-misinfo

Repository for collecting misinformation-focused text datasets and testing model behavior with StrongREJECT-style harmfulness evaluation.

## Project purpose

This project is used to:
- build and store misinformation-related benchmark datasets,
- probe model behavior on bias/conspiracy/fake-news prompts,
- score outputs with StrongREJECT rubric signals (refusal, convincingness, specificity, and final score).

## Repository layout

- `data/dataset/`: processed CSV datasets used for experiments.
- `data/fake/`: raw fake-news text files (e.g., `*.fake.txt`) grouped by domain.
- `notebooks/`: Jupyter notebooks for dataset construction, filtering, and model/evaluator testing.
- `models/`: model artifacts or model-related files.
- `results/`: saved outputs and experiment results.
- `scripts/`: utility scripts for data processing/experiments.

## Dataset used in experiments

Primary experiment datasets are under `data/dataset`:

- `ds_bias.csv`
	- Bias-focused misinformation statements.
	- Key columns: `index`, `content`, `type`, `bias_type`, `stereotyped_entity`.

- `ds_conspiracy.csv`
	- Conspiracy-themed statements with assigned category labels.
	- Key columns: `content`, `type`.

- `ds_fakenews.csv`
	- Fake-news headlines/articles with associated domain labels.
	- Key columns: `content`, `details`, `type`.

## Notebook summary

- `notebooks/test_dataset.ipynb`
	- Builds/curates three datasets:
		- loads and filters BiasShades into `ds_bias`,
		- constructs a conspiracy dataset from tabulated factor-loadings into `ds_conspiracy`,
		- downloads and parses FakeNewsAMT files into `ds_fakenews`.
	- Includes export steps to CSV and initial StrongREJECT evaluation setup.

- `notebooks/filter_data.ipynb`
	- Loads the processed CSV datasets and defines a StrongREJECT rubric scoring function.
	- Pulls official StrongREJECT judge templates and applies scoring to sample bias content.

- `notebooks/test_StrongREJECT.ipynb`
	- Demonstrates StrongREJECT benchmark workflow with example harmful/refusal responses.
	- Implements rubric evaluation via OpenRouter/OpenAI-compatible API and maps scores across a dataset.

- `notebooks/test-llama2-uncensored.ipynb`
	- Sends prompt examples to `llama2-uncensored` via Ollama.
	- Used for quick qualitative checks on model responses to bias and conspiracy-style prompts.

## Notes

- Some notebook cells are environment-specific (e.g., Colab drive mount or external API keys).
- Reproducibility depends on available model/API access and local package setup.
