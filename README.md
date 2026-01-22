# Prompt-based LLM Classification for Stock/Time-Series Tabular Data

Notebook experiment using Hugging Face Transformers to run **few-shot (prompt-based) inference** with **OpenLLaMA 3B** on a tabular/time-series dataset, framed as a **binary classification** task.

## Overview
**Goal:** Predict a binary target (`Target` = 0/1) from a small set of tabular features by converting rows into a text prompt and applying few-shot prompting.

**Approach:** Sample a few labeled examples → build a prompt → run inference → parse the predicted label → evaluate.

## Model
- `openlm-research/open_llama_3b_v2`
- Inference only (no fine-tuning)

## What it does
- Loads a CSV dataset (**not included** in this repo)
- Builds prompts from randomly sampled labeled examples (few-shot)
- Runs inference with OpenLLaMA
- Evaluates predictions on a held-out split (binary target)

## Repository structure
- `notebook/` – main Colab/Jupyter notebook
- `docs/` – report / notes
- `data/` – local data (**ignored by git**)

## How to run (Colab)
1. Open the notebook in Google Colab
2. Run the install cell (`pip install ...`)
3. Provide the dataset CSV (see **Data** below)
4. Run all cells

## Data (not included)
The dataset is not included due to usage/licensing restrictions.

### Expected format
A CSV with:
- feature columns (the notebook uses a subset of columns)
- `Target` column with binary labels: 0/1

### Expected path
Place the file at:
- `data/Rohdaten CSV.csv`

(or update the path in the notebook.)

## Reproducibility
Few-shot sampling is random by default. For reproducible results, set a fixed random seed in the notebook before sampling examples.

## Notes / limitations
- This notebook performs **inference**, not fine-tuning.
- Prompt sampling affects results; fixed seed is recommended.
- This is an experimental baseline; next steps include stronger baselines and cleaner prompt templates.

## Next steps
- Add baseline models (e.g., logistic regression / XGBoost) for comparison
- Improve prompt template + parsing robustness
- Add cross-validation and more robust evaluation
