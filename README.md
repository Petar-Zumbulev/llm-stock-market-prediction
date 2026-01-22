# LLM Time-Series / Stock Prediction (Colab Notebook)

Notebook-based experiment using Hugging Face Transformers to run **prompt-based (few-shot) inference** with an OpenLLaMA model on a tabular/time-series CSV dataset.

## What it does
- Loads a CSV dataset (not included in this repo)
- Builds prompts from a few randomly sampled training examples
- Runs inference with `openlm-research/open_llama_3b_v2`
- Evaluates predictions (binary target)

## Repository structure
- `notebooks/` – main Colab/Jupyter notebook
- `data/` – local data (ignored by git)

## How to run (Colab)
1. Open the notebook in Colab
2. Run the install cell (`pip install ...`)
3. Upload / provide your dataset CSV (see below)
4. Run all cells

## Data expectations

The notebook expects a CSV with:
- feature columns (the notebook uses 5 columns)
- a `Target` column with a binary label (0/1)

Place your CSV locally at:
- `data/Rohdaten CSV.csv`
(or edit the path in the notebook.)

## Notes / Limitations
- This notebook currently performs **inference**, not full fine-tuning.
- Prompt sampling is random; set a fixed seed for reproducible results.
