"""Shared manifest of large files hosted outside Git.

These files exceed GitHub's 100 MB per-file limit, so they are kept out of the
repository (see ../.gitignore) and distributed via the Hugging Face Hub:

  * the fine-tuned model -> a HF *model* repo
  * the large price CSVs -> a HF *dataset* repo

Then:
  * upload  : python scripts/upload_assets.py            (all assets)
              python scripts/upload_assets.py finbert     (only matching paths)
  * download: python scripts/download_assets.py

Each asset records the local repo-relative ``path``, the Hugging Face
``repo_id`` + ``repo_type`` it lives in, and ``path_in_repo`` (its name on the
Hub). For the dataset we preserve the repo layout; the model sits at the repo
root by convention.
"""

# Hugging Face repositories that store the large assets.
HF_MODEL_REPO = "Kosmosas/BERT-bitcoin-sentiment"
HF_DATASET_REPO = "Kosmosas/BERT-bitcoin-sentiment-assets"

ASSETS = [
    {
        "path": "fine_tuned_BERT/fine-tuned-finbert.pth",   # ~419 MB fine-tuned FinBERT regression weights
        "repo_id": HF_MODEL_REPO,
        "repo_type": "model",
        "path_in_repo": "fine-tuned-finbert.pth",
    },
    {
        "path": "backtesting/data/BTCUSDT-last.csv",         # ~393 MB 1-min BTCUSDT OHLCV (extended)
        "repo_id": HF_DATASET_REPO,
        "repo_type": "dataset",
        "path_in_repo": "backtesting/data/BTCUSDT-last.csv",
    },
    {
        "path": "weights_comparison_and_derive/BTCUSDT.csv", # ~363 MB 1-min BTCUSDT OHLCV (headline-impact study)
        "repo_id": HF_DATASET_REPO,
        "repo_type": "dataset",
        "path_in_repo": "weights_comparison_and_derive/BTCUSDT.csv",
    },
]


def repo_url(asset: dict) -> str:
    """Human-facing URL for an asset's repo."""
    prefix = "datasets/" if asset["repo_type"] == "dataset" else ""
    return f"https://huggingface.co/{prefix}{asset['repo_id']}"
