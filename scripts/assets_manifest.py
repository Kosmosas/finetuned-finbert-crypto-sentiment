"""Shared manifest of large files hosted outside Git.

These files exceed GitHub's 100 MB per-file limit, so they are kept out of the
repository (see ../.gitignore) and distributed via the Hugging Face Hub:

  * the fine-tuned models -> a HF *model* repo
  * the large price CSVs -> a HF *dataset* repo

The out-of-fold FinBERT checkpoints are the current weights; fold 6 is the one
that scores the evaluation window and drives every downstream result.
``fine-tuned-finbert.pth`` is the superseded single checkpoint, kept because the
paper reports an in-sample-vs-out-of-fold comparison against it.

Then:
  * upload  : python scripts/upload_assets.py            (all assets)
              python scripts/upload_assets.py finbert     (only matching paths)
  * download: python scripts/download_assets.py

Each asset records the local repo-relative ``path``, the Hugging Face
``repo_id`` + ``repo_type`` it lives in, and ``path_in_repo`` (its name on the
Hub). For the dataset we preserve the repo layout; the models sit under the
model repo root. Assets marked ``optional`` are skipped by the downloader unless
asked for by name or with ``--all``.
"""

# Hugging Face repositories that store the large assets.
HF_MODEL_REPO = "Kosmosas/BERT-bitcoin-sentiment"
HF_DATASET_REPO = "Kosmosas/BERT-bitcoin-sentiment-assets"

ASSETS = [
    {
        "path": "fine_tuned_BERT/fine-tuned-finbert.pth",   # ~419 MB superseded single checkpoint
        "repo_id": HF_MODEL_REPO,
        "repo_type": "model",
        "path_in_repo": "fine-tuned-finbert.pth",
    },
    # Out-of-fold FinBERT checkpoints (~419 MB each). Fold N scores slice N;
    # fold 6 covers 2024-06 -> 2025-07, the paper's evaluation window.
    *[
        {
            "path": f"sentiment_analysis/oof_checkpoints/finbert_oof_tone3_cond_fold{k}.pth",
            "repo_id": HF_MODEL_REPO,
            "repo_type": "model",
            "path_in_repo": f"oof/finbert_oof_tone3_cond_fold{k}.pth",
            # Not needed to rerun the pipeline: the scores these produced are
            # committed under backtesting/data/cleaned/. Fetch only to re-score.
            "optional": True,
        }
        for k in range(1, 7)
    ],
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
