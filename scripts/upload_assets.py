#!/usr/bin/env python3
"""Upload the large model/data files to the Hugging Face Hub (one-time, by the author).

Usage:
    pip install huggingface_hub
    huggingface-cli login                 # or set HF_TOKEN
    python scripts/upload_assets.py        # upload every asset
    python scripts/upload_assets.py finbert   # only assets whose path matches a filter

Creates each target repo if it does not exist, then uploads each asset to the
repo and path declared in scripts/assets_manifest.py. Repo cards at
scripts/model_card.md and scripts/dataset_card.md are uploaded as the model and
dataset repos' README.md.
"""
import os
import sys

from huggingface_hub import HfApi, create_repo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assets_manifest import ASSETS, HF_MODEL_REPO, repo_url  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CARDS = {
    "model": os.path.join(SCRIPT_DIR, "model_card.md"),
    "dataset": os.path.join(SCRIPT_DIR, "dataset_card.md"),
}


def main() -> None:
    filters = sys.argv[1:]
    api = HfApi()
    created = set()

    for asset in ASSETS:
        if filters and not any(f in asset["path"] for f in filters):
            continue

        src = os.path.join(REPO_ROOT, asset["path"])
        if not os.path.exists(src):
            print(f"[warn] missing locally, skipping: {asset['path']}")
            continue

        key = (asset["repo_id"], asset["repo_type"])
        if key not in created:
            create_repo(asset["repo_id"], repo_type=asset["repo_type"], exist_ok=True)
            created.add(key)
            # Attach the repo card, if one exists for this repo type.
            card = CARDS.get(asset["repo_type"])
            if card and os.path.exists(card):
                print(f"[put ] {asset['repo_type']} card -> {asset['repo_id']}/README.md")
                api.upload_file(
                    path_or_fileobj=card,
                    path_in_repo="README.md",
                    repo_id=asset["repo_id"],
                    repo_type=asset["repo_type"],
                )

        print(f"[put ] {asset['path']} -> {asset['repo_id']}/{asset['path_in_repo']}")
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=asset["path_in_repo"],
            repo_id=asset["repo_id"],
            repo_type=asset["repo_type"],
        )

    for asset in ASSETS:
        if asset["repo_id"] == HF_MODEL_REPO:
            print(f"Done. Model: {repo_url(asset)}")
            break


if __name__ == "__main__":
    main()
