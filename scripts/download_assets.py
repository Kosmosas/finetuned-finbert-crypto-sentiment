#!/usr/bin/env python3
"""Download the large model/data files from the Hugging Face Hub.

Usage:
    pip install huggingface_hub
    python scripts/download_assets.py           # everything needed to run the code
    python scripts/download_assets.py --all     # including the optional checkpoints
    python scripts/download_assets.py oof       # only assets whose path matches a filter

Files are placed at their correct locations inside the repository so the
notebooks find them via their existing relative paths.

Assets flagged ``optional`` in assets_manifest.py (the ~2.5 GB of out-of-fold
FinBERT checkpoints) are skipped by default: the scores they produced are already
committed under backtesting/data/cleaned/, so they are only needed to re-score
headlines from scratch.
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assets_manifest import ASSETS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    args = sys.argv[1:]
    fetch_all = "--all" in args
    filters = [a for a in args if not a.startswith("-")]

    for asset in ASSETS:
        if asset.get("optional") and not fetch_all:
            if not any(f in asset["path"] for f in filters):
                print(f"[opt ] optional, skipping: {asset['path']}")
                continue
        elif filters and not any(f in asset["path"] for f in filters):
            continue

        dest = os.path.join(REPO_ROOT, asset["path"])
        if os.path.exists(dest):
            print(f"[skip] already present: {asset['path']}")
            continue
        print(f"[get ] {asset['path']}  <-  {asset['repo_id']}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cached = hf_hub_download(
            repo_id=asset["repo_id"],
            repo_type=asset["repo_type"],
            filename=asset["path_in_repo"],
        )
        shutil.copy2(cached, dest)
        print(f"       -> {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
