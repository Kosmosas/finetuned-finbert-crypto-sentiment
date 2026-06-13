#!/usr/bin/env python3
"""Download the large model/data files from the Hugging Face Hub.

Usage:
    pip install huggingface_hub
    python scripts/download_assets.py

Files are placed at their correct locations inside the repository so the
notebooks find them via their existing relative paths.
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assets_manifest import ASSETS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    for asset in ASSETS:
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
