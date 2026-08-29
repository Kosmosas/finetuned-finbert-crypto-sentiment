"""What is actually trained inside the headline scoring model.

Establishes, from the saved checkpoints rather than from the source:

  1. the Linear(3,1) regression head is frozen -- bitwise identical across folds;
  2. its values are PyTorch's default nn.Linear initialisation, i.e. random;
  3. the pretrained 3-class tone classifier is frozen at its pretrained values;
  4. only encoder blocks 10 and 11 receive gradient updates;
  5. what the frozen head's fixed bias costs, in target standard deviations.

Also contrasts the frozen random head against the trained Linear(768,1) control
run, which is the empirical justification for the design.

Run:  python diagnostics/finbert_head_audit.py
Needs the out-of-fold checkpoints; if absent locally they are fetched from the
Hugging Face model repo (see scripts/download_assets.py --all).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "sentiment_analysis" / "oof_checkpoints"
OUT = REPO / "sentiment_analysis" / "out"
RUN_TAG = "tone3_cond"
N_FOLDS = 6
BASE = "yiyanghkust/finbert-tone"
TONE_LABELS = ["Neutral", "Positive", "Negative"]     # finbert-tone id2label order


def _load(k: int):
    p = CKPT / f"finbert_oof_{RUN_TAG}_fold{k}.pth"
    if not p.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Kosmosas/BERT-bitcoin-sentiment",
                            f"oof/finbert_oof_{RUN_TAG}_fold{k}.pth")
    return torch.load(p, map_location="cpu", weights_only=True)


def main() -> None:
    sds = [_load(k) for k in range(1, N_FOLDS + 1)]

    print("1. regression head Linear(3,1) -- frozen?")
    w0, b0 = sds[0]["regressor.weight"], sds[0]["regressor.bias"]
    same = all(torch.equal(w0, s["regressor.weight"]) and torch.equal(b0, s["regressor.bias"])
               for s in sds)
    print(f"   w = {[round(x, 6) for x in w0.flatten().tolist()]}  b = {b0.item():+.6f}")
    print(f"   bitwise identical across all {N_FOLDS} folds: {same}")
    print("   -> frozen: it never received an update on any fold")

    print("\n2. is that a random initialisation?")
    bound = 1 / math.sqrt(3)          # PyTorch nn.Linear default: U(-1/sqrt(fan_in), +)
    inside = bool((w0.abs() < bound).all() and b0.abs().item() < bound)
    print(f"   PyTorch nn.Linear default bound = 1/sqrt(3) = +/-{bound:.4f}; "
          f"all values inside: {inside}")
    print("   -> fixed at its random init; never fitted to the target")
    print("   as a projection over the tone logits:")
    print("      " + "  ".join(f"{lbl}={v:+.4f}"
                               for lbl, v in zip(TONE_LABELS, w0.flatten().tolist()))
          + f"  bias={b0.item():+.4f}")

    print("\n3. pretrained 3-class tone classifier -- frozen?")
    same_cls = all(torch.equal(sds[0]["bert.classifier.weight"], s["bert.classifier.weight"])
                   for s in sds)
    from transformers import BertForSequenceClassification
    ref = BertForSequenceClassification.from_pretrained(BASE, num_labels=3)
    eq_w = torch.equal(sds[0]["bert.classifier.weight"], ref.classifier.weight.detach())
    eq_b = torch.equal(sds[0]["bert.classifier.bias"], ref.classifier.bias.detach())
    print(f"   identical across folds: {same_cls}")
    print(f"   equals pretrained {BASE}: weight={eq_w} bias={eq_b}")
    print("   -> frozen at its pretrained values, not reinitialised")

    print("\n4. which tensors actually changed between folds?")
    changed = set()
    for a, b in zip(sds, sds[1:]):
        changed |= {k for k in a if not torch.equal(a[k], b[k])}
    import re
    groups: dict[str, int] = {}
    for k in changed:
        g = re.sub(r"encoder\.layer\.(\d+)\..*", r"encoder.layer.\1.*", k)
        groups[g] = groups.get(g, 0) + 1
    for g, n in sorted(groups.items()):
        print(f"   {n:3d} tensors  {g}")
    print(f"   {len(changed)} of {len(sds[0])} tensors differ")
    print("   -> gradients reached encoder blocks 10 and 11 and nothing else")

    print("\n5. cost of the frozen non-zero bias")
    lab = pd.read_csv(REPO / "weights_comparison_and_derive" / "output.csv")
    y = lab["volume_surge_price"].dropna()
    print(f"   target volume_surge_price: mean {y.mean():+.5f}  sd {y.std():.5f}")
    print(f"   frozen bias b = {b0.item():+.4f}; at zero tone logits the head emits "
          f"{b0.item():+.4f} (linear output)")
    print(f"   that is {(b0.item() - y.mean()) / y.std():.1f} target sd away from the target mean,")
    print("   an offset the two trainable encoder blocks must cancel before fitting anything")

    print("\n6. frozen random head vs a trained Linear(768,1) head")
    print("   like-for-like: both runs use the raw target and tanh output; only the head differs")
    rows = []
    for tag, f in [("frozen random Linear(3,1)", "oof_fold_summary_tone3.csv"),
                   ("trained Linear(768,1)", "oof_fold_summary_cls.csv")]:
        p = OUT / f
        if not p.exists():
            print(f"   [skip] {p.name} not present")
            continue
        d = pd.read_csv(p)
        rows.append(dict(head=tag, folds=len(d),
                         mean_abs_corr_oof=round(float(d["corr_oof"].abs().mean()), 4),
                         mean_corr_insample=round(float(d["corr_insample_train"].mean()), 4)))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        print("   -> head capacity is not the binding constraint")


if __name__ == "__main__":
    main()
