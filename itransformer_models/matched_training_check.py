"""Price+volume vs Price+volume+sentiment on IDENTICAL training rows.

Robustness check for the feature-set comparison.

Protocol is byte-identical to 01_itransformer_training.ipynb (same hparams, same
masks, same epoch selection, same 3 seeds). The only change: both feature sets
are restricted to the intersection of their hourly indices, so the two models see
exactly the same rows, the same windows and the same targets.
"""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import itransformer_common as ic
from itransformer_common import (
    TRAIN_END, INNER_VAL_START, UPTREND, SIDEWAYS,
    build_feature_sets, restrict_to_common, common_window, make_windows,
    regime_mask, metrics, train_eval, SENT_FILE_TRAIN, OUT, DEVICE)

SEEDS, MAX_EPOCHS, PATIENCE = [0, 1, 2], 60, 8
PAIR = ["price+volume", "price+volume+sentiment"]
HP = json.loads((OUT / "itransformer_hparams.json").read_text())

H = build_feature_sets(SENT_FILE_TRAIN, verbose=False)
HC = restrict_to_common(H)
_, common_end = common_window(H)
sideways_end = min(SIDEWAYS[1], common_end)

common_idx = HC[PAIR[0]].index
for n in PAIR:
    common_idx = common_idx.intersection(HC[n].index)
print(f"device: {DEVICE}")
print(f"matched index: {len(common_idx):,} hours  ({common_idx.min()} -> {common_idx.max()})")
for n in PAIR:
    print(f"   {n:24s} unmatched {len(HC[n]):,} -> matched {len(common_idx):,} "
          f"(dropped {len(HC[n]) - len(common_idx)})")

fit_mask   = lambda idx: idx < INNER_VAL_START
inner_mask = lambda idx: (idx >= INNER_VAL_START) & (idx < TRAIN_END)
train_mask = lambda idx: idx < TRAIN_END
test_mask  = lambda idx: (idx >= UPTREND[0]) & (idx < sideways_end)

rows, splits = [], []
t0 = time.time()
for name in PAIR:
    dfh = HC[name].loc[common_idx]
    hp = HP[name]
    X, y, idx, _, _ = make_windows(dfh)
    splits.append(dict(feature_set=name, variates=dfh.shape[1], rows=len(dfh),
                       train_seqs=int((idx < TRAIN_END).sum()),
                       fit_seqs=int((idx < INNER_VAL_START).sum()),
                       first_target=str(idx[idx < TRAIN_END].min()),
                       last_target=str(idx[idx < TRAIN_END].max())))
    preds = {}
    for seed in SEEDS:
        sel = train_eval(dfh, fit_mask, inner_mask, hp, epochs=MAX_EPOCHS,
                         seed=seed, track=True, patience=PATIENCE)
        n_ep = max(1, sel["best_epoch"] + 1)
        preds[seed] = train_eval(dfh, train_mask, test_mask, hp, epochs=n_ep, seed=seed)
        print(f"  {name:24s} seed={seed} epochs={n_ep:2d} "
              f"inner-val MAE={sel['history'][sel['best_epoch']]:8.2f}")
    for regime in ("Uptrend", "Sideways"):
        per_seed = []
        for seed in SEEDS:
            r = preds[seed]
            m = regime_mask(r["idx"], regime)
            if regime == "Sideways":
                m &= r["idx"] < sideways_end
            per_seed.append(metrics(r["pred"][m], r["y"][m], r["close_prev"][m], name, regime))
        agg = {k: float(np.mean([d[k] for d in per_seed]))
               for k in ("MAE_usd", "RMSE_usd", "corr", "dir_acc")}
        agg |= {f"{k}_sd": float(np.std([d[k] for d in per_seed]))
                for k in ("MAE_usd", "RMSE_usd", "corr", "dir_acc")}
        rows.append(dict(regime=regime, model=name, n=per_seed[0]["n"], **agg))

print(f"\ntotal: {(time.time()-t0)/60:.1f} min")
pd.DataFrame(splits).to_csv(OUT / "matched_training_split.csv", index=False)
out = pd.DataFrame(rows)
out.to_csv(OUT / "matched_training_forecasting.csv", index=False)
print("\n=== MATCHED training rows ===")
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
