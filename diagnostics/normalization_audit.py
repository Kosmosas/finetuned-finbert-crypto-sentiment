"""Preprocessing and normalisation actually applied to the forecasting inputs.

Two stages are applied to the iTransformer, one to the LSTM:

  1. an external per-variate z-score (lstm_common._standardise), fitted on the
     training windows only and applied unchanged to the evaluation split;
  2. RevIN inside the iTransformer (use_reversible_instance_norm=True): per
     variate, per 168-hour window, learnable affine, variance floor eps=1e-5.
     The LSTM has no RevIN.

Predictions are returned to USD as pred * sd_train + mean_train.

This script measures why both stages are needed: RevIN's variance floor silently
under-scales any variate whose within-window variance falls below eps, and the
headline-score column did exactly that before the external scaler was added.

Run:  python diagnostics/normalization_audit.py
"""
import _common  # noqa: F401

import numpy as np
import pandas as pd
import lstm_common as L

EPS = 1e-5          # iTransformer/revin.py: var.clamp(min=eps).rsqrt()
SENT_COL = "predicted_sentiment_24h_avg"


def _share_below_eps(sent_file: str) -> pd.DataFrame:
    H = L.restrict_to_common(L.build_feature_sets(sent_file, verbose=False))
    rows = []
    for name in L.FEATURE_SET_ORDER:
        dfh = H[name]
        X, y, idx, _, _ = L.make_windows(dfh)
        tr = idx < L.TRAIN_END
        Xtr = X[tr]
        Xs, _, _, mu, sd, ymu, ysd = L._standardise(Xtr, Xtr[:1], y[tr])
        raw_v, std_v = Xtr.var(axis=1), Xs.var(axis=1)
        r = dict(feature_set=name,
                 all_variates_raw_pct=round(100 * float((raw_v < EPS).mean()), 2),
                 all_variates_std_pct=round(100 * float((std_v < EPS).mean()), 2))
        if SENT_COL in dfh.columns:
            j = list(dfh.columns).index(SENT_COL)
            r |= dict(sentiment_sd=float(Xtr[:, :, j].std()),
                      sentiment_raw_pct=round(100 * float((raw_v[:, j] < EPS).mean()), 2),
                      sentiment_std_pct=round(100 * float((std_v[:, j] < EPS).mean()), 2))
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    print("external scaler: per-variate mean/sd over TRAINING windows only;")
    print("                 the evaluation split is transformed with those same statistics.")
    print("target         : standardised separately on the training targets;")
    print("                 predictions returned to USD as pred * sd_train + mean_train.")
    print(f"RevIN          : per variate, per window, learnable affine, eps = {EPS:g}\n")

    print(f"Share of (window, variate) pairs whose within-window variance < eps.")
    print("A variate below the floor is passed to the model with its scale destroyed.\n")

    print("--- current out-of-fold, per-fold calibrated headline scores ---")
    print(_share_below_eps(L.SENT_FILE_TRAIN).to_string(index=False))

    print("\n--- earlier single-checkpoint headline scores, for contrast ---")
    try:
        print(_share_below_eps("predicted_sentiments_finbert.csv").to_string(index=False))
    except FileNotFoundError as e:
        print(f"[skip] {e}")
    print("\n-> the external scaler keeps every variate inside RevIN's valid range;")
    print("   RevIN handles non-stationarity within each window. The stages are complementary.")


if __name__ == "__main__":
    main()
