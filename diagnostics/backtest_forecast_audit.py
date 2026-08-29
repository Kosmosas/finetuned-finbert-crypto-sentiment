"""How the forecasts used for strategy-parameter tuning were produced.

The strategy search cross-validates trading parameters over the training period.
This measures how much of that period the forecaster had already been fitted on,
i.e. what fraction of the tuning signal is in-sample.

Mechanism, for reference (lstm_common._fold_predictions):
  the forecaster is fitted once on data before TRAIN_END; its predictions are
  generated once over the whole search frame; KFold(shuffle=False) then splits
  those predictions into chronological folds. The model is never refit, so there
  is no rolling-origin or expanding-window re-estimation.

Run:  python diagnostics/backtest_forecast_audit.py
"""
import _common  # noqa: F401

import pandas as pd
import lstm_common as L


def main() -> None:
    print(f"forecaster fitted on targets < TRAIN_END    = {L.TRAIN_END.date()}")
    print(f"strategy search frame is hours < BT_TRAIN_END = {L.BT_TRAIN_END.date()}")
    print(f"backtest regimes: " + ", ".join(
        f"{p['short']} {p['start'].date()}->{p['end'].date()}" for p in L.BT_PERIODS.values()))
    print()

    H = L.restrict_to_common(L.build_feature_sets(L.SENT_FILE_TRAIN, verbose=False))
    rows = []
    for name in L.FEATURE_SET_ORDER:
        tf = H[name].loc[H[name].index < L.BT_TRAIN_END]
        _, _, idx, _, _ = L.make_windows(tf)
        ins = int((idx < L.TRAIN_END).sum())
        rows.append(dict(feature_set=name, search_seqs=len(idx),
                         in_sample=ins, in_sample_pct=round(100 * ins / len(idx), 2),
                         out_of_sample=len(idx) - ins))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nthe out-of-sample remainder is the buffer month "
          f"{L.TRAIN_END.date()} -> {(L.BT_TRAIN_END - pd.Timedelta(days=1)).date()}")
    print("the reported backtests themselves start "
          f"{L.BT_PERIODS[1]['start'].date()} and remain fully out-of-sample")


if __name__ == "__main__":
    main()
