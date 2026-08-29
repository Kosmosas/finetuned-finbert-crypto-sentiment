"""Exact training interval and sequence counts per feature set.

Answers: do Price + volume and Price + volume + sentiment see the same training
data? Reports the interval, the sequence counts, and every hour on which the two
sets disagree.

Run:  python diagnostics/training_data_audit.py
"""
import _common  # noqa: F401  (path bootstrap)

import pandas as pd
import lstm_common as L

PAIR = ["price+volume", "price+volume+sentiment"]


def main() -> None:
    H = L.build_feature_sets(L.SENT_FILE_TRAIN, verbose=False)
    HC = L.restrict_to_common(H)
    lo, hi = L.common_window(H)
    print(f"common window enforced on every feature set: {lo} -> {hi}")
    print(f"(ANALYSIS_START = {L.ANALYSIS_START}, the first hour with out-of-fold scores)\n")

    rows = []
    for name in L.FEATURE_SET_ORDER:
        dfh = HC[name]
        _, _, idx, _, _ = L.make_windows(dfh)
        tr = idx < L.TRAIN_END
        fit = idx < L.INNER_VAL_START
        up = L.regime_mask(idx, "Uptrend")
        sw = L.regime_mask(idx, "Sideways")
        rows.append(dict(feature_set=name, variates=dfh.shape[1], hourly_rows=len(dfh),
                         train_seqs=int(tr.sum()), fit_seqs=int(fit.sum()),
                         inner_val_seqs=int((tr & ~fit).sum()),
                         first_target=idx[tr].min(), last_target=idx[tr].max(),
                         uptrend_seqs=int(up.sum()), sideways_seqs=int(sw.sum())))
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    a, b = PAIR
    ia, ib = set(HC[a].index), set(HC[b].index)
    only_a, only_b = sorted(ia - ib), sorted(ib - ia)
    na = int(tbl.loc[tbl.feature_set == a, "train_seqs"].iloc[0])
    nb = int(tbl.loc[tbl.feature_set == b, "train_seqs"].iloc[0])
    print(f"\n{a} vs {b}")
    print(f"  identical training interval : "
          f"{tbl.loc[tbl.feature_set == a, 'first_target'].iloc[0]} -> "
          f"{tbl.loc[tbl.feature_set == a, 'last_target'].iloc[0]}")
    print(f"  training sequences          : {na:,} vs {nb:,}   "
          f"difference {abs(na - nb)} ({100 * abs(na - nb) / na:.3f}%)")
    print(f"  evaluation sequences        : uptrend "
          f"{tbl.loc[tbl.feature_set == a, 'uptrend_seqs'].iloc[0]:,} vs "
          f"{tbl.loc[tbl.feature_set == b, 'uptrend_seqs'].iloc[0]:,}, sideways "
          f"{tbl.loc[tbl.feature_set == a, 'sideways_seqs'].iloc[0]:,} vs "
          f"{tbl.loc[tbl.feature_set == b, 'sideways_seqs'].iloc[0]:,}")
    print(f"  hours only in {a:24s}: {len(only_a)}")
    print(f"  hours only in {b:24s}: {len(only_b)}")
    for t in only_a:
        print(f"     {t}")
    print(f"  intersection: {len(ia & ib):,} hours "
          f"({100 * len(ia & ib) / len(ia):.3f}% of {a})")
    print("\nTo rerun both on exactly the intersection, see "
          "itransformer_models/matched_training_check.py")


if __name__ == "__main__":
    main()
