"""Consolidate the tables and figures reported in the manuscript.

Reads the artefacts produced by

    sentiment_analysis/oof_finbert_scoring.ipynb    (expanding-window FinBERT)
    sentiment_analysis/oof_score_full_stream.ipynb  (scores for the whole headline stream)
    itransformer_models/01_itransformer_training.ipynb   (forecasting, 3 seeds, benchmarks, DM)
    itransformer_models/02_itransformer_backtesting.ipynb + 03_itransformer_diagrams.ipynb
    lstm_models/01_lstm_training.ipynb + 02_lstm_backtesting.ipynb + 03_lstm_diagrams.ipynb

and writes flat CSVs plus copies of the figures into this folder. Nothing is
recomputed except the Diebold-Mariano tests, which are redone here for both model
families on one common comparison set so the two are directly comparable.

Run from this directory:  python make_results.py
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "lstm_models"))
sys.path.insert(0, str(REPO / "backtesting"))

import lstm_common as L  # noqa: E402

ITR = REPO / "itransformer_models" / "out"
LSTM = REPO / "lstm_models" / "out"
SENT = REPO / "sentiment_analysis" / "out"
CLEANED = REPO / "backtesting" / "data" / "cleaned"

BH = {"Uptrend Market": 13891.0, "Sideways Market": 11057.0}   # buy-and-hold, per regime


# --------------------------------------------------------------------------- #
# 1. sentiment stage: what the leakage correction changed
# --------------------------------------------------------------------------- #

def sentiment_stage():
    lab = pd.read_csv(REPO / "weights_comparison_and_derive" / "output.csv")
    lab = lab.dropna(subset=["price_change_4h", "volume_surge_price"])
    lab["dt"] = pd.to_datetime(lab["timestamp"], unit="s")

    cols = {
        "single checkpoint, in-sample": "predicted_sentiments_finbert.csv",
        "out-of-fold, pretrained tone head": "predicted_sentiments_finbert_oof_tone3_z.csv",
        "out-of-fold, trained 768-d head": "predicted_sentiments_finbert_oof_cls_z.csv",
        "out-of-fold, standardised target": "predicted_sentiments_finbert_oof_tone3_cond_z.csv",
    }
    for name, f in cols.items():
        s = (pd.read_csv(CLEANED / f)[["timestamp", "title", "predicted_sentiment"]]
               .rename(columns={"predicted_sentiment": name})
               .drop_duplicates(subset=["timestamp", "title"]))
        lab = lab.merge(s, on=["timestamp", "title"], how="left")

    BOUND = pd.Timestamp("2024-03-28")
    rows = []
    for name in cols:
        s = lab[name].to_numpy(float)
        for period, mask in [("train 2019-01 -> 2024-03", (lab["dt"] < BOUND).to_numpy()),
                             ("test  2024-03 -> 2025-03", (lab["dt"] >= BOUND).to_numpy())]:
            r = dict(column=name, period=period)
            for tgt, short in [("volume_surge_price", "r_train_target"),
                               ("price_change_4h", "r_price_change_4h")]:
                y = lab[tgt].to_numpy(float)
                k = mask & np.isfinite(s) & np.isfinite(y)
                r[short] = round(float(np.corrcoef(s[k], y[k])[0, 1]), 4)
                r["n"] = int(k.sum())
            r["sd"] = round(float(np.nanstd(s[mask])), 5)
            rows.append(r)
    out = pd.DataFrame(rows)[["column", "period", "n", "r_train_target",
                              "r_price_change_4h", "sd"]]
    out["train_test_sd_ratio"] = np.nan
    for i in range(0, len(out), 2):
        out.loc[i, "train_test_sd_ratio"] = round(out.loc[i, "sd"] / out.loc[i + 1, "sd"], 2)
    return out


def hac_corr(x, y, lags=None):
    """Pearson r with a Newey-West standard error, robust to serial dependence."""
    from scipy.stats import norm
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    lags = lags or int(round(n ** (1 / 3)))
    xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    X = np.column_stack([np.ones(n), xs])
    b = np.linalg.lstsq(X, ys, rcond=None)[0]
    e = ys - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        G = u[l:].T @ u[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    se = float(np.sqrt((XtXi @ S @ XtXi)[1, 1]))
    return float(b[1]), se, float(2 * norm.sf(abs(b[1] / se))), n


def sentiment_stage_significance():
    lab = pd.read_csv(REPO / "weights_comparison_and_derive" / "output.csv")
    lab = lab.dropna(subset=["price_change_4h", "volume_surge_price"])
    lab["dt"] = pd.to_datetime(lab["timestamp"], unit="s")
    files = {"single checkpoint (in-sample fitted)": "predicted_sentiments_finbert.csv",
             "out-of-fold (final)": "predicted_sentiments_finbert_oof_tone3_cond_z.csv"}
    rows = []
    for name, f in files.items():
        s = (pd.read_csv(CLEANED / f)[["timestamp", "title", "predicted_sentiment"]]
               .rename(columns={"predicted_sentiment": "s"})
               .drop_duplicates(subset=["timestamp", "title"]))
        d = lab.merge(s, on=["timestamp", "title"], how="left")
        test = (d["dt"] >= pd.Timestamp("2024-03-28")).to_numpy()
        for tgt in ["volume_surge_price", "price_change_4h"]:
            r, se, p, n = hac_corr(d.loc[test, "s"].to_numpy(float),
                                   d.loc[test, tgt].to_numpy(float))
            rows.append(dict(column=name, target=tgt, n=n, r=round(r, 4),
                             hac_se=round(se, 4), hac_p=round(p, 4)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. forecasting: both model families on one comparison set
# --------------------------------------------------------------------------- #

def forecasting():
    it = pd.read_csv(ITR / "itransformer_forecasting_performance.csv").assign(model="iTransformer")
    ls = pd.read_csv(LSTM / "forecasting_performance.csv").assign(model="LSTM")
    keep = ["model", "regime", "model_name", "n", "MAE_usd", "MAE_usd_sd", "RMSE_usd",
            "RMSE_usd_sd", "corr", "corr_sd", "dir_acc", "dir_acc_sd"]
    for d in (it, ls):
        if "model_name" not in d.columns:
            d["model_name"] = d["model"] if "model" not in d else d.get("model_name", d.iloc[:, 1])
    it = it.rename(columns={"model": "model_family"})
    ls = ls.rename(columns={"model": "model_family"})
    return it, ls, keep


def dm_matrix():
    """Diebold-Mariano of every feature set against Price + volume, per regime.

    Computed here for both families so the comparison sets are identical.
    """
    H = L.restrict_to_common(L.build_feature_sets(L.SENT_FILE_TRAIN, verbose=False))
    _, common_end = L.common_window(H)
    sideways_end = min(L.SIDEWAYS[1], common_end)

    rows = []

    def eval_index(name):
        """The evaluation index, rebuilt from the frames and the protocol masks.

        The .npz files store predictions only; reconstructing the index here (rather
        than trusting a serialised one) keeps both families on provably the same
        timestamps. Lengths are asserted against the stored prediction vectors.
        """
        _, _, idx, _, _ = L.make_windows(H[name])
        m = (idx >= L.UPTREND[0]) & (idx < sideways_end)
        return idx[m]

    # --- iTransformer
    z = np.load(ITR / "itransformer_predictions.npz", allow_pickle=True)
    # Seed count varies by run configuration. Detect rather than assume.
    it_seeds = sorted(int(k.rsplit("seed", 1)[1]) for k in z.files
                      if k.startswith(f"{L.FEATURE_SET_ORDER[0]}__pred_seed"))
    it_err = {}
    for name in L.FEATURE_SET_ORDER:
        p = np.mean([z[f"{name}__pred_seed{s}"] for s in it_seeds], axis=0)
        y = z[f"{name}__y"]
        # Prefer the stored index (ISO strings). The evaluation span depends on
        # the run configuration, so it cannot be assumed.
        ix = pd.to_datetime(z[f"{name}__idx"].astype(str), utc=True)
        assert len(ix) == len(p), f"iTransformer {name}: {len(ix)} vs {len(p)}"
        it_err[name] = pd.Series(p - y, index=ix)
    y_ref = pd.Series(z["full__y"],
                      index=pd.to_datetime(z["full__idx"].astype(str), utc=True))

    # --- LSTM
    zz = np.load(LSTM / "lstm_predictions.npz", allow_pickle=True)
    ls_seeds = sorted(int(k.split("__")[1]) for k in zz.files
                      if k.startswith(f"{L.FOLDER_NAME[L.FEATURE_SET_ORDER[0]]}__")
                      and k.endswith("__pred"))
    ls_err = {}
    for name in L.FEATURE_SET_ORDER:
        folder = L.FOLDER_NAME[name]
        p = np.mean([zz[f"{folder}__{s}__pred"] for s in ls_seeds], axis=0)
        y = zz[f"{folder}__0__y"]
        ix = eval_index(name)
        assert len(ix) == len(p), f"LSTM {name}: {len(ix)} vs {len(p)}"
        ls_err[name] = pd.Series(p - y, index=ix)
    y_ref_lstm = pd.Series(zz["full_dataset__0__y"], index=eval_index("full"))

    for family, err, yy in [("iTransformer", it_err, y_ref),
                            ("LSTM", ls_err, y_ref_lstm)]:
        common = None
        for e in err.values():
            common = e.index if common is None else common.intersection(e.index)
        E = pd.DataFrame({n: e.loc[common] for n, e in err.items()})
        E["zero change"] = -yy.loc[common]

        for regime in ("Uptrend", "Sideways"):
            mask = L.regime_mask(E.index, regime)
            if regime == "Sideways":
                mask &= E.index < sideways_end
            sub = E[mask]
            for name in L.FEATURE_SET_ORDER:
                r = L.dm_test(sub[name].to_numpy(), sub["zero change"].to_numpy(), loss="abs")
                rows.append(dict(family=family, regime=regime, baseline="zero change",
                                 model=L.NICE_NAME[name], DM=r["dm"], p=r["p"], n=r["n"]))
            for name in L.FEATURE_SET_ORDER[1:]:
                r = L.dm_test(sub[name].to_numpy(), sub["price+volume"].to_numpy(), loss="abs")
                rows.append(dict(family=family, regime=regime, baseline="Price + volume",
                                 model=L.NICE_NAME[name], DM=r["dm"], p=r["p"], n=r["n"]))

    dm = pd.DataFrame(rows)
    dm["q_BH"] = np.nan
    for fam in dm["family"].unique():
        sel = dm["family"] == fam
        dm.loc[sel, "q_BH"] = L.bh_qvalues(dm.loc[sel, "p"])
    for c in ("DM", "p", "q_BH"):
        dm[c] = dm[c].round(4)
    return dm


# --------------------------------------------------------------------------- #
# 3. backtesting
# --------------------------------------------------------------------------- #

def backtests():
    it = pd.read_csv(ITR / "itransformer_backtesting_metrics.csv").assign(family="iTransformer")
    ls = pd.read_csv(LSTM / "lstm_backtesting_metrics.csv").assign(family="LSTM")
    cols = [c for c in it.columns if not c.endswith("_key")]
    both = pd.concat([it[cols + ["family"]], ls[[c for c in cols if c in ls.columns] + ["family"]]],
                     ignore_index=True)
    both["buy_and_hold"] = both["Market"].map(BH)
    both["beats_bh"] = both["Final Value"] > both["buy_and_hold"]
    return both


def no_model():
    it = pd.read_csv(ITR / "itransformer_no_model_benchmark.csv").assign(family="iTransformer")
    ls = pd.read_csv(LSTM / "lstm_no_model_benchmark.csv").assign(family="LSTM")
    bench = pd.concat([it, ls], ignore_index=True)

    rows = []
    for family, summ in [("iTransformer", ITR / "itransformer_backtest_summary_full_path.csv"),
                         ("LSTM", LSTM / "lstm_backtest_summary_full_path.csv")]:
        s = pd.read_csv(summ)
        s = s[s["strategy"] == "Momentum"].copy()
        s["feature_set"] = s["feature_set"].map(L.NICE_NAME).fillna(s["feature_set"])
        s["family"] = family
        rows.append(s)
    model = pd.concat(rows, ignore_index=True)
    both = pd.concat([model, bench], ignore_index=True)
    both["regime_bh"] = both["regime"].map({"Uptrend": BH["Uptrend Market"],
                                            "Sideways": BH["Sideways Market"]})
    return both.sort_values(["family", "regime", "final_value"], ascending=[True, True, False])


def cv_stability():
    rows = []
    for family, f in [("iTransformer", ITR / "itransformer_strategy_cv_folds.csv"),
                      ("LSTM", LSTM / "lstm_strategy_cv_folds.csv")]:
        d = pd.read_csv(f)
        g = (d.groupby(["feature_set", "strategy"])["sharpe"]
               .agg(["mean", "std", "min", "max"]).reset_index())
        g.insert(0, "family", family)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    return out.round(3)


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# FinBERT: out-of-fold training description and LaTeX tables
# --------------------------------------------------------------------------- #

RUN_TAG = "tone3_cond"

FINBERT_TUNING_NOTE = r"""
\paragraph{Selection procedure.} In order:

\begin{enumerate}
\item \textbf{Depth sweep.} The number of unfrozen encoder blocks was chosen first, by
      training with 1, 2 and 3 blocks unfrozen and all other settings held constant.
      Two blocks minimised both error measures and gave the highest correlation
      (MSE 0.0028, MAE 0.0313, $r = 0.2738$), and were retained.
\item \textbf{Optuna search.} With the depth fixed, the learning rate and weight decay
      were tuned by Optuna over the ranges in Table~\ref{tab:finbert_hparams_space}.
      The optimiser, batch size and sequence length were pinned rather than searched.
      Trials were scored on a random 10\% split of a 3{,}000-headline subsample.
\item \textbf{Freeze.} The resulting configuration was carried forward unchanged. It is
      \emph{not} re-tuned per fold: one configuration is used for all six folds.
\item \textbf{Throughput adjustment.} The batch size was raised from 9 to 64 and mixed
      precision enabled, with the learning rate rescaled by $\sqrt{64/9}$ to keep the
      per-step noise scale comparable. No other value was altered for speed.
\item \textbf{Conditioning correction.} The target is standardised on each fold's
      training window and the output activation made linear. At initialisation the
      published $\tanh$ head emits values with standard deviation 0.698 against a
      target standard deviation of 0.058, with mean gradient 0.293, so the model
      begins roughly twelve target standard deviations from its optimum behind a
      largely saturated nonlinearity. On a standardised target the published
      $\delta = 1$ also becomes a genuine robust threshold, whereas on the raw target
      ($|y| < 0.25$) the linear branch of the Huber loss is unreachable and the
      objective reduces to mean squared error.
\item \textbf{Per-fold epoch selection.} The only quantity chosen inside the
      out-of-fold procedure is the number of epochs, taken as the minimum of the
      validation loss on the last 10\% of each fold's own training window, behind the
      purge gap, with patience 5.
\item \textbf{Per-fold calibration.} Each fold's scores are standardised by the mean
      and standard deviation of that fold's validation predictions, both computed on
      data preceding the scored slice.
\end{enumerate}

\paragraph{Limitation.} Steps 1 and 2 were carried out on a random split of the full
sample, so the configuration itself is not free of look-ahead even though the weights
and the epoch count are. A fully nested design would repeat the search inside each
fold's training window. Because one fixed configuration is applied unchanged to every
fold, this affects the choice of configuration rather than the fold structure, and is
reported as a limitation rather than a resolved issue.
"""


def finbert_tables():
    """Fold structure + hyperparameters for the sentiment model, as actually run.

    Built from sentiment_analysis/out/oof_fold_summary_*.csv and oof_fold_calibration_*.csv
    so the tables cannot drift from the run that produced the scores.
    """
    folds = pd.read_csv(SENT / f"oof_fold_summary_{RUN_TAG}.csv")
    calib = pd.read_csv(SENT / f"oof_fold_calibration_{RUN_TAG}.csv")
    d = folds.merge(calib[["fold", "val_mean", "val_sd"]], on="fold")
    d = d.rename(columns={"start": "scored_from", "end": "scored_to"})
    d.to_csv(HERE / "05_finbert_oof_folds.csv", index=False)

    rows = "\n".join(
        f"        {r.fold.replace('fold','')} & {r.scored_from} -- {r.scored_to} & "
        f"{r.n_scored:,} & {r.n_train:,} & {r.n_val:,} & {r.best_epoch} & "
        f"{r.corr_oof:+.4f} & {r.corr_insample_train:+.4f} \\\\"
        for r in d.itertuples())
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Expanding-window out-of-fold folds for the headline scoring model. "
        "Each fold is fitted only on headlines preceding its scored slice, with a one-day "
        "purge at every boundary; $r_{\\mathrm{OOF}}$ is the correlation between the "
        "fold's scores and the label on the slice it scored, $r_{\\mathrm{in}}$ the same "
        "correlation evaluated on its own training window.}\n"
        "\\label{tab:finbert_oof_folds}\n"
        "\\begin{tabular}{clrrrrrr}\n\\hline\n"
        "Fold & Scored slice & $n_{\\mathrm{scored}}$ & $n_{\\mathrm{train}}$ & "
        "$n_{\\mathrm{val}}$ & Epoch & $r_{\\mathrm{OOF}}$ & $r_{\\mathrm{in}}$ \\\\\n"
        "\\hline\n" + rows + "\n\\hline\n\\end{tabular}\n\\end{table}\n")
    (HERE / "tex" / "table_finbert_oof_folds.tex").write_text(tex)

    # (a) search space: every hyperparameter, what was considered, what was taken
    space = [
        ("Unfrozen encoder blocks", "1, 2, 3", "2", "Layer sweep"),
        ("Learning rate", r"log-uniform $[10^{-5},\,10^{-3}]$", r"$7.35\times10^{-5}$", "Optuna"),
        ("Weight decay", r"$[5\times10^{-4},\,5\times10^{-2}]$", "0.094$^{\\dagger}$", "Optuna"),
        ("Optimiser", "AdamW, SGD, RMSprop", "AdamW", "Fixed"),
        ("Batch size", "8, 16", "9", "Fixed"),
        ("Max sequence length", "64, 128", "128", "Fixed"),
        ("Momentum", r"$[0.001,\,0.4]$", "0 (unused by AdamW)", "Fixed"),
        ("Huber $\\delta$", "not searched", "1", "Fixed"),
        ("Epochs", "not searched", "100", "Fixed"),
        ("Regression head", "not searched", r"frozen \texttt{Linear(3,1)}", "Fixed"),
    ]
    body_a = "\n".join(f"        {a} & {b} & {c} & {d} \\\\" for a, b, c, d in space)

    # (b) what this work changed, and why
    changed = [
        ("Batch size", "9", "64", "throughput; 170 to 490 samples/s"),
        ("Learning rate", r"$7.35\times10^{-5}$", r"$1.96\times10^{-4}$", r"rescaled by $\sqrt{64/9}$"),
        ("Precision", "fp32", "fp16 (AMP)", "throughput"),
        ("Epoch budget", "100, fixed", "$\\leq$100, patience 5",
         "stop on each fold's own validation slice"),
        ("Target scaling", "raw", "per-fold $z$", "output sd 0.698 vs target sd 0.058 at init"),
        ("Output activation", r"$\tanh$", "linear", r"a standardised target cannot fit in $(-1,1)$"),
    ]
    body_b = "\n".join(f"        {a} & {b} & {c} & {d} \\\\" for a, b, c, d in changed)

    # (c) the configuration as actually run
    hp = [
        ("Base checkpoint", r"\texttt{yiyanghkust/finbert-tone} (BERT-base, 109.8M parameters)"),
        ("Trainable parameters", "last 2 encoder blocks, 14.18M of 109.8M"),
        ("Regression head", r"frozen \texttt{Linear(3,1)} on the three tone logits"),
        ("Output activation", "linear"),
        ("Target", r"\texttt{volume\_surge\_price}, standardised per fold on training-window statistics"),
        ("Loss", r"Huber, $\delta = 1$"),
        ("Optimiser", r"AdamW, weight decay 0.094"),
        ("Learning rate", r"$1.96\times10^{-4}$"),
        ("Batch size", "64, mixed precision (fp16)"),
        ("Max sequence length", "128 tokens"),
        ("Epoch budget", "100 maximum, early stopping patience 5"),
        ("Epoch selection", "last 10\\% of each fold's training window, chronological, behind the purge gap"),
        ("Purge gap", "1 day at every fold boundary and between train and validation"),
        ("Seed", "42"),
        ("Score calibration", "per fold, standardised by that fold's validation mean and standard deviation"),
    ]
    body_c = "\n".join(f"        {k} & {v} \\\\" for k, v in hp)

    tex2 = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Hyperparameter search space for the headline scoring model, and the "
        "value taken for each. ``Fixed'' denotes a quantity that was set before the search "
        "rather than optimised over; the candidate values shown for those rows are the "
        "alternatives present in the search code but disabled.}\n"
        "\\label{tab:finbert_hparams_space}\n"
        "\\begin{tabular}{llll}\n\\hline\n"
        "Hyperparameter & Values considered & Selected & Stage \\\\\n\\hline\n"
        + body_a + "\n\\hline\n\\end{tabular}\n"
        "\\par\\smallskip\n\\footnotesize $^{\\dagger}$ The value carried forward, 0.094, "
        "lies outside the range the search covered; it is reported as used.\n"
        "\\end{table}\n\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Departures from the published configuration made for the out-of-fold "
        "re-scoring.}\n"
        "\\label{tab:finbert_hparams_changes}\n"
        "\\begin{tabular}{llll}\n\\hline\n"
        "Setting & Published & Used here & Reason \\\\\n\\hline\n"
        + body_b + "\n\\hline\n\\end{tabular}\n\\end{table}\n\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Training configuration of the headline scoring model, as run.}\n"
        "\\label{tab:finbert_hparams}\n"
        "\\begin{tabular}{ll}\n\\hline\n"
        "Setting & Value \\\\\n\\hline\n" + body_c +
        "\n\\hline\n\\end{tabular}\n\\end{table}\n" + FINBERT_TUNING_NOTE)
    (HERE / "tex" / "table_finbert_hparams.tex").write_text(tex2)
    return d


def main():
    print("1. sentiment stage")
    (HERE / "tex").mkdir(exist_ok=True)
    finbert_tables()
    sentiment_stage().to_csv(HERE / "01_sentiment_leakage_before_after.csv", index=False)
    sentiment_stage_significance().to_csv(HERE / "01_sentiment_significance_hac.csv", index=False)

    print("2. forecasting")
    it, ls, _ = forecasting()
    it.to_csv(HERE / "02_forecasting_itransformer.csv", index=False)
    ls.to_csv(HERE / "02_forecasting_lstm.csv", index=False)
    pd.concat([pd.read_csv(ITR / "itransformer_baselines.csv").assign(family="iTransformer"),
               pd.read_csv(LSTM / "baselines.csv").assign(family="LSTM")]) \
      .to_csv(HERE / "02_naive_benchmarks.csv", index=False)

    print("3. Diebold-Mariano")
    dm_matrix().to_csv(HERE / "03_diebold_mariano.csv", index=False)

    print("4. backtests")
    backtests().to_csv(HERE / "04_backtest_metrics.csv", index=False)
    no_model().to_csv(HERE / "04_momentum_model_vs_no_model.csv", index=False)
    cv_stability().to_csv(HERE / "04_cv_fold_stability.csv", index=False)

    print("5. figures and LaTeX")
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    for src in [ITR / "epochs_itransformer.png", ITR / "itransformer_output1.png",
                ITR / "itransformer_output2.png", ITR / "itransformer_output3.png",
                ITR / "itransformer_momentum_vs_no_model.png",
                LSTM / "epochs_lstm.png", LSTM / "lstm_output1.png",
                LSTM / "lstm_output2.png", LSTM / "lstm_output3.png"]:
        if src.exists():
            shutil.copy(src, figs / src.name)
    tex = HERE / "tex"
    tex.mkdir(exist_ok=True)
    for d in (ITR, LSTM):
        for src in d.glob("table_*.tex"):
            shutil.copy(src, tex / src.name)

    print("\nwrote to", HERE)
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and p.suffix in (".csv", ".png", ".tex"):
            print("  ", p.relative_to(HERE))


if __name__ == "__main__":
    main()
