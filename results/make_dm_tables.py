"""Regenerate both Diebold-Mariano tables in one uniform format.

The notebooks each write their own DM table, and the two ended up reporting
different columns: the iTransformer table omits the mean loss differential, the
LSTM table omits the raw p-value. Every statistic exists in the source CSVs, so
this rewrites both with the same four: d-bar, DM, p, q_BH.

Run AFTER make_results.py, which copies the notebooks' versions into tex/;
this overwrites them.

    python make_results.py
    python make_dm_tables.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TEX = HERE / "tex"

SOURCES = {
    "itransformer": dict(
        csv=REPO / "itransformer_models" / "out" / "itransformer_dm_tests.csv",
        out=TEX / "table_itransformer_dm.tex",
        label="tab:itransformer_dm_oof",
        caption=("Diebold--Mariano tests of equal predictive accuracy for the iTransformer "
                 "forecasts, on absolute errors of the seed-averaged forecast."),
    ),
    "lstm": dict(
        csv=REPO / "lstm_models" / "out" / "dm_tests.csv",
        out=TEX / "table_lstm_dm.tex",
        label="tab:lstm_dm_tests",
        caption=("Diebold--Mariano tests of equal predictive accuracy for the LSTM "
                 "forecasts, on absolute errors of the seed-averaged forecast."),
    ),
}

NOTE = (r"\item \textit{Note:} $\bar{d}$ is the mean difference in absolute forecast error "
        r"(first model minus second), in USD. A negative DM statistic indicates lower "
        r"absolute error for the first model. The long-run variance uses a Newey--West "
        r"estimator with the Harvey--Leybourne--Newbold small-sample correction. $p$ is the "
        r"unadjusted p-value and $q_{BH}$ the Benjamini--Hochberg adjusted p-value across "
        r"the whole family of tests in this table.")


def _p(v: float) -> str:
    """p-values below the 4-decimal grid are reported as an inequality, not as 0.0000."""
    return r"$<0.0001$" if v < 1e-4 else f"{v:.4f}"


def _rows(df: pd.DataFrame) -> str:
    out = []
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r["regime"] != prev:
            out.append(r"\midrule")
        prev = r["regime"]
        comparison = " ".join(str(r["comparison"]).split())     # collapse stray whitespace
        out.append(f"{r['regime']} & {comparison} & {int(r['n']):,} & "
                   f"{r['d_mean']:.3f} & {r['DM']:.3f} & {_p(r['p'])} & {_p(r['q_BH'])} \\\\")
    return "\n".join(out)


def build(spec: dict) -> str:
    df = pd.read_csv(spec["csv"])
    missing = {"regime", "comparison", "n", "d_mean", "DM", "p", "q_BH"} - set(df.columns)
    if missing:
        raise SystemExit(f"{spec['csv'].name} is missing columns: {sorted(missing)}")
    return (
        "\\begin{table}[H]\n\\footnotesize\n\\centering\n\\begin{threeparttable}\n"
        f"\\caption{{{spec['caption']}}}\n\\label{{{spec['label']}}}\n"
        "\\renewcommand{\\arraystretch}{1.1}\n\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "\\textbf{Period} & \\textbf{Comparison} & \\textbf{n} & $\\bm{\\bar{d}}$ & "
        "\\textbf{DM} & $\\bm{p}$ & $\\bm{q_{BH}}$ \\\\\n\\midrule\n"
        + _rows(df) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\begin{tablenotes}[flushleft]\n\\footnotesize\n" + NOTE +
        "\n\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n")


def main() -> None:
    TEX.mkdir(exist_ok=True)
    for name, spec in SOURCES.items():
        spec["out"].write_text(build(spec))
        df = pd.read_csv(spec["csv"])
        print(f"{name:13s} {len(df):2d} rows -> {spec['out'].relative_to(REPO)}")
    print("\nBoth tables now report: d-bar (USD), DM, raw p, BH-adjusted q.")


if __name__ == "__main__":
    main()
