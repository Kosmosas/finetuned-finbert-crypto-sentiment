# iTransformer forecasting and backtesting

The iTransformer half of the study: headline sentiment plus price, volume and
technical features feeding a multivariate transformer, evaluated as a forecaster
and then as a trading signal.

`lstm_models/` is the same study with a stacked LSTM in place of the
iTransformer. Both families share one protocol module, so "same target,
lookback, periods and preprocessing" is guaranteed by construction rather than
asserted.

## Notebooks

Run them in order.

| Notebook | What it does | Runtime |
|---|---|---|
| `01_itransformer_training.ipynb` | Builds the four feature sets, runs an Optuna study per set, trains 3 seeds per set, and reports forecasting statistics, naive benchmarks and Diebold–Mariano tests | ~1.5 h on a 12 GB GPU |
| `02_itransformer_backtesting.ipynb` | Tunes the three trading strategies by 10-fold CV + Optuna on the training period, then backtests both held-out regimes | ~50 min, CPU-bound |
| `03_itransformer_diagrams.ipynb` | Portfolio-trajectory figures per strategy, the backtesting metrics table, and the LaTeX drop-ins | ~1 min |

`itransformer_common.py` holds the model, the training loop and the
iTransformer-specific hyperparameters. Everything else — data loading, feature
construction, metrics, benchmarks, statistical tests and the CV + Optuna
strategy search — is imported from `../lstm_models/lstm_common.py`. The trading
strategies themselves come from `../backtesting/backtest_functions.py`, so both
model families are evaluated by literally the same strategy code.

## Protocol

| | |
|---|---|
| Feature sets | `price+volume` (4 variates), `+sentiment` (5), `+technicals` (10), `full` (11) |
| Lookback | 168 hours; target = next-hour change of the BTCUSDT close, in USD |
| Sentiment | out-of-fold, from `../sentiment_analysis/oof_finbert_scoring.ipynb` |
| Date window | one window common to all four sets, so row-wise deletion cannot shift the split |
| Split | training ends **2024-04-30**; May 2024 is a buffer month |
| Test regimes | uptrend Jun 2024 – Jan 2025; sideways Jan 2025 – end of sentiment coverage |
| Scaling | inputs and target standardised on training-period statistics |
| Epoch selection | on a 120-day inner-validation slice at the end of the training period |
| Seeds | 3 per feature set; tables report mean ± sd |
| Benchmarks | zero-change, training historical mean, random walk, AR(1), AR(24) |
| Tests | Diebold–Mariano on absolute errors, Newey–West variance, HLN correction, BH across the family |

The backtesting stage uses the wider `_ps2` sentiment column and the validation
periods from `../backtesting/data_loader.py`, so that the two model families'
backtesting tables line up row by row. Both periods remain out-of-sample: the
models are fitted on data before 2024-04-30.

## Outputs

Everything lands in `out/`: one CSV per table, combined
`itransformer_forecasting_tables.xlsx`, the epoch-selection and portfolio
figures, and the LaTeX drop-ins (`table_itransformer_*.tex`). Trade logs go to
`itransformer_backtest_results/<feature_set>/`, under the same file names as the
LSTM run. `../results/make_results.py` consolidates both families into the
tables and figures the manuscript reports.

## Matched-training-set check

`matched_training_check.py` reruns Price + volume and Price + volume + sentiment
on the *intersection* of their hourly indices, so the two models see exactly the
same rows, windows and targets. The two sets already share one date window
(2020-01-01 onward, enforced by `ANALYSIS_START`); they differ only in 8 hours out
of 50,290 that the sentiment column is missing. Matching them moves MAE by at
most 0.5 USD, inside the seed spread. Outputs: `out/matched_training_split.csv`,
`out/matched_training_forecasting.csv`. Runtime ~1.5 min.

## Reproducibility caveat

`02_itransformer_backtesting.ipynb` calls `study.optimize(..., n_jobs=8)`, and
Optuna's TPE sampler is not deterministic under parallelism even with a fixed
seed. Re-running the identical code gives different strategy parameters and
therefore different backtest figures. Measured: the pure-momentum benchmark —
which involves no model at all and whose inputs did not change — moved from 85
entries / \$11,153 to 201 entries / \$9,964 between two consecutive runs.
Backtest point estimates carry search noise on top of seed noise; differences
smaller than a few thousand dollars are not interpretable. Set `n_jobs=1` for a
reproducible (and much slower) search.

## Requirements

Model weights and the large price CSVs are not in git; run
`python ../scripts/download_assets.py` from the repo root first. A CUDA GPU is
assumed for notebook 01.
