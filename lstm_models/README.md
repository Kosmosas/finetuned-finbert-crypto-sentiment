# LSTM forecasting and backtesting

The LSTM counterpart of the iTransformer study. Same data, same evaluation
protocol, same trading strategies, same statistics and figures — only the
forecaster changes.

Nothing here imports from the iTransformer notebooks. The trading strategies are
imported from `../backtesting/backtest_functions.py`, so both model families are
evaluated by literally the same strategy code.

## Notebooks

Run them in order.

| Notebook | What it does | Runtime |
|---|---|---|
| `01_lstm_training.ipynb` | Builds the four feature sets, runs an Optuna study per set, trains 3 seeds per set, and reports forecasting statistics, naive benchmarks and Diebold–Mariano tests | ~1 h on a 12 GB GPU |
| `02_lstm_backtesting.ipynb` | Tunes the three trading strategies by 10-fold CV + Optuna on the training period, then backtests both held-out regimes | ~40 min, CPU-bound |
| `03_lstm_diagrams.ipynb` | Portfolio-trajectory figures per strategy, the backtesting metrics table, and the LaTeX drop-ins | ~1 min |

`lstm_common.py` holds everything the three notebooks share: data loading, the
model, the training loop, the metric and test definitions, and the CV + Optuna
strategy search.

## Protocol

Identical to the iTransformer protocol reported in the manuscript.

| | |
|---|---|
| Feature sets | `price+volume` (4 variates), `+sentiment` (5), `+technicals` (10), `full` (11) |
| Lookback | 168 hours; target = next-hour change of the BTCUSDT close, in USD |
| Date window | one window common to all four sets, so row-wise deletion cannot shift the split |
| Split | training ends **2024-04-30**; May 2024 is a buffer month |
| Test regimes | uptrend Jun 2024 – Jan 2025; sideways Jan 2025 – end of sentiment coverage |
| Scaling | inputs and target standardised on training-period statistics |
| Epoch selection | on a 120-day inner-validation slice at the end of the training period |
| Seeds | 3 per feature set; tables report mean ± sd |
| Benchmarks | zero-change, training historical mean, random walk, AR(1), AR(24) |
| Tests | Diebold–Mariano on absolute errors, Newey–West variance, HLN correction, BH across the family |
| Backtesting | 10-fold chronological CV + Optuna on the training period only; $10,000 initial capital; 0.0275 % per-trade cost |

The backtest signal is the seed-averaged forecast of the three LSTMs trained per
feature set — the same object the Diebold–Mariano tests use.

### The forecasting and backtesting stages sit on different windows

Notebook 01 produces the forecasting table; notebooks 02–03 produce the
backtesting table. The two do not sit on the same windows or the same sentiment
file, and the iTransformer run makes the same split, so the two families stay
comparable table by table.

| | Forecasting (nb 01) | Backtesting (nb 02–03) |
|---|---|---|
| Sentiment file | `_full` column | `_ps2` column |
| Uptrend | 2024-06-01 → 2025-01-01 (n = 5,136 / 5,120) | 2024-06-08 → 2024-11-30 (n = 4,224 / 4,208) |
| Sideways | 2025-01-01 → 2025-03-31 (n = 2,150 / 2,147) | 2025-01-08 → 2025-06-30 (n = 4,176 / 4,171) |
| Train cut-off | 2024-04-30 (+ May buffer) | strategy search on data before 2024-06-01 |

Three things follow, and they are worth knowing before comparing any two rows.

**The train/backtest sentiment split is shared with the iTransformer run.**
Both families train on the `_full` sentiment column and backtest on the `_ps2`
one ([data_loader.py:27](../backtesting/data_loader.py#L27)). Both files are
FinBERT scorings of the same headline stream and correlate at 0.998 on their
21,019 shared timestamps; the second carries ~3,000 more headlines and runs to
Nov 2025 instead of Mar 2025, which is the only reason a Jan–Jul 2025 sideways
window exists at all.

**The backtest periods lose their first 168 hours.** `create_sequences()` runs
on each validation period in isolation, so the first week of each period becomes
lookback and the first tradable hour is `start + 168h`. `period_windows()` in
`lstm_common.py` implements that, which is why the evaluation index matches the
iTransformer trade logs exactly.

**Forecasting `n` and backtesting `n` differ for the same regime name.** That
follows from the two stages using different windows, as the table above shows.

## Model

A unidirectional stacked LSTM encoder with a linear head on the final hidden
state. Capacity (hidden size, depth), dropout and the learning rate are chosen by
an Optuna study per feature set on the same budget the iTransformer study used —
50 trials, each trained for 5 epochs on a reduced corpus of the first 10,000
training sequences (5,000 for the price-and-volume set) — scored by
inner-validation MAE in USD, with median pruning. Batch size 256, AdamW, Huber
loss on the standardised target — no scheduler, no gradient clipping, matching
the iTransformer setup.

Two deliberate departures from the iTransformer search: trials are scored on the
inner-validation slice rather than on the final-year window, so selection never
touches the test regimes; and the encoder is kept
unidirectional, since a bidirectional pass over the lookback window costs one to
two orders of magnitude more compute on a task where every configuration
converges to the same error level.

## Outputs

Everything lands in `out/`, the trained models in `checkpoints/`, the trade logs
in `backtest_results/<feature_set>/` under the same file names the iTransformer
runs used.

**Figures**

| File | Contents |
|---|---|
| `epochs_lstm.png` | inner-validation MAE per epoch, one panel per feature set |
| `lstm_output1.png` | portfolio value, threshold strategy, both regimes |
| `lstm_output2.png` | portfolio value, volatility-adjusted strategy, both regimes |
| `lstm_output3.png` | portfolio value, momentum strategy, both regimes |

**LaTeX tables**

| File | Contents |
|---|---|
| `table_lstm_forecasting.tex` | forecasting performance by regime and feature set |
| `table_lstm_dm.tex` | Diebold–Mariano tests |
| `table_lstm_hparams.tex` | selected model hyperparameters |
| `table_lstm_strategy_params.tex` | selected strategy parameters |
| `table_lstm_backtesting.tex` | backtesting results |

**Data**

`forecasting_performance.csv`, `baselines.csv`, `dm_tests.csv`,
`epoch_selection.csv`, `data_split.csv`, `lstm_hparams.json`,
`lstm_predictions.npz`, `lstm_strategy_params.json`,
`lstm_strategy_cv_folds.csv`, `lstm_backtesting_metrics.csv`,
`lstm_buy_and_hold.csv`, and a combined `lstm_forecasting_tables.xlsx`.

## Results as run

Tuning 4.5 min, training 2.7 min, strategy search 24 min, figures under a minute.

**Forecasting.** Every feature set lands at the zero-change benchmark or a little
below it. In the uptrend the LSTM is 0.4–0.6 % *better* than a constant zero
forecast, and the Diebold–Mariano tests confirm it (negative statistics,
q_BH 0.0005–0.0024); in the sideways regime it matches the benchmark and no
comparison is significant. Uptrend correlations (+0.048 to +0.057) sit above the
1/√n ≈ 0.014 noise floor; sideways correlations (+0.022 to +0.029) sit at it.

| Regime | Feature set | MAE (USD) | vs zero-change | Corr. | Dir. Acc. |
|---|---|---|---|---|---|
| Uptrend | Price + volume | 255.3 ± 0.2 | 0.994× | +0.0569 | 52.69 % |
| Uptrend | + sentiment | 255.4 ± 0.2 | 0.995× | +0.0571 | 52.33 % |
| Uptrend | + technicals | 255.7 ± 0.3 | 0.996× | +0.0499 | 53.00 % |
| Uptrend | Full | 255.8 ± 0.2 | 0.996× | +0.0477 | 52.81 % |
| Sideways | Price + volume | 358.1 ± 0.5 | 1.000× | +0.0266 | 51.58 % |
| Sideways | + sentiment | 358.3 ± 0.4 | 1.000× | +0.0290 | 51.44 % |
| Sideways | + technicals | 357.9 ± 0.6 | 0.999× | +0.0224 | 51.56 % |
| Sideways | Full | 357.8 ± 0.5 | 0.999× | +0.0234 | 51.50 % |

Feature content makes no difference. The sentiment-vs-no-sentiment comparisons
are insignificant in both regimes (q_BH 0.33–0.59), and the spread between
feature sets (about 0.5 USD) is roughly twice the seed-to-seed spread (0.2 USD).
This matches the iTransformer conclusion; what differs is that the LSTM reaches
the naive baseline rather than sitting about 10 % above it.

**Backtesting**, on the shared evaluation windows (buy-and-hold $13,891
uptrend, $11,057 sideways).

| Strategy | Uptrend best | Beat B&H | Sideways best | Beat B&H |
|---|---|---|---|---|
| Threshold | $15,063 (+ sentiment) | 1/4 | $10,000 (no trades) | 0/4 |
| Volatility-adjusted | $11,925 (+ sentiment) | 0/4 | $12,992 (price + volume) | 3/4 |
| Momentum | $16,615 (full) | 4/4 | $8,010 (+ technicals) | 0/4 |

Momentum beats buy-and-hold on all four feature sets in the uptrend and loses on
all four in the sideways period; the volatility-adjusted strategy does the
reverse. That regime dependence, not the feature set, is what moves these
numbers — consistent with the forecasting result that no feature set carries
usable signal.

One artefact of the LSTM's conservatism is worth knowing before reading the
table. Its predicted next-hour moves have a standard deviation near 24–41 USD,
about 0.04 % of the price, while CV-selected entry thresholds can land well above
that. The threshold strategy therefore fires only a handful of times on most
feature sets, and in four of the twenty-four runs never at all. Those runs are
reported as zeros in italics and excluded from the best-value comparison in the
LaTeX table, since a 0.00 % drawdown earned by never opening a position is
inactivity rather than risk control. Momentum is unaffected: its signal compares
the predicted price against the price `lookback` hours earlier, so it is driven
mostly by the realised move and trades 153–255 times per period.

## Requirements

```
torch  optuna  pandas  numpy  scipy  scikit-learn  matplotlib
```

The raw inputs are `../backtesting/data/BTCUSDT-last.csv` and
`../backtesting/data/cleaned/`. The 1-minute price file is not in git; run
`python ../scripts/download_assets.py` from the repo root if it is missing. The
first notebook caches the four hourly feature sets to `cache/`, so the 400 MB
price file is parsed once.
