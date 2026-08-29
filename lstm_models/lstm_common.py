"""Shared code for the LSTM forecasting / backtesting study.

This mirrors the iTransformer pipeline one-for-one:

  * the same four feature sets built from the same raw files,
  * the same evaluation protocol (one common date window, a fixed
    chronological split with a buffer month, standardised inputs and target,
    epoch selection on an inner-validation slice, three seeds),
  * the same forecasting metrics, naive benchmarks and Diebold--Mariano tests,
  * the same three trading strategies, tuned by k-fold CV + Optuna on the
    training period only.

Only the forecaster changes: `iTransformer` becomes a stacked LSTM.

Nothing here imports from the iTransformer notebooks. The trading strategies
themselves are imported from `backtesting/backtest_functions.py` so that both
model families are evaluated by literally the same code.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "backtesting" / "data"
CLEANED = DATA / "cleaned"
PRICE = DATA / "BTCUSDT-last.csv"          # 1-min OHLCV
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CKPT_DIR = HERE / "checkpoints"
CACHE = HERE / "cache"
BT_DIR = HERE / "backtest_results"
for _d in (OUT, CKPT_DIR, CACHE, BT_DIR):
    _d.mkdir(exist_ok=True)

# the strategy implementations are shared with the iTransformer study
sys.path.insert(0, str(REPO / "backtesting"))

# --------------------------------------------------------------------------- #
# protocol constants  (identical to the iTransformer protocol)
# --------------------------------------------------------------------------- #

SEQ = 24 * 7                                        # 168-hour lookback
TARGET = "price_change_1h"

TRAIN_END = pd.Timestamp("2024-04-30", tz="UTC")    # training ends here
INNER_VAL_DAYS = 120                                # epoch-selection slice
INNER_VAL_START = TRAIN_END - pd.Timedelta(days=INNER_VAL_DAYS)
UPTREND = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
SIDEWAYS = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-06-01", tz="UTC"))
#                       ^ May 2024 is the buffer month between train and test

# ---------------------------------------------------------------- backtesting
# The backtesting stage follows the iTransformer backtesting notebooks rather
# than the forecasting protocol above, so that the two studies' backtesting
# tables land on the same evaluation windows and can be compared row by row.
# See create_validation_periods() in ../backtesting/data_loader.py.
BT_PERIODS = {
    1: dict(label="Uptrend Market", short="Uptrend",
            start=pd.Timestamp("2024-06-01", tz="UTC"),
            end=pd.Timestamp("2024-12-01", tz="UTC")),
    2: dict(label="Sideways Market", short="Sideways",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-07-01", tz="UTC")),
}
BT_TRAIN_END = pd.Timestamp("2024-06-01", tz="UTC")   # strategy search window

# Sentiment source.
#
# Scoring headlines with a single FinBERT checkpoint fine-tuned on
# 2019-01..2024-03 and then applying it to those same headlines makes the
# forecaster's training-period sentiment feature a set of in-sample fitted values
# (corr 0.477 in-sample vs 0.071 out-of-sample; a 4.7x train/test
# standard-deviation break). The feature is therefore rebuilt out of fold.
#
# Both files below come from the expanding-window out-of-fold scoring in
# `sentiment_analysis/oof_finbert_scoring.ipynb` +
# `sentiment_analysis/oof_score_full_stream.ipynb`: every headline is scored by a
# model trained only on strictly earlier data, with a one-day purge around each
# split. The two files are the same column; the `_full` one carries title/source,
# the `_ps2` one is the two-column shape `backtesting/data_loader.py` expects.
#
# Set SENT_USE_OOF = False to fall back on the single-checkpoint (in-sample)
# column instead.
SENT_USE_OOF = True

if SENT_USE_OOF:
    SENT_FILE_TRAIN = "predicted_sentiments_oof_tone3_cond_full.csv"
    SENT_FILE_BACKTEST = "predicted_sentiments_oof_tone3_cond_ps2.csv"
else:
    SENT_FILE_TRAIN = "predicted_sentiments_finbert.csv"   # single-checkpoint scoring
    SENT_FILE_BACKTEST = "predicted_sentiments2.csv"       # backtesting/data_loader.py

# Out-of-fold scoring cannot produce a value before the first trainable window
# closes, so headlines before 2020-01-01 carry no score. `build_feature_sets`
# zero-fills hours with no headlines, which would silently turn "no model
# available" into "neutral sentiment". Every feature set is therefore restricted
# to start here, so the four sets stay row-comparable.
ANALYSIS_START = pd.Timestamp("2020-01-01", tz="UTC") if SENT_USE_OOF else None

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST_PCT = 0.0275

FEATURE_SET_ORDER = [
    "price+volume",
    "price+volume+sentiment",
    "price+volume+technicals",
    "full",
]

NICE_NAME = {
    "price+volume": "Price + volume",
    "price+volume+sentiment": "Price + volume + headline score",
    "price+volume+technicals": "Price + volume + technicals",
    "full": "Full dataset",
}

# folder names used by the iTransformer backtest results, kept identical so the
# two studies can be diffed folder-by-folder
FOLDER_NAME = {
    "price+volume": "price_only",
    "price+volume+sentiment": "sentiment_only",
    "price+volume+technicals": "technicals_only",
    "full": "full_dataset",
}

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

RENAME = {"1502942400": "timestamp", "4261.48": "open", "4261.48.1": "high",
          "4261.48.2": "low", "4261.48.3": "close", "1.775183": "btc_volume_sell",
          "320.390851": "usd_volume_buy", "0.075183": "btc_volume_buy",
          "7564.906851": "usd_volume_sell", "3": "activity"}

AGG_BASE = {"price_change_1h": "last", "close": "last",
            "usd_volume_buy": "sum", "usd_volume_sell": "sum"}
AGG_TECH = {"close_gold": "last", "close_vxx": "last", "close_sp500": "last",
            "tx_count": "sum", "blocks_mined": "sum", "est_hashrate": "mean"}
AGG_SENT = {"predicted_sentiment_24h_avg": "mean"}


def load_price_minutes() -> pd.DataFrame:
    """1-minute BTCUSDT with the derived columns the original notebooks build."""
    df = pd.read_csv(PRICE, sep="|").rename(columns=RENAME)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("datetime")
    df["price_change_1h"] = df["close"].diff(periods=60)      # dollars, 60 x 1-min
    return df[["close", "usd_volume_buy", "usd_volume_sell", "price_change_1h"]]


def _load_unix_series(name: str) -> pd.DataFrame:
    d = pd.read_csv(CLEANED / name)
    d["datetime"] = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    return d.set_index("datetime").drop(columns=["timestamp"])


def _load_hour_series(name: str) -> pd.DataFrame:
    d = pd.read_csv(CLEANED / name)
    d["datetime"] = pd.to_datetime(d["hour"], utc=True)
    d = d.set_index("datetime").drop(columns=["hour"]).sort_index()
    return d.interpolate(method="time", limit_direction="both")


def build_feature_sets(sentiment_file: str = SENT_FILE_TRAIN,
                       use_cache: bool = True,
                       verbose: bool = True) -> dict[str, pd.DataFrame]:
    """The four hourly feature sets, built exactly as in the iTransformer study.

    `sentiment_file` selects which FinBERT scoring feeds the sentiment column:
    SENT_FILE_TRAIN for the forecasting stage, SENT_FILE_BACKTEST for the
    backtesting stage — mirroring the split in the original notebooks.
    """
    # The cache key must include the sentiment file, or switching SENT_USE_OOF
    # silently reuses frames built from the other scoring.
    stage = "" if sentiment_file == SENT_FILE_TRAIN else "_bt"
    tag = f"{stage}__{Path(sentiment_file).stem}"
    cache_files = {n: CACHE / f"hourly_{FOLDER_NAME[n]}{tag}.pkl" for n in FEATURE_SET_ORDER}
    if use_cache and all(p.exists() for p in cache_files.values()):
        hourly = {n: pd.read_pickle(p) for n, p in cache_files.items()}
        if verbose:
            for n, h in hourly.items():
                print(f"{n:26s} variates={h.shape[1]:2d}  rows={len(h):,}  (cache)")
        return hourly

    px = load_price_minutes()
    df_sent = _load_unix_series(sentiment_file)[["predicted_sentiment"]]
    df_gold = _load_unix_series("GOLD_hourly.csv")
    df_sp = _load_unix_series("SP500_hourly.csv")
    df_vxx = _load_unix_series("VXX_hourly.csv")
    df_tx = _load_hour_series("tx_count_10years.csv")
    df_hash = _load_hour_series("hashrate_10years.csv")

    # hourly sentiment features, exactly as the notebooks build them
    hs = df_sent.resample("1h").agg({"predicted_sentiment": ["min", "max", "mean"]}).fillna(0)
    hs.columns = ["_".join(c) for c in hs.columns]
    hs = hs.rename(columns={"predicted_sentiment_mean": "predicted_sentiment_24h_avg"})
    hs["predicted_sentiment_24h_avg"] = hs["predicted_sentiment_24h_avg"].rolling(24).mean()
    hs = hs.fillna(0)

    merged = (px
              .merge(hs, left_index=True, right_index=True, how="left")
              .merge(df_gold, left_index=True, right_index=True, how="left", suffixes=("", "_gold"))
              .merge(df_sp, left_index=True, right_index=True, how="left", suffixes=("", "_sp500"))
              .merge(df_vxx, left_index=True, right_index=True, how="left", suffixes=("", "_vxx"))
              .merge(df_tx, left_index=True, right_index=True, how="left")
              .merge(df_hash, left_index=True, right_index=True, how="left"))
    merged[["close_gold", "close_sp500", "close_vxx"]] = \
        merged[["close_gold", "close_sp500", "close_vxx"]].ffill()

    specs = {
        "price+volume":            (px,     dict(AGG_BASE)),
        "price+volume+sentiment":  (merged, {**AGG_BASE, **AGG_SENT}),
        "price+volume+technicals": (merged, {**AGG_BASE, **AGG_TECH}),
        "full":                    (merged, {**AGG_BASE, **AGG_TECH, **AGG_SENT}),
    }
    hourly = {n: src.resample("1h").agg(agg).dropna() for n, (src, agg) in specs.items()}

    for n, h in hourly.items():
        h.to_pickle(cache_files[n])
        if verbose:
            print(f"{n:26s} variates={h.shape[1]:2d}  rows={len(h):,}")
    return hourly


def common_window(hourly: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = max(h.index.min() for h in hourly.values())
    if ANALYSIS_START is not None:
        lo = max(lo, ANALYSIS_START)      # no zero-filled "sentiment unavailable" hours
    return lo, min(h.index.max() for h in hourly.values())


def restrict_to_common(hourly: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    lo, hi = common_window(hourly)
    return {n: h.loc[(h.index >= lo) & (h.index <= hi)] for n, h in hourly.items()}


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #

def make_windows(dfh: pd.DataFrame, target: str = TARGET, seq: int = SEQ):
    """Sliding windows, matching create_sequences() in the original notebooks.

    Returns X (n, seq, variates), y (n,), the timestamp of each target, the
    close price one hour before the target, and the close price at the target.
    """
    vals = np.ascontiguousarray(dfh.to_numpy(dtype=np.float32))
    n = len(dfh) - seq
    X = np.lib.stride_tricks.sliding_window_view(vals, (seq, vals.shape[1]))[:n, 0]
    y = dfh[target].to_numpy(dtype=np.float32)[seq:seq + n]
    idx = dfh.index[seq:seq + n]
    close = dfh["close"].to_numpy(dtype=np.float32)[seq:seq + n]
    close_prev = dfh["close"].to_numpy(dtype=np.float32)[seq - 1:seq - 1 + n]
    return X, y, idx, close_prev, close


def period_windows(dfh: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                   target: str = TARGET, seq: int = SEQ):
    """Windows built *inside* one evaluation period, as the iTransformer
    backtesting notebooks do.

    Those notebooks call create_sequences() on `val_df_periodN` alone, so the
    first `seq` hours of each period are consumed as lookback and the first
    tradable hour is `start + seq`. Reproducing that here keeps the LSTM and
    iTransformer backtests on an identical evaluation index.
    """
    sub = dfh.loc[(dfh.index >= start) & (dfh.index < end)]
    return make_windows(sub, target=target, seq=seq)


def regime_mask(idx: pd.DatetimeIndex, regime: str, sideways_end: pd.Timestamp | None = None):
    lo, hi = UPTREND if regime == "Uptrend" else SIDEWAYS
    if regime == "Sideways" and sideways_end is not None:
        hi = min(hi, sideways_end)
    return (idx >= lo) & (idx < hi)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMForecaster(nn.Module):
    """Stacked LSTM encoder + linear head on the final hidden state.

    Input  : (batch, lookback, variates)
    Output : (batch,) — the standardised next-hour change of the target column.
    """

    def __init__(self, num_variates: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_variates,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]                 # final time step of the sequence
        return self.head(self.dropout(last)).squeeze(-1)


DEFAULT_HP = dict(hidden_size=128, num_layers=2, dropout=0.1,
                  bidirectional=False, lr=1e-3, batch_size=256)


def build_lstm(num_variates: int, hp: dict, seed: int = 0) -> LSTMForecaster:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return LSTMForecaster(
        num_variates=num_variates,
        hidden_size=int(hp.get("hidden_size", 128)),
        num_layers=int(hp.get("num_layers", 2)),
        dropout=float(hp.get("dropout", 0.1)),
        bidirectional=bool(hp.get("bidirectional", False)),
    ).to(DEVICE)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def _standardise(Xtr, Xev, ytr):
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0)
    sd[sd < 1e-12] = 1.0
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    if ysd < 1e-12:
        ysd = 1.0
    return ((Xtr - mu) / sd, (Xev - mu) / sd, (ytr - ymu) / ysd, mu, sd, ymu, ysd)


EVAL_BATCH = 512          # LSTM activations over a 168-step sequence are large


def _predict_batched(model, X_t, batch=EVAL_BATCH):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            outs.append(model(X_t[i:i + batch].to(DEVICE)).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def train_eval(dfh: pd.DataFrame,
               train_mask: Callable,
               eval_mask: Callable,
               hp: dict,
               epochs: int = 60,
               seed: int = 0,
               track: bool = False,
               patience: int | None = None,
               target: str = TARGET,
               return_model: bool = False,
               trial=None,
               max_train: int | None = None,
               verbose: bool = False) -> dict:
    """One LSTM training run.

    Inputs and target are standardised with training-period statistics; the
    Huber loss is applied on the standardised target; predictions are returned
    in USD. `patience` enables epoch selection on the eval window (which, during
    model selection, is the inner-validation slice — never the test regimes).

    `max_train` keeps only the first N training sequences, which is how the
    iTransformer hyperparameter search was accelerated in the original study.
    """
    X, y, idx, close_prev, close = make_windows(dfh, target=target)
    tr, ev = train_mask(idx), eval_mask(idx)
    Xtr, ytr = X[tr], y[tr]
    Xev, yev = X[ev], y[ev]
    if max_train is not None:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]

    Xtr, Xev, ytr_s, mu, sd, ymu, ysd = _standardise(Xtr, Xev, ytr)

    model = build_lstm(Xtr.shape[-1], hp, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 1e-3)))
    loss_fn = nn.HuberLoss()

    # keep the tensors resident on the GPU when they fit; the windows are the
    # dominant cost and host->device copies otherwise dominate the epoch time
    def _to_device(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        nbytes = t.element_size() * t.nelement()
        if DEVICE == "cuda" and nbytes < 3.5e9:
            return t.to(DEVICE)
        return t

    Xtr_t = _to_device(Xtr)
    ytr_t = _to_device(ytr_s)
    Xev_t = _to_device(Xev)

    batch_size = int(hp.get("batch_size", 1024))
    n = len(Xtr_t)
    g = torch.Generator().manual_seed(seed)

    history, best = [], (np.inf, None, -1, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(Xtr_t.device)
        for i in range(0, n, batch_size):
            sel = perm[i:i + batch_size]
            xb = Xtr_t[sel].to(DEVICE, non_blocking=True)
            yb = ytr_t[sel].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        if track or patience is not None:
            model.eval()
            pv = _predict_batched(model, Xev_t) * ysd + ymu
            mae = float(np.abs(pv - yev).mean())
            history.append(mae)
            if verbose:
                print(f"    epoch {ep + 1:3d}  eval MAE {mae:9.2f}")
            if mae < best[0]:
                best = (mae, pv.copy(),
                        ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if trial is not None:
                trial.report(mae, ep)
                import optuna
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if patience is not None and ep - best[2] >= patience:
                break

    model.eval()
    pred = _predict_batched(model, Xev_t) * ysd + ymu

    result = dict(pred=pred, y=yev, idx=idx[ev], close_prev=close_prev[ev],
                  close=close[ev], history=history, best_epoch=best[2],
                  best_mae=best[0], best_pred=best[1] if best[1] is not None else pred,
                  scaler=dict(mu=mu, sd=sd, ymu=ymu, ysd=ysd), hp=dict(hp), seed=seed,
                  columns=list(dfh.columns))
    if return_model:
        result["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        result["best_state_dict"] = best[3]

    del model, Xtr_t, ytr_t, Xev_t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


class LSTMPredictor:
    """A trained LSTM (or a seed ensemble of them) plus its input scaler."""

    def __init__(self, state_dicts: list[dict], scaler: dict, hp: dict,
                 columns: list[str], num_variates: int):
        self.models = []
        for sd_ in state_dicts:
            m = build_lstm(num_variates, hp, seed=0)
            m.load_state_dict(sd_)
            m.eval()
            self.models.append(m)
        self.scaler = scaler
        self.hp = dict(hp)
        self.columns = list(columns)
        self.num_variates = num_variates

    def predict_windows(self, X: np.ndarray) -> np.ndarray:
        """X: (n, seq, variates) raw. Returns next-hour changes in USD."""
        mu, sd = self.scaler["mu"], self.scaler["sd"]
        Xs = ((X - mu) / sd).astype(np.float32)
        Xt = torch.from_numpy(np.ascontiguousarray(Xs))
        preds = [_predict_batched(m, Xt) for m in self.models]
        return np.mean(preds, axis=0) * self.scaler["ysd"] + self.scaler["ymu"]

    def predict_frame(self, dfh: pd.DataFrame, mask: Callable | None = None):
        X, y, idx, close_prev, close = make_windows(dfh)
        if mask is not None:
            m = mask(idx)
            X, y, idx, close_prev, close = X[m], y[m], idx[m], close_prev[m], close[m]
        return dict(pred=self.predict_windows(X), y=y, idx=idx,
                    close_prev=close_prev, close=close)

    def save(self, path: Path):
        torch.save(dict(state_dicts=[{k: v.cpu() for k, v in s.state_dict().items()}
                                     for s in self.models],
                        scaler=self.scaler, hp=self.hp, columns=self.columns,
                        num_variates=self.num_variates), path)

    @staticmethod
    def load(path: Path) -> "LSTMPredictor":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return LSTMPredictor(blob["state_dicts"], blob["scaler"], blob["hp"],
                             blob["columns"], blob["num_variates"])


# --------------------------------------------------------------------------- #
# forecasting metrics  (identical definitions to the iTransformer study)
# --------------------------------------------------------------------------- #

def metrics(pred, y, close_prev, label, regime, extra=None) -> dict:
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    err = pred - y
    pct_pred = 100 * pred / close_prev
    pct_y = 100 * y / close_prev
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(pred, y)[0, 1] if pred.std() > 0 else np.nan
    nz = (pred != 0) & (y != 0)
    d = dict(regime=regime, model=label, n=len(y),
             MAE_usd=float(np.abs(err).mean()),
             RMSE_usd=float(np.sqrt((err ** 2).mean())),
             MAE_pct=float(np.abs(pct_pred - pct_y).mean()),
             corr=float(c),
             dir_acc=float(100 * (np.sign(pred[nz]) == np.sign(y[nz])).mean()) if nz.any() else np.nan,
             mean_pred=float(pred.mean()), std_pred=float(pred.std()))
    if extra:
        d.update(extra)
    return d


def ar_forecast(y_full, idx_full, mask_train, mask_eval, p):
    """AR(p) on the target, fitted by OLS on the training window only."""
    Y = np.asarray(y_full, float)
    lags = np.column_stack([np.roll(Y, k) for k in range(1, p + 1)])
    valid = np.zeros(len(Y), bool)
    valid[p:] = True
    Xd = np.column_stack([np.ones(len(Y)), lags])
    tr = mask_train(idx_full) & valid
    beta = np.linalg.lstsq(Xd[tr], Y[tr], rcond=None)[0]
    ev = mask_eval(idx_full) & valid
    return Xd[ev] @ beta, Y[ev], ev


# --------------------------------------------------------------------------- #
# statistical tests  (same implementations used for the iTransformer tables)
# --------------------------------------------------------------------------- #

from scipy.stats import t as tdist                              # noqa: E402


def dm_test(e1, e2, loss="abs", h=1, lags=None) -> dict:
    """Diebold--Mariano. Negative statistic => model 1 has the lower loss."""
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    d = (np.abs(e1) - np.abs(e2)) if loss == "abs" else (e1 ** 2 - e2 ** 2)
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    if lags is None:
        lags = int(round(n ** (1 / 3)))
    dc = d - dbar
    var = (dc ** 2).mean()
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1.0)) * (dc[l:] * dc[:-l]).mean()
    var = max(var, 1e-300)
    dm = dbar / math.sqrt(var / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)   # Harvey-Leybourne-Newbold
    dm_adj = dm * hln
    return dict(dbar=dbar, dm=dm_adj, p=2 * tdist.sf(abs(dm_adj), n - 1), n=n, lags=lags)


def bh_qvalues(pvals):
    """Benjamini--Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        running = min(running, p[i] * n / k)
        q[i] = running
    return q


# --------------------------------------------------------------------------- #
# backtest metrics
# --------------------------------------------------------------------------- #

def calculate_sharpe_ratio(portfolio_values, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe from an hourly portfolio-value path."""
    values = np.asarray(portfolio_values, float)
    if len(values) < 2:
        return 0.0
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    sd = np.std(returns)
    if sd == 0:
        return 0.0
    return float((np.mean(returns) - risk_free_rate) / sd * np.sqrt(8760))


def calculate_max_drawdown(values) -> float:
    s = pd.Series(np.asarray(values, float))
    return float(((s / s.cummax()) - 1).min())


def simple_sharpe_ratio(returns) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=0))


def create_trades_dataframe(trade_times, trade_values, trade_types,
                            plot_index, strategy: str = "Backtest") -> pd.DataFrame:
    rows = []
    for i, t in enumerate(trade_times):
        if t < len(plot_index):
            rows.append(dict(datetime=plot_index[t], trade_index=t,
                             portfolio_value=trade_values[i],
                             trade_type=trade_types[i], strategy=strategy))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# strategy hyperparameter search:  k-fold CV + Optuna on the training period
# --------------------------------------------------------------------------- #

def _fold_predictions(dfh: pd.DataFrame, predictor: "LSTMPredictor",
                      n_splits: int = 10, forecast: str = "model", seed: int = 0):
    """Predictions over the training period, split into k chronological folds.

    Sequences are built over the training frame, KFold(shuffle=False) yields
    chronological folds, and each fold is scored with the model's predictions on
    that fold.

    `forecast` selects what drives the strategy signal, which is how the
    forecaster's contribution is separated from the trading rule's:

    * ``"model"``    - the model's predicted next-hour change (the normal case)
    * ``"zero"``     - a constant zero forecast. For the momentum rule this
      reduces the signal to pure realised time-series momentum, since
      ``predict_func`` computes ``(price[i-1] + pred[i] - price[i-lookback]) /
      price[i-lookback]``; with ``pred = 0`` no model information enters at all.
    * ``"shuffled"`` - the model's predictions permuted in time. Keeps the
      forecast distribution and destroys only its timing, so it isolates whether
      the *timing* of the forecast carries anything.
    """
    from sklearn.model_selection import KFold

    X, y, idx, close_prev, close = make_windows(dfh)
    if forecast == "zero":
        pred = np.zeros(len(X), dtype=np.float32)
    elif forecast == "shuffled":
        pred = predictor.predict_windows(X)
        pred = np.random.default_rng(seed).permutation(pred)
    elif forecast == "model":
        pred = predictor.predict_windows(X)
    else:
        raise ValueError(f"unknown forecast mode {forecast!r}")

    folds = []
    for _, val_idx in KFold(n_splits=n_splits, shuffle=False).split(X):
        folds.append(dict(pred=pred[val_idx], close=close[val_idx], idx=idx[val_idx]))
    return folds


def _cv_search(folds, space_fn, run_fn, n_trials, timeout, n_jobs, seed,
               agg="mean", verbose=True, fmt=None):
    """One Optuna study scored by the aggregate Sharpe across all folds.

    An alternative would be a separate study per fold, averaging the per-fold
    winners. Averaging argmaxes is not an optimisation of anything -- it is
    undefined for integer and categorical parameters, and the averaged
    configuration need not be good on any fold. Here a single configuration is
    scored by its mean (or median) Sharpe across the chronological folds, so the
    selected point is the one that actually performs best overall.

    Returns the chosen parameters and the per-fold Sharpe *at that configuration*,
    which is what makes fold-to-fold stability readable.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _score(f, p):
        try:
            pv = run_fn(f, p)
            s = calculate_sharpe_ratio(pv)
        except Exception:
            return 0.0
        return 0.0 if not np.isfinite(s) else float(s)

    def objective(trial):
        p = space_fn(trial)                     # suggest once, score every fold with it
        scores = [_score(f, p) for f in folds]
        return float(np.mean(scores)) if agg == "mean" else float(np.median(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)

    params = dict(study.best_params)
    rows = [dict(fold=k, sharpe=_score(f, params), **params)
            for k, f in enumerate(folds, 1)]
    fold_df = pd.DataFrame(rows)
    if verbose:
        desc = fmt(params) if fmt else str(params)
        print(f"  selected by {agg} fold Sharpe: {desc}")
        print(f"  aggregate Sharpe {study.best_value:.3f}  |  per-fold "
              f"{fold_df['sharpe'].min():.3f}..{fold_df['sharpe'].max():.3f} "
              f"(sd {fold_df['sharpe'].std():.3f})")
    return params, fold_df


def cv_optimize_threshold(dfh, predictor, n_splits=10, n_trials=200, timeout=600,
                          initial_capital=INITIAL_CAPITAL,
                          transaction_cost_pct=TRANSACTION_COST_PCT,
                          n_jobs=1, seed=0, verbose=True, forecast="model", agg="mean"):
    """Optimise (threshold, hold) for the simple threshold strategy."""
    from backtest_functions import backtest_profit_threshold_with_trades

    folds = _fold_predictions(dfh, predictor, n_splits, forecast=forecast, seed=seed)

    def space(trial):
        return dict(threshold=trial.suggest_float("threshold", 0.01, 2.0),
                    hold_hours=trial.suggest_int("hold_hours", 1, 24 * 7))

    def run(f, p):
        pv, *_ = backtest_profit_threshold_with_trades(
            f["close"], f["pred"], initial_capital, p["threshold"],
            transaction_cost_pct, p["hold_hours"])
        return pv

    return _cv_search(folds, space, run, n_trials, timeout, n_jobs, seed, agg, verbose,
                      fmt=lambda p: f"threshold={p['threshold']:.3f}%  hold={p['hold_hours']}h")


def cv_optimize_risk(dfh, predictor, n_splits=10, n_trials=500, timeout=900,
                     initial_capital=INITIAL_CAPITAL,
                     transaction_cost_pct=TRANSACTION_COST_PCT,
                     n_jobs=1, seed=0, verbose=True, forecast="model", agg="mean"):
    """Optimise (threshold, target risk, hold) for the volatility-adjusted strategy."""
    from backtest_functions import backtest_profit_threshold

    folds = _fold_predictions(dfh, predictor, n_splits, forecast=forecast, seed=seed)

    def space(trial):
        return dict(threshold=trial.suggest_float("threshold", 0.01, 2.0),
                    target_risk=trial.suggest_float("target_risk", 0.01, 2.5),
                    hold_hours=trial.suggest_int("hold_hours", 1, 24 * 7))

    def run(f, p):
        pv, *_ = backtest_profit_threshold(
            f["close"], f["pred"], initial_capital, p["threshold"],
            target_risk_per_trade=p["target_risk"],
            transaction_cost_pct=transaction_cost_pct, hold_hours=p["hold_hours"])
        return pv

    return _cv_search(folds, space, run, n_trials, timeout, n_jobs, seed, agg, verbose,
                      fmt=lambda p: (f"threshold={p['threshold']:.3f}%  "
                                     f"risk={p['target_risk']:.3f}  hold={p['hold_hours']}h"))


def cv_optimize_momentum(dfh, predictor, n_splits=10, n_trials=1000, timeout=900,
                         initial_capital=INITIAL_CAPITAL,
                         transaction_cost_pct=TRANSACTION_COST_PCT,
                         n_jobs=1, seed=0, verbose=True, forecast="model", agg="mean"):
    """Optimise (lookback, hold, signal threshold, target risk) for momentum.

    With ``forecast="zero"`` this is a pure time-series momentum benchmark: the
    trading rule is tuned and evaluated identically, but no model output enters
    the signal.
    """
    from backtest_functions import backtest_ts_momentum

    folds = _fold_predictions(dfh, predictor, n_splits, forecast=forecast, seed=seed)

    def space(trial):
        return dict(lookback=trial.suggest_int("lookback", 1, 24 * 7),
                    hold_hours=trial.suggest_int("hold_hours", 1, 24 * 7),
                    signal_thresh=trial.suggest_float("signal_thresh", 0.01, 2.0),
                    target_risk=trial.suggest_float("target_risk", 0.0, 2.5))

    def run(f, p):
        _, pv, *_ = backtest_ts_momentum(
            prices=f["close"], predicted_changes=f["pred"],
            lookback=p["lookback"], hold_hours=p["hold_hours"],
            target_risk_per_trade=p["target_risk"], threshold=p["signal_thresh"],
            initial_capital=initial_capital, transaction_cost_pct=transaction_cost_pct)
        return pv

    return _cv_search(folds, space, run, n_trials, timeout, n_jobs, seed, agg, verbose,
                      fmt=lambda p: (f"lookback={p['lookback']}  hold={p['hold_hours']}h  "
                                     f"signal={p['signal_thresh']:.3f}%  "
                                     f"risk={p['target_risk']:.3f}"))


# --------------------------------------------------------------------------- #
# portfolio-trajectory figures
# --------------------------------------------------------------------------- #
#
# Shared style for the portfolio-trajectory figures: the seaborn-v0_8 style, the
# 15x6 two-panel layout, the per-feature-set palette, the grey buy-and-hold line
# rebased to the initial capital, the blue y-axis, the upper-left legend with
# buy-and-hold moved to the front, and the reference line at the initial capital.
# Kept here rather than in each diagrams notebook so the LSTM and iTransformer
# figures cannot drift apart.

import matplotlib.pyplot as plt                                   # noqa: E402

BT_STYLE = "seaborn-v0_8"
BT_FIGSIZE = (15, 6)

# keyed by FOLDER_NAME values
BT_PALETTE = {
    "price_only": "#FF6B6B",        # coral red
    "sentiment_only": "#FFA726",    # warm orange
    "technicals_only": "#B565FF",   # light purple
    "full_dataset": "#00E096",      # bright teal
}

BT_STRATEGIES = {
    "threshold": "Threshold Strategy",
    "risk_based": "Risk-Based Strategy",
    "momentum": "Momentum Strategy",
}
BT_FILE_PREFIX = {"threshold": "threshold_backtest_trades",
                  "risk_based": "risk_based_backtest_trades",
                  "momentum": "momentum_backtest_trades"}
# '-1' is the first validation period, '-2' the second, as written by the
# iTransformer backtesting notebooks
BT_MARKETS = {"uptrend": dict(suffix="-1.csv", label="Uptrend Market", col=0),
              "sideways": dict(suffix="-2.csv", label="Sideways Market", col=1)}


def load_trade_log(bt_dir: Path, folder: str, strategy: str, market: str):
    """One trade log, or None if the run produced no file."""
    path = Path(bt_dir) / folder / f"{BT_FILE_PREFIX[strategy]}{BT_MARKETS[market]['suffix']}"
    if not path.exists():
        print(f"Warning: file not found - {path}")
        return None
    d = pd.read_csv(path)
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True)
    return d


# Model-free momentum controls written by the 02 notebooks, drawn in neutral
# monochrome so they read as benchmarks rather than as another feature set.
BT_BENCHMARKS = {
    "no_model_zero": dict(label="Pure momentum (no forecast)", color="black",
                          linestyle="--", linewidth=2.4),
    "no_model_shuffled": dict(label="Shuffled forecast", color="dimgray",
                              linestyle=":", linewidth=2.2),
}


def plot_portfolio_panels(bt_dir: Path, btc_close: pd.Series, strategy: str,
                          save_to: Path | None = None,
                          initial_capital: float = INITIAL_CAPITAL,
                          label_of=None, dpi: int = 150,
                          benchmarks: dict | None = None):
    """Uptrend (left) vs sideways (right) portfolio trajectories for one strategy.

    `btc_close` is an hourly close series indexed by UTC timestamp; it is rebased
    to `initial_capital` at the first hour of each panel, as the source notebook
    does. `label_of` maps a folder name to a series label and defaults to the
    manuscript's feature-set names.
    """
    if label_of is None:
        rev = {v: k for k, v in FOLDER_NAME.items()}
        label_of = lambda folder: NICE_NAME[rev[folder]]      # noqa: E731

    plt.style.use(BT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=BT_FIGSIZE)

    for market, info in BT_MARKETS.items():
        ax = axes[info["col"]]
        first = True
        for name in FEATURE_SET_ORDER:
            folder = FOLDER_NAME[name]
            tdf = load_trade_log(bt_dir, folder, strategy, market)
            if tdf is None or tdf.empty:
                continue
            ax.plot(tdf["datetime"], tdf["portfolio_value"], label=label_of(folder),
                    color=BT_PALETTE[folder], linewidth=2, alpha=0.8)
            if first:
                sub = btc_close[(btc_close.index >= tdf["datetime"].min()) &
                                (btc_close.index <= tdf["datetime"].max())]
                if len(sub):
                    ax.plot(sub.index, sub * (initial_capital / sub.iloc[0]),
                            color="gray", linewidth=1.5, alpha=0.7, label="BTC Buy & Hold")
                first = False

        for folder, style in (benchmarks or {}).items():
            bdf = load_trade_log(bt_dir, folder, strategy, market)
            if bdf is None or bdf.empty:
                continue
            ax.plot(bdf["datetime"], bdf["portfolio_value"], alpha=0.9, **style)

        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Portfolio Value ($)", fontsize=14, color="blue")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        if "BTC Buy & Hold" in labels:                 # buy-and-hold first
            i = labels.index("BTC Buy & Hold")
            handles.insert(0, handles.pop(i))
            labels.insert(0, labels.pop(i))
        ax.legend(handles, labels, fontsize=12, loc="upper left")
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12, colors="blue")
        ax.axhline(y=initial_capital, color="gray", alpha=0.5, linewidth=1)

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = [n for n in dir() if not n.startswith("_")]
