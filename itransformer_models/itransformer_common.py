"""Shared code for the iTransformer forecasting pipeline.

The protocol — feature construction, lookback, split dates, scaling, epoch
selection, metrics, benchmarks, Diebold-Mariano tests and the CV + Optuna
strategy search — is *imported* from ``lstm_models/lstm_common.py`` rather than
reimplemented. Only the forecaster changes. That is deliberate: the LSTM is
evaluated on the same target, lookback, periods and preprocessing as the
iTransformer, and sharing one module is the only way to guarantee that claim
rather than assert it.

What this module adds:

* ``ITransformer`` built on the ``iTransformer`` package (the model the
  manuscript uses), with the inverted-attention + RevIN configuration.
* ``train_eval`` — the same training loop as the LSTM's, differing only in the
  model it builds and in exposing the iTransformer's own hyperparameters.
* ``ITransformerPredictor`` — a seed ensemble exposing ``predict_windows``,
  which is all ``lstm_common``'s backtesting machinery requires.

Sentiment comes from the out-of-fold column; see ``oof_finbert_scoring.ipynb``
and ``oof_score_full_stream.ipynb``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_LSTM_DIR = REPO / "lstm_models"
for _p in (str(_LSTM_DIR), str(REPO / "backtesting")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The whole protocol, imported rather than copied.
from lstm_common import (  # noqa: E402,F401
    SEQ, TARGET, TRAIN_END, INNER_VAL_DAYS, INNER_VAL_START, UPTREND, SIDEWAYS,
    ANALYSIS_START, BT_PERIODS, BT_TRAIN_END, SENT_FILE_TRAIN, SENT_FILE_BACKTEST,
    INITIAL_CAPITAL, TRANSACTION_COST_PCT, FEATURE_SET_ORDER, NICE_NAME, FOLDER_NAME,
    CLEANED, DATA, PRICE,
    build_feature_sets, common_window, restrict_to_common,
    make_windows, period_windows, regime_mask,
    metrics, ar_forecast, dm_test, bh_qvalues,
    calculate_sharpe_ratio, calculate_max_drawdown, simple_sharpe_ratio,
    create_trades_dataframe,
    cv_optimize_threshold, cv_optimize_risk, cv_optimize_momentum,
    _standardise, _predict_batched, DEVICE,
    # shared figure style for the portfolio-trajectory panels
    BT_STYLE, BT_FIGSIZE, BT_PALETTE, BT_STRATEGIES, BT_FILE_PREFIX, BT_MARKETS,
    BT_BENCHMARKS, load_trade_log, plot_portfolio_panels,
)

OUT = HERE / "out"
CKPT_DIR = HERE / "itransformer_checkpoints"
CACHE = HERE / "cache"
BT_DIR = HERE / "itransformer_backtest_results"
for _d in (OUT, CKPT_DIR, CACHE, BT_DIR):
    _d.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

from iTransformer import iTransformer as _ITransformer  # noqa: E402


class ITransformerForecaster(nn.Module):
    """iTransformer with a single-step head on the target variate.

    Input  : (batch, lookback, variates)
    Output : (batch,) — the standardised next-hour change of the target column.

    The package returns ``{pred_length: (batch, pred_length, variates)}``; the
    target column is selected and the horizon axis squeezed.
    """

    def __init__(self, num_variates: int, dim: int = 128, depth: int = 2,
                 heads: int = 8, dim_head: int = 32, target_col: int = 0):
        super().__init__()
        self.net = _ITransformer(
            num_variates=num_variates,
            lookback_len=SEQ,
            dim=int(dim),
            depth=int(depth),
            heads=int(heads),
            dim_head=int(dim_head),
            pred_length=1,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True,
        )
        self.target_col = int(target_col)

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return out[:, :, self.target_col].squeeze(-1)


# Optimisation schedule: shuffled minibatches, one optimizer step per batch.
BATCH_SIZE = 1024
ACCUMULATION_STEPS = 1
SHUFFLE = True

DEFAULT_HP = dict(dim=128, depth=2, heads=8, dim_head=32, lr=1e-3,
                  batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS,
                  shuffle=SHUFFLE)


def build_itransformer(num_variates: int, hp: dict, seed: int = 0,
                       target_col: int = 0) -> ITransformerForecaster:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ITransformerForecaster(
        num_variates=num_variates,
        dim=int(hp.get("dim", 128)),
        depth=int(hp.get("depth", 2)),
        heads=int(hp.get("heads", 8)),
        dim_head=int(hp.get("dim_head", 32)),
        target_col=target_col,
    ).to(DEVICE)


# --------------------------------------------------------------------------- #
# training  (same loop as lstm_common.train_eval, different model)
# --------------------------------------------------------------------------- #

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
    """One iTransformer training run.

    Inputs and target are standardised on training-period statistics, Huber loss
    on the standardised target, predictions returned in USD. `patience` enables
    epoch selection on the eval window — during model selection that window is
    the inner-validation slice, never a test regime.
    """
    X, y, idx, close_prev, close = make_windows(dfh, target=target)
    tr, ev = train_mask(idx), eval_mask(idx)
    Xtr, ytr = X[tr], y[tr]
    Xev, yev = X[ev], y[ev]
    if max_train is not None:
        Xtr, ytr = Xtr[:max_train], ytr[:max_train]

    Xtr, Xev, ytr_s, mu, sd, ymu, ysd = _standardise(Xtr, Xev, ytr)

    target_col = list(dfh.columns).index(target)
    model = build_itransformer(Xtr.shape[-1], hp, seed=seed, target_col=target_col)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp.get("lr", 1e-3)))
    loss_fn = nn.HuberLoss()

    def _to_device(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        if DEVICE == "cuda" and t.element_size() * t.nelement() < 3.5e9:
            return t.to(DEVICE)
        return t

    Xtr_t, ytr_t, Xev_t = _to_device(Xtr), _to_device(ytr_s), _to_device(Xev)
    batch_size = int(hp.get("batch_size", BATCH_SIZE))
    accum = int(hp.get("accumulation_steps", ACCUMULATION_STEPS))
    shuffle = bool(hp.get("shuffle", SHUFFLE))
    n = len(Xtr_t)
    g = torch.Generator().manual_seed(seed)

    history, best = [], (np.inf, None, -1, None)
    for ep in range(epochs):
        model.train()
        order = (torch.randperm(n, generator=g).to(Xtr_t.device) if shuffle
                 else torch.arange(n, device=Xtr_t.device))
        opt.zero_grad()
        nb = 0
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            xb = Xtr_t[sel].to(DEVICE, non_blocking=True)
            yb = ytr_t[sel].to(DEVICE, non_blocking=True)
            (loss_fn(model(xb), yb) / accum).backward()
            nb += 1
            if nb % accum == 0:
                opt.step(); opt.zero_grad()
        if nb % accum != 0:                      # trailing partial batch
            opt.step(); opt.zero_grad()

        if track or patience is not None:
            model.eval()
            pv = _predict_batched(model, Xev_t) * ysd + ymu
            mae = float(np.abs(pv - yev).mean())
            history.append(mae)
            if verbose:
                print(f"    epoch {ep + 1:3d}  eval MAE {mae:9.2f}")
            if mae < best[0]:
                best = (mae, pv.copy(), ep,
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
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


class ITransformerPredictor:
    """A trained iTransformer (or a seed ensemble) plus its input scaler.

    Exposes ``predict_windows``, which is the only thing ``lstm_common``'s
    backtesting machinery needs, so both model families are driven by literally
    the same strategy code.
    """

    def __init__(self, state_dicts: list[dict], scaler: dict, hp: dict,
                 columns: list[str], num_variates: int, target_col: int = 0):
        self.models = []
        for sd_ in state_dicts:
            m = build_itransformer(num_variates, hp, seed=0, target_col=target_col)
            m.load_state_dict(sd_)
            m.eval()
            self.models.append(m)
        self.scaler = scaler
        self.hp = dict(hp)
        self.columns = list(columns)
        self.num_variates = num_variates
        self.target_col = target_col

    def predict_windows(self, X: np.ndarray) -> np.ndarray:
        mu, sd = self.scaler["mu"], self.scaler["sd"]
        Xt = torch.from_numpy(np.ascontiguousarray(((X - mu) / sd).astype(np.float32)))
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
                        num_variates=self.num_variates, target_col=self.target_col), path)

    @staticmethod
    def load(path: Path) -> "ITransformerPredictor":
        b = torch.load(path, map_location="cpu", weights_only=False)
        return ITransformerPredictor(b["state_dicts"], b["scaler"], b["hp"],
                                     b["columns"], b["num_variates"], b.get("target_col", 0))


# --------------------------------------------------------------------------- #
# hyperparameter search
# --------------------------------------------------------------------------- #

# Same budget as the LSTM study: 50 trials, 5 epochs each, on a reduced corpus,
# scored by inner-validation MAE with median pruning. The search space follows
# the ranges in appendix_hyperparameters.tex.
TUNE_TRIALS = 50
TUNE_EPOCHS = 5
TUNE_MAX_TRAIN = 10_000
TUNE_MAX_TRAIN_SMALL = 5_000     # price+volume, as in the original study


def tune(dfh: pd.DataFrame, name: str, n_trials: int = TUNE_TRIALS,
         seed: int = 0, verbose: bool = True) -> dict:
    """Optuna search on the inner-validation slice. Never touches a test regime."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_train = TUNE_MAX_TRAIN_SMALL if name == "price+volume" else TUNE_MAX_TRAIN
    fit_mask = lambda idx: idx < INNER_VAL_START                       # noqa: E731
    inner_mask = lambda idx: (idx >= INNER_VAL_START) & (idx < TRAIN_END)  # noqa: E731

    def objective(trial):
        hp = dict(
            dim=trial.suggest_categorical("dim", [64, 128, 256]),
            depth=trial.suggest_int("depth", 1, 3),
            heads=trial.suggest_categorical("heads", [8, 16, 32]),
            dim_head=32,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size=BATCH_SIZE, shuffle=SHUFFLE,
            # The Optuna objective steps on every batch; only the final training
            # loop accumulates. With accumulation a 20-epoch trial on the 10k
            # subset makes ~24 updates, which cannot rank configurations.
            accumulation_steps=1,
        )
        r = train_eval(dfh, fit_mask, inner_mask, hp, epochs=TUNE_EPOCHS,
                       seed=seed, track=True, trial=trial, max_train=max_train)
        return float(min(r["history"])) if r["history"] else float("inf")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=n_trials)
    hp = dict(DEFAULT_HP) | study.best_params
    hp.update(batch_size=BATCH_SIZE, dim_head=32,
              accumulation_steps=ACCUMULATION_STEPS, shuffle=SHUFFLE)
    if verbose:
        print(f"{name:26s} best inner-val MAE {study.best_value:8.2f}  {study.best_params}")
    return hp
