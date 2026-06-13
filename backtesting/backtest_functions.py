
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable

def _generic_backtest(hourly_price: np.ndarray,
                     predicted_changes: np.ndarray,
                     initial_capital: float,
                     threshold: float,
                     transaction_cost: float,
                     hold_hours: int,
                     size_func: Callable[[int], float],
                     predict_func: Callable[[int], float],
                     start_index: int = 1) -> Tuple[List[float], List[int], List[float], List[str]]:
    """Generic backtesting function with customizable size and prediction logic"""
    capital = initial_capital
    portfolio_values = [capital]
    position = 0  # 1 = long, -1 = short
    prices = np.asarray(hourly_price)

    trade_times = []
    trade_values = []
    trade_types = []  # 'long', 'short', 'exit_long', 'exit_short'

    trade_times.append(0)
    trade_values.append(capital)
    trade_types.append('start')

    bars_in_position = 0

    for i in range(start_index, len(hourly_price)):
        actual_pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]

        if position != 0:
            capital *= (1 + position * position_size * actual_pct_change)
            bars_in_position += 1

        predicted_pct_change = predict_func(i)

        desired = 0
        if not np.isnan(predicted_pct_change):
            if predicted_pct_change > threshold:
                desired = 1
            elif predicted_pct_change < -threshold:
                desired = -1

        if position == 0:
            if desired != 0:
                position = desired
                position_size = size_func(i - 1)
                bars_in_position = 0
                capital -= position_size * capital * transaction_cost
                trade_times.append(i)
                trade_values.append(capital)
                trade_types.append('long' if desired == 1 else 'short')
        else:
            should_flip = desired != 0 and desired != position
            if should_flip:
                trade_times.append(i)
                trade_values.append(capital)
                trade_types.append('exit_long' if position == 1 else 'exit_short')
                position = desired
                position_size = size_func(i - 1)
                bars_in_position = 0
                capital -= position_size * capital * transaction_cost
                trade_times.append(i)
                trade_values.append(capital)
                trade_types.append('long' if desired == 1 else 'short')
            elif bars_in_position >= hold_hours:
                trade_times.append(i)
                trade_values.append(capital)
                trade_types.append('exit_long' if position == 1 else 'exit_short')
                position = 0
                position_size = 0.0
                bars_in_position = 0

        portfolio_values.append(capital)

    portfolio_values.append(capital)
    trade_times.append(len(hourly_price)-1)
    trade_values.append(capital)
    trade_types.append('end')

    return portfolio_values, trade_times, trade_values, trade_types

def backtest_profit_threshold_with_trades(hourly_price: np.ndarray,
                                         predicted_changes: np.ndarray,
                                         initial_capital: float = 10000,
                                         threshold: float = 0.01,
                                         transaction_cost_pct: float = 0.1,                                         
                                         hold_hours: int = 1) -> Tuple[List[float], List[int], List[float], List[str]]:
    """Basic threshold-based backtesting with trade tracking"""
    threshold /= 100
    transaction_cost = transaction_cost_pct / 100
    prices = np.asarray(hourly_price)
    preds = np.asarray(predicted_changes)

    def size_func(idx: int) -> float:
        return 0.99

    def predict_func(i: int) -> float:
        predicted_price = prices[i - 1] + preds[i]
        return (predicted_price - prices[i - 1]) / prices[i - 1]

    return _generic_backtest(hourly_price, predicted_changes, initial_capital, threshold, transaction_cost, hold_hours, size_func, predict_func)

def backtest_profit_threshold(hourly_price: np.ndarray,
                             predicted_changes: np.ndarray,
                             initial_capital: float = 10000,
                             threshold: float = 0.01,
                             vol_window: int = 24*30,
                             target_risk_per_trade: float = 0.5,
                             transaction_cost_pct: float = 0.1,
                             min_size: float = 0.05,
                             max_size: float = 1.0,                             
                             hold_hours: int = 1,) -> Tuple[List[float], List[int], List[float], List[str]]:
    """Risk-based threshold backtesting with position sizing"""
    threshold /= 100.0
    target_risk_per_trade /= 100.0
    transaction_cost = transaction_cost_pct / 100.0
    prices = np.asarray(hourly_price)
    preds = np.asarray(predicted_changes)

    # Calculate volatility
    returns = np.diff(prices) / prices[:-1]
    vol_series = pd.Series(returns).rolling(vol_window).std().to_numpy()
    vol_series = np.concatenate([[0], vol_series])

    def size_func(idx: int) -> float:
        v = vol_series[idx]
        if np.isnan(v) or v < 1e-8:
            v = 0.01
        raw = target_risk_per_trade / v
        return float(np.clip(raw, min_size, max_size))

    def predict_func(i: int) -> float:
        predicted_price = prices[i - 1] + preds[i]
        return (predicted_price - prices[i - 1]) / prices[i - 1]

    return _generic_backtest(hourly_price, predicted_changes, initial_capital, threshold, transaction_cost, hold_hours, size_func, predict_func)

def backtest_ts_momentum(prices: np.ndarray,
                        predicted_changes: Optional[np.ndarray] = None,
                        lookback: int = 24,
                        vol_window: int = 24*30,
                        target_risk_per_trade: float = 0.5,
                        threshold: float = 0.1,
                        min_size: float = 0.05,
                        max_size: float = 1.0,
                        initial_capital: float = 10000,
                        hold_hours: int = 1,
                        transaction_cost_pct: float = 0.1) -> Tuple[float, List[float], List[int], List[float], List[str]]:

    prices = np.asarray(prices, float)
    preds = np.asarray(predicted_changes, float) if predicted_changes is not None else None
    n = len(prices)
    if n < lookback + 2:
        raise ValueError(
            f"Not enough data for lookback window of {lookback}. Need at least {lookback + 2} data points."
        )

    target_risk_per_trade /= 100.0
    transaction_cost = transaction_cost_pct / 100.0
    threshold /= 100.0

    returns = np.diff(prices) / prices[:-1]
    vol_series = pd.Series(returns).rolling(vol_window).std().to_numpy()
    vol_series = np.concatenate([[0], vol_series])

    def size_func(idx: int) -> float:
        v = vol_series[idx]
        if np.isnan(v) or v < 1e-8:
            v = 0.01
        raw = target_risk_per_trade / v
        return float(np.clip(raw, min_size, max_size))

    def predict_func(i: int) -> float:
        predicted_price = prices[i - 1] + preds[i]
        return (predicted_price - prices[i - lookback]) / prices[i - lookback]

    portfolio_values, trade_times, trade_values, trade_types = _generic_backtest(
        prices, preds, initial_capital, threshold, transaction_cost, hold_hours, 
        size_func, predict_func, start_index=1 + lookback
    )

    return portfolio_values[-1], portfolio_values, trade_times, trade_values, trade_types

