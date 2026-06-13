import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing
from typing import Tuple, List, Dict, Any, Optional
import optuna

def calculate_sharpe_ratio(portfolio_values: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from portfolio values"""
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Remove NaN and infinite values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate annualized Sharpe ratio (assuming hourly data, ~8760 hours per year)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0  # Avoid division by zero
    
    # Annualize: multiply by sqrt(8760) for hourly to annual
    annualized_sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(8760)
    
    return annualized_sharpe

def run_single_threshold_backtest(params):
    """Helper function for threshold optimization"""
    from backtest_functions import backtest_profit_threshold_with_trades
    threshold, hold, close_price, val_pred_1h_change, initial_capital, fixed_transaction_cost_pct = params
    try:
        portfolio_values, _, _, _ = backtest_profit_threshold_with_trades(
            close_price, val_pred_1h_change, initial_capital, threshold, fixed_transaction_cost_pct, hold
        )
        sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
        return (portfolio_values[-1], sharpe_ratio, threshold, hold)
    except Exception as e:
        return (initial_capital, 0.0, threshold, hold)

def optimize_threshold_parameters(close_price: np.ndarray,
                                val_pred_1h_change: np.ndarray,
                                initial_capital: float = 10000,
                                fixed_transaction_cost_pct: float = 0.0275) -> float:
    """Optimize threshold parameter for basic threshold backtest using Sharpe ratio"""

    thresholds = np.arange(0.01, 5.0, 0.01)
    hold_hours = np.arange(1, 24*7, 1)
    param_combinations = []
    for threshold in thresholds:
        for hold in hold_hours:
            param_combinations.append((threshold, hold, close_price, val_pred_1h_change,
                                     initial_capital, fixed_transaction_cost_pct))


    print(f"Testing {len(param_combinations)} threshold values...")

    best_threshold = 0
    best_sharpe = -np.inf
    best_final_capital = initial_capital

    # Run sequentially to avoid pickling issues
    for params in param_combinations:
        final_capital, sharpe_ratio, threshold, hold = run_single_threshold_backtest(params)

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_threshold = threshold
            best_final_capital = final_capital
            best_hold = hold

    print(f"Best threshold: {best_threshold:.2f}%, Hold hours: {best_hold}, Sharpe ratio: {best_sharpe:.2f}, Final capital: ${best_final_capital:.2f}")
    return best_threshold, best_hold

def run_single_risk_threshold_backtest(params):
    """Helper function for risk threshold optimization"""
    from backtest_functions import backtest_profit_threshold
    threshold, target_risk, hold_hours, close_price, val_pred_1h_change, initial_capital, fixed_transaction_cost_pct = params
    try:
        portfolio_values, _, _, _ = backtest_profit_threshold(
            close_price,
            val_pred_1h_change,
            initial_capital,
            threshold,
            target_risk_per_trade=target_risk,
            transaction_cost_pct=fixed_transaction_cost_pct,
            hold_hours=hold_hours
        )
        sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
        return (portfolio_values[-1], sharpe_ratio, threshold, target_risk, hold_hours)
    except Exception as e:
        return (initial_capital, 0.0, threshold, target_risk, hold_hours)

def run_single_momentum_backtest(params):
    """Helper function for momentum optimization"""
    from backtest_functions import backtest_ts_momentum
    lookback, hold_hours, signal_strength_threshold, close_price, val_pred_1h_change, best_target_risk, fixed_transaction_cost_pct = params
    try:
        final_capital, portfolio_values, _, _, _ = backtest_ts_momentum(
            close_price,
            predicted_changes=val_pred_1h_change,
            mode='prediction_only',
            lookback=lookback,
            vol_window=24*30,
            hold_hours=hold_hours,
            target_risk_per_trade=best_target_risk,
            signal_strength_threshold=signal_strength_threshold,
            transaction_cost_pct=fixed_transaction_cost_pct
        )
        sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
        return (final_capital, sharpe_ratio, lookback, hold_hours, signal_strength_threshold)
    except ValueError:
        return (0, 0.0, lookback, hold_hours, signal_strength_threshold)

def optimize_risk_parameters(close_price: np.ndarray,
                           val_pred_1h_change: np.ndarray,
                           initial_capital: float = 10000,
                           fixed_transaction_cost_pct: float = 0.0275) -> Tuple[float, float, int]:
    """Optimize threshold, risk, and hold_hours parameters using Sharpe ratio"""

    thresholds = np.arange(0.00, 1.01, 0.1)  # Reduced for testing
    target_risks = np.arange(0.0, 2.50, 0.1)  # Reduced for testing
    hold_hours_range = range(1, 24*7, 1)

    param_combinations = []
    for threshold in thresholds:
        for target_risk in target_risks:
            for hold_hours in hold_hours_range:
                param_combinations.append((threshold, target_risk, hold_hours, close_price, val_pred_1h_change,
                                         initial_capital, fixed_transaction_cost_pct))

    print(f"Testing {len(param_combinations)} parameter combinations...")

    best_threshold = 0
    best_target_risk = 0
    best_hold_hours = 0
    best_sharpe = -np.inf
    best_final_capital = initial_capital

    # Run sequentially to avoid pickling issues
    completed = 0
    for params in param_combinations:
        final_capital, sharpe_ratio, threshold, target_risk, hold_hours = run_single_risk_threshold_backtest(params)

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_threshold = threshold
            best_target_risk = target_risk
            best_hold_hours = hold_hours
            best_final_capital = final_capital

        completed += 1
        if completed % 1000 == 0:
            print(f"Completed {completed}/{len(param_combinations)} combinations")

    print(f"Optimal Profit Threshold: {best_threshold:.2f}%")
    print(f"Optimal Target Risk Per Trade: {best_target_risk:.2f}%")
    print(f"Optimal Hold Hours: {best_hold_hours}")
    print(f"Best Sharpe ratio: {best_sharpe:.2f}, Final capital: ${best_final_capital:.2f}")
    return best_threshold, best_target_risk, best_hold_hours

def optimize_momentum_parameters(close_price: np.ndarray,
                               val_pred_1h_change: np.ndarray,
                               best_target_risk: float,
                               fixed_transaction_cost_pct: float = 0.0275) -> Tuple[int, int, float]:
    """Optimize momentum parameters using Sharpe ratio"""
    from backtest_functions import backtest_ts_momentum

    lookback_range = range(1, 24*7, 1)
    hold_hours_range = range(1, 24*7, 1)
    signal_strength_thresholds = [0.01]  # Will be set from previous optimization

    param_combinations = []
    for lookback, hold_hours, signal_strength_threshold in product(lookback_range, hold_hours_range, signal_strength_thresholds):
        param_combinations.append((lookback, hold_hours, signal_strength_threshold,
                                  close_price, val_pred_1h_change, best_target_risk, fixed_transaction_cost_pct))

    print(f"Total combinations to test: {len(param_combinations)}")

    best_lookback = 0
    best_hold_hours = 0
    best_signal_strength_threshold = 0
    initial_capital_opt = 10000
    best_sharpe = -np.inf
    best_final_capital_opt = initial_capital_opt

    # Run sequentially to avoid pickling issues
    completed = 0
    for params in param_combinations:
        final_capital, sharpe_ratio, lookback, hold_hours, signal_strength_threshold = run_single_momentum_backtest(params)

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_final_capital_opt = final_capital
            best_lookback = lookback
            best_hold_hours = hold_hours
            best_signal_strength_threshold = signal_strength_threshold

        completed += 1
        if completed % 1000 == 0:
            print(f"Completed {completed}/{len(param_combinations)} combinations")

    print(f"Optimal lookback: {best_lookback}")
    print(f"Optimal hold_hours: {best_hold_hours}")
    print(f"Optimal signal_strength_threshold: {best_signal_strength_threshold}")
    print(f"Best Sharpe ratio: {best_sharpe:.2f}, Final capital: ${best_final_capital_opt:.2f}")

    return best_lookback, best_hold_hours, best_signal_strength_threshold


def plot_trades(trades_df: pd.DataFrame,
               title: str = "Trading Points",
               save_path: Optional[str] = None) -> None:
    """Plot trading points"""
    plt.figure(figsize=(12, 6))

    # Define colors for different trade types
    color_map = {
        'long': 'green', 'buy': 'green',
        'short': 'red', 'sell': 'red',
        'exit_long': 'lightgreen',
        'exit_short': 'lightcoral'
    }

    marker_map = {
        'long': '^', 'buy': '^',
        'short': 'v', 'sell': 'v',
        'exit_long': 'o',
        'exit_short': 'o'
    }

    for trade_type in trades_df['trade_type'].unique():
        subset = trades_df[trades_df['trade_type'] == trade_type]
        plt.scatter(subset['datetime'], subset['portfolio_value'],
                   label=f'{trade_type.replace("_", " ").title()} ({len(subset)} trades)',
                   marker=marker_map.get(trade_type, 'o'),
                   s=50 if trade_type in ['long', 'short', 'buy', 'sell'] else 30,
                   alpha=0.8,
                   color=color_map.get(trade_type, 'blue'))

    plt.xlabel('Time')
    plt.ylabel('Portfolio Value at Trade ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def save_trades_to_csv(trades_df: pd.DataFrame,
                      filepath: str,
                      strategy: str = "Backtest") -> None:
    """Save trades dataframe to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    trades_df.to_csv(filepath, index=False)
    print(f"{strategy} trades saved to {filepath} ({len(trades_df)} trades)")

def create_trades_dataframe(trade_times: List[int],
                          trade_values: List[float],
                          trade_types: List[str],
                          plot_index: pd.DatetimeIndex,
                          strategy: str = "Backtest") -> pd.DataFrame:
    """Create standardized trades dataframe"""
    trades_data = []
    for i, trade_time in enumerate(trade_times):
        if trade_time < len(plot_index):
            trades_data.append({
                'datetime': plot_index[trade_time],
                'trade_index': trade_time,
                'portfolio_value': trade_values[i],
                'trade_type': trade_types[i],
                'strategy': strategy
            })

    return pd.DataFrame(trades_data)

def cv_optimize_threshold_parameters(train_df: pd.DataFrame,
                                   model: Any,
                                   device: str,
                                   seq_length: int,
                                   target_col: str,
                                   initial_capital: float = 10000,
                                   fixed_transaction_cost_pct: float = 0.0275,
                                   n_splits: int = 10,
                                   include_target: bool = True) -> Tuple[float, int]:
    """Optimize threshold parameter using k-fold CV on training data with Optuna"""
    from sklearn.model_selection import KFold
    from model_utils import prepare_tensor, make_predictions
    from data_loader import create_sequences
    from backtest_functions import backtest_profit_threshold_with_trades
    
    # Create sequences from training data
    X_train, _ = create_sequences(train_df, target_col, seq_length, include_target=include_target)
    
    # Get close prices (shifted by seq_length)
    close_prices = train_df['close'].values[seq_length:]
    
    kf = KFold(n_splits=n_splits, shuffle=False)  # Time series, no shuffle
    
    # Force CPU usage for CV to avoid memory issues
    cv_device = 'cpu'
    original_device = next(model.parameters()).device
    model.to(cv_device)
    
    n_cores = multiprocessing.cpu_count() - 1
    print(f"Using {n_cores} cores for parallel Optuna optimization (on {cv_device})")
    
    fold_best_thresholds = []
    fold_best_hold_hours = []
    fold_best_sharpes = []
    
    fold_idx = 0
    for train_idx, val_idx in kf.split(X_train):
        fold_idx += 1
        print(f"Processing CV fold {fold_idx}/{n_splits}...")
        
        # Get validation sequences for this fold
        X_val_fold = X_train[val_idx]
        close_price_fold = close_prices[val_idx]
        
        # Make predictions on CPU
        X_val_tensor = prepare_tensor(X_val_fold, cv_device)
        val_pred_fold = make_predictions(model, X_val_tensor, cv_device, train_df.columns.get_loc(target_col))
        
        # Define objective function for this fold
        def objective(trial):
            threshold = trial.suggest_float('threshold', 0.01, 2.0)
            hold_hours = trial.suggest_int('hold_hours', 1, 24*7)
            try:
                portfolio_values, _, _, _ = backtest_profit_threshold_with_trades(
                    close_price_fold, val_pred_fold, initial_capital, threshold, fixed_transaction_cost_pct, hold_hours
                )
                sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
                return sharpe_ratio
            except Exception as e:
                return 0.0
        
        # Run Optuna optimization for this fold
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200, n_jobs=n_cores, timeout=3000)  # 50 trials, 5 min timeout
        
        best_threshold = study.best_params['threshold']
        best_hold_hours = study.best_params['hold_hours']
        best_sharpe = study.best_value
        
        fold_best_thresholds.append(best_threshold)
        fold_best_hold_hours.append(best_hold_hours)
        fold_best_sharpes.append(best_sharpe)
        
        print(f"Fold {fold_idx} - Best threshold: {best_threshold:.2f}%, Hold hours: {best_hold_hours}, Sharpe: {best_sharpe:.2f}")
    
    # Restore original device if we moved the model
    if 'original_device' in locals():
        model.to(original_device)
    
    # Average the best thresholds across folds
    best_threshold = np.mean(fold_best_thresholds)
    best_hold_hours = int(np.mean(fold_best_hold_hours))
    avg_sharpe = np.mean(fold_best_sharpes)
    
    print(f"CV Best threshold: {best_threshold:.2f}%, Hold hours: {best_hold_hours}, Average Sharpe ratio: {avg_sharpe:.2f}")
    return best_threshold, best_hold_hours

def cv_optimize_risk_parameters(train_df: pd.DataFrame,
                              model: Any,
                              device: str,
                              seq_length: int,
                              target_col: str,
                              initial_capital: float = 10000,
                              fixed_transaction_cost_pct: float = 0.0275,
                              n_splits: int = 10,
                              include_target: bool = True) -> Tuple[float, float, int]:
    """Optimize threshold and risk parameters using k-fold CV on training data with Optuna"""
    from sklearn.model_selection import KFold
    from model_utils import prepare_tensor, make_predictions
    from data_loader import create_sequences
    from backtest_functions import backtest_profit_threshold
    
    # Create sequences from training data
    X_train, _ = create_sequences(train_df, target_col, seq_length, include_target=include_target)
    
    # Get close prices (shifted by seq_length)
    close_prices = train_df['close'].values[seq_length:]
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    # Force CPU usage for CV to avoid memory issues
    cv_device = 'cpu'
    original_device = next(model.parameters()).device
    model.to(cv_device)
    
    n_cores = multiprocessing.cpu_count() - 1
    print(f"Using {n_cores} cores for parallel Optuna optimization (on {cv_device})")
    
    fold_best_thresholds = []
    fold_best_risks = []
    fold_best_hold_hours = []
    fold_best_sharpes = []
    
    fold_idx = 0
    for train_idx, val_idx in kf.split(X_train):
        fold_idx += 1
        print(f"Processing CV fold {fold_idx}/{n_splits} for risk optimization...")
        
        X_val_fold = X_train[val_idx]
        close_price_fold = close_prices[val_idx]
        
        X_val_tensor = prepare_tensor(X_val_fold, cv_device)
        val_pred_fold = make_predictions(model, X_val_tensor, cv_device, train_df.columns.get_loc(target_col))
        
        # Define objective function for this fold
        def objective(trial):
            threshold = trial.suggest_float('threshold', 0.01, 2.0)
            target_risk = trial.suggest_float('target_risk', 0.01, 2.5)
            hold_hours = trial.suggest_int('hold_hours', 1, 24*7)
            try:
                portfolio_values, _, _, _ = backtest_profit_threshold(
                    close_price_fold, val_pred_fold, initial_capital, threshold, target_risk_per_trade=target_risk, transaction_cost_pct=fixed_transaction_cost_pct, hold_hours=hold_hours
                )
                sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
                return sharpe_ratio
            except Exception as e:
                return 0.0
        
        # Run Optuna optimization for this fold
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=500, n_jobs=n_cores, timeout=6000)  # 100 trials, 10 min timeout
        
        best_threshold = study.best_params['threshold']
        best_target_risk = study.best_params['target_risk']
        best_hold_hours = study.best_params['hold_hours']
        best_sharpe = study.best_value
        
        fold_best_thresholds.append(best_threshold)
        fold_best_risks.append(best_target_risk)
        fold_best_hold_hours.append(best_hold_hours)
        fold_best_sharpes.append(best_sharpe)
        
        print(f"Fold {fold_idx} - Best threshold: {best_threshold:.2f}%, Risk: {best_target_risk:.2f}%, Hold hours: {best_hold_hours}, Sharpe: {best_sharpe:.2f}")
    
    # Restore original device
    model.to(original_device)
    
    # Average the best parameters across folds
    best_threshold = np.mean(fold_best_thresholds)
    best_target_risk = np.mean(fold_best_risks)
    best_hold_hours = int(np.mean(fold_best_hold_hours))
    avg_sharpe = np.mean(fold_best_sharpes)
    
    print(f"CV Optimal threshold: {best_threshold:.2f}%, Risk: {best_target_risk:.2f}%, Hold hours: {best_hold_hours}, Average Sharpe: {avg_sharpe:.2f}")
    return best_threshold, best_target_risk, best_hold_hours

def cv_optimize_momentum_parameters(train_df: pd.DataFrame,
                                  model: Any,
                                  seq_length: int,
                                  target_col: str,
                                  fixed_transaction_cost_pct: float = 0.0275,
                                  n_splits: int = 10,
                                  include_target: bool = True) -> Tuple[int, int, float, float]:
    """Optimize momentum parameters using k-fold CV on training data with Optuna"""
    from sklearn.model_selection import KFold
    from model_utils import prepare_tensor, make_predictions
    from data_loader import create_sequences
    from backtest_functions import backtest_ts_momentum
    
    # Create sequences from training data
    X_train, _ = create_sequences(train_df, target_col, seq_length, include_target=include_target)
    
    # Get close prices (shifted by seq_length)
    close_prices = train_df['close'].values[seq_length:]
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    # Force CPU usage for CV to avoid memory issues
    cv_device = 'cpu'
    original_device = next(model.parameters()).device
    model.to(cv_device)
    
    n_cores = multiprocessing.cpu_count() - 1
    print(f"Using {n_cores} cores for parallel Optuna optimization (on {cv_device})")
    
    fold_best_lookbacks = []
    fold_best_hold_hours = []
    fold_best_signal_thresholds = []
    fold_best_target_risks = []
    fold_best_sharpes = []
    
    fold_idx = 0
    for train_idx, val_idx in kf.split(X_train):
        fold_idx += 1
        print(f"Processing CV fold {fold_idx}/{n_splits} for momentum optimization...")
        
        X_val_fold = X_train[val_idx]
        close_price_fold = close_prices[val_idx]
        
        X_val_tensor = prepare_tensor(X_val_fold, cv_device)
        val_pred_fold = make_predictions(model, X_val_tensor, cv_device, train_df.columns.get_loc(target_col))
        
        # Define objective function for this fold
        def objective(trial):
            lookback = trial.suggest_int('lookback', 1, 24*7)
            hold_hours = trial.suggest_int('hold_hours', 1, 24*7)
            signal_thresh = trial.suggest_float('signal_thresh', 0.01, 2.0)
            target_risk = trial.suggest_float('target_risk', 0.0, 2.5)
            try:
                final_capital, portfolio_values, _, _, _ = backtest_ts_momentum(
                    prices=close_price_fold,
                    predicted_changes=val_pred_fold,
                    lookback=lookback, 
                    hold_hours=hold_hours,
                    target_risk_per_trade=target_risk, 
                    threshold=signal_thresh,
                    initial_capital=10000, 
                    transaction_cost_pct=fixed_transaction_cost_pct)
                sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
                return sharpe_ratio
            except Exception as e:
                return 0.0
        
        # Run Optuna optimization for this fold
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1000, n_jobs=n_cores, timeout=1200)  # 200 trials, 20 min timeout
        
        best_lookback = study.best_params['lookback']
        best_hold_hours = study.best_params['hold_hours']
        best_signal_thresh = study.best_params['signal_thresh']
        best_target_risk = study.best_params['target_risk']
        best_sharpe = study.best_value
        
        fold_best_lookbacks.append(best_lookback)
        fold_best_hold_hours.append(best_hold_hours)
        fold_best_signal_thresholds.append(best_signal_thresh)
        fold_best_target_risks.append(best_target_risk)
        fold_best_sharpes.append(best_sharpe)
        
        print(f"Fold {fold_idx} - Best lookback: {best_lookback}, hold_hours: {best_hold_hours}, signal_thresh: {best_signal_thresh:.3f}, target_risk: {best_target_risk:.3f}, Sharpe: {best_sharpe:.2f}")
    
    # Restore original device
    model.to(original_device)
    
    # Average the best parameters across folds
    best_lookback = int(np.mean(fold_best_lookbacks))
    best_hold_hours = int(np.mean(fold_best_hold_hours))
    best_signal_thresh = np.mean(fold_best_signal_thresholds)
    best_target_risk = np.mean(fold_best_target_risks)
    avg_sharpe = np.mean(fold_best_sharpes)
    
    print(f"CV Optimal lookback: {best_lookback}, hold_hours: {best_hold_hours}, signal_thresh: {best_signal_thresh:.3f}, target_risk: {best_target_risk:.3f}, Average Sharpe: {avg_sharpe:.2f}")
    return best_lookback, best_hold_hours, best_signal_thresh, best_target_risk