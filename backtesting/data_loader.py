import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def load_btc_data(path: str = "./data/") -> pd.DataFrame:
    """Load BTCUSDT price data"""
    df = pd.read_csv(path+"BTCUSDT-last.csv", sep='|')
    df = df.rename(columns={
        '1502942400': 'timestamp',
        '4261.48': 'open',
        '4261.48.1': 'high',
        '4261.48.2': 'low',
        '4261.48.3': 'close',
        '1.775183': 'btc_volume_sell',
        '320.390851': 'usd_volume_buy',
        '0.075183': 'btc_volume_buy',
        '7564.906851': 'usd_volume_sell',
        '3': 'activity'
    })
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('datetime', inplace=True)
    df['price_change_1h'] = df['close'].diff(periods=60)  # Assuming 1-minute data
    return df

def load_sentiment_data(path: str = "./data/") -> pd.DataFrame:
    """Load sentiment data"""
    df_sentiments = pd.read_csv(path+"cleaned/predicted_sentiments2.csv")
    df_sentiments['datetime'] = pd.to_datetime(df_sentiments['timestamp'], unit='s')
    df_sentiments.set_index('datetime', inplace=True)
    df_sentiments.drop('timestamp', axis=1, inplace=True)
    return df_sentiments

def load_technical_data(path: str = "./data/") -> Dict[str, pd.DataFrame]:
    """Load technical indicator data"""
    data = {}
    data['gold'] = pd.read_csv(path+"cleaned/GOLD_hourly.csv")
    data['sp500'] = pd.read_csv(path+"cleaned/SP500_hourly.csv")
    data['vxx'] = pd.read_csv(path+"cleaned/VXX_hourly.csv")
    data['tx_count'] = pd.read_csv(path+"cleaned/tx_count_10years.csv")
    data['hashrate'] = pd.read_csv(path+"cleaned/hashrate_10years.csv")

    for name, df in data.items():
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
        elif 'hour' in df.columns:
            df['datetime'] = pd.to_datetime(df['hour'], utc=True)
            df.set_index('datetime', inplace=True)
            df.drop('hour', axis=1, inplace=True)
        
        # Rename columns to avoid conflicts
        if name in ['gold', 'sp500', 'vxx']:
            df.rename(columns={'close': f'close_{name}'}, inplace=True)
            # Keep only close for these
            df = df[[f'close_{name}']]
        elif name == 'tx_count':
            pass  # tx_count column is already correct
        elif name == 'hashrate':
            pass  # blocks_mined and est_hashrate columns are already correct
        
        data[name] = df
    
    return data

def process_sentiment_hourly(df_sentiments: pd.DataFrame) -> pd.DataFrame:
    """Process sentiment data to hourly with aggregations and decay"""
    df_hourly = df_sentiments.resample('1h').agg({
        'predicted_sentiment': ['min', 'max', 'mean']
    })
    df_hourly.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                        for col in df_hourly.columns.values]
    df_hourly = df_hourly.rename(columns={'predicted_sentiment_mean': 'predicted_sentiment_24h_avg'})
    df_hourly['predicted_sentiment_24h_avg'] = df_hourly['predicted_sentiment_24h_avg'].rolling(window=24).mean()

    # Add exponentially decaying sentiment
    decay_sentiment_values = calculate_exponential_decay_sentiment(df_hourly)
    df_hourly['predicted_sentiment_4h_decay'] = decay_sentiment_values
    df_hourly = df_hourly.fillna(0)
    return df_hourly

def calculate_exponential_decay_sentiment(df: pd.DataFrame,
                                        decay_hours: int = 4,
                                        decay_rate: float = 0.5) -> np.ndarray:
    """Calculate exponentially decaying sentiment score"""
    decay_sentiment = np.zeros(len(df))

    for i in range(len(df)):
        current_sentiment = df['predicted_sentiment_24h_avg'].iloc[i]
        if current_sentiment != 0:
            for h in range(decay_hours):
                future_idx = i + h
                if future_idx < len(df):
                    decay_factor = decay_rate ** h
                    decay_sentiment[future_idx] += current_sentiment * decay_factor

    return decay_sentiment

def merge_dataframes(base_df: pd.DataFrame,
                    sentiment_df: Optional[pd.DataFrame] = None,
                    technical_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Merge all dataframes with proper timezone handling"""
    # Ensure UTC timezone
    dfs_to_merge = [base_df]
    if sentiment_df is not None:
        dfs_to_merge.append(sentiment_df)

    if technical_data:
        dfs_to_merge.extend(technical_data.values())

    # Set UTC timezone for all
    for df in dfs_to_merge:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')

    # Start with base df
    merged = base_df.copy()

    # Merge sentiment
    if sentiment_df is not None:
        merged = merged.merge(sentiment_df, left_index=True, right_index=True, how='left')

    # Merge technical data
    if technical_data:
        for name, df in technical_data.items():
            # For technical data, merge without suffixes since we want the column names as-is
            merged = merged.merge(df, left_index=True, right_index=True, how='left')

        # Forward fill technical indicators
        tech_cols = [col for col in merged.columns if any(x in col.lower() for x in ['gold', 'sp500', 'vxx'])]
        merged[tech_cols] = merged[tech_cols].ffill()

    return merged

def create_hourly_data(merged_df: pd.DataFrame,
                      features: List[str],
                      target_column: str = 'price_change_1h') -> pd.DataFrame:
    """Resample to hourly data with specified features"""
    agg_dict = {col: 'last' for col in features}
    agg_dict[target_column] = 'last'
    agg_dict.update({
        'usd_volume_buy': 'sum',
        'usd_volume_sell': 'sum'
    })

    df_hourly = merged_df.resample('1h').agg(agg_dict).dropna()
    return df_hourly

def create_validation_periods(df_hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create training and validation datasets"""
    # Check if the dataframe index is timezone-aware
    is_tz_aware = df_hourly.index.tzinfo is not None
    
    if is_tz_aware:
        period1_start = pd.Timestamp('2024-06-01', tz='UTC')
        period1_end = pd.Timestamp('2024-12-01', tz='UTC')
        period2_start = pd.Timestamp('2025-01-01', tz='UTC')
        period2_end = pd.Timestamp('2025-07-01', tz='UTC')
    else:
        period1_start = pd.Timestamp('2024-06-01')
        period1_end = pd.Timestamp('2024-12-01')
        period2_start = pd.Timestamp('2025-01-01')
        period2_end = pd.Timestamp('2025-07-01')

    val_df_period1 = df_hourly[(df_hourly.index >= period1_start) & (df_hourly.index < period1_end)]
    val_df_period2 = df_hourly[(df_hourly.index >= period2_start) & (df_hourly.index < period2_end)]
    train_df = df_hourly[df_hourly.index < period1_start]

    return train_df, val_df_period1, val_df_period2

def create_sequences(data_df: pd.DataFrame,
                    target_column: str,
                    seq_length: int,
                    include_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for model input"""
    X, y = [], []
    if include_target:
        data = data_df.values
    else:
        # Exclude target column from input features
        feature_columns = [col for col in data_df.columns if col != target_column]
        data = data_df[feature_columns].values
    target_data = data_df[target_column].values

    for i in range(len(data_df) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target_data[i+seq_length])

    return np.array(X), np.array(y)