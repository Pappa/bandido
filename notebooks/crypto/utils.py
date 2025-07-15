import pandas as pd

# List of symbols to be used throughout the project.
SYMBOLS = ['BTC', 'DOGE', 'XRP', 'ETH', 'SOL']


def preprocess_data(df):
    df.sort_values(by=['symbol', 'timestamp'], inplace=True)
    # Ensure symbol format is consistent (e.g., 'BTC' not 'BTC/USDT')
    df['symbol'] = df['symbol'].str.split('/', n=1).str[0]

    df['close_return'] = df['close'].pct_change()
    df['volume_return'] = df['volume'].pct_change()

    df['rsi'] = calculate_rsi(df)

    return df.dropna()

def calculate_rsi(data, window=14):
    delta = data['close'] - data['open']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.groupby(data['symbol']).transform(
        lambda x: x.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    )
    avg_loss = loss.groupby(data['symbol']).transform(
        lambda x: x.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    )

    relative_strength = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi



def create_wide_format_data(long_df: pd.DataFrame, symbols: list[str], features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a long-format DataFrame into two wide-format DataFrames.

    1. An observation DataFrame where each row is a timestamp and columns are flattened features.
    2. A prices DataFrame for easy reward calculation.

    Args:
        long_df (pd.DataFrame): The input DataFrame in long format.
        symbols (list[str]): The ordered list of crypto symbols.
        features (list[str]): The list of feature columns to use for observations.

    Returns:
        A tuple containing (observation_df, prices_df).
    """
    print("Converting data to wide format for observations and prices...")
    
    # --- Create the wide observation DataFrame ---
    observation_df = long_df.pivot(
        index='timestamp', 
        columns='symbol', 
        values=features
    )
    
    # The pivot creates a multi-level column index, e.g., ('rsi', 'BTC').
    # We need to flatten this into a single-level index, e.g., 'BTC_rsi'.
    # We must also enforce the correct order: all of BTC's features, then all of ETH's, etc.
    new_obs_cols = []
    final_obs_cols = []
    for symbol in symbols:
        for feature in features:
            new_obs_cols.append(f'{symbol}_{feature}')
            final_obs_cols.append((feature, symbol)) # Original multi-level name
            
    observation_df = observation_df[final_obs_cols] # Enforce column order
    observation_df.columns = new_obs_cols # Rename to single-level
    
    # --- Create the wide prices DataFrame ---
    prices_df = long_df.pivot(
        index='timestamp',
        columns='symbol',
        values='close'
    )
    prices_df = prices_df[symbols] # Enforce column order
    
    # Drop any rows with NaN values that might result from pivoting
    observation_df.dropna(inplace=True)
    prices_df.dropna(inplace=True)
    
    # Align the indexes of both dataframes to ensure they match
    aligned_index = observation_df.index.intersection(prices_df.index)
    observation_df = observation_df.loc[aligned_index]
    prices_df = prices_df.loc[aligned_index]
    
    print("Wide format conversion complete.")
    return observation_df, prices_df