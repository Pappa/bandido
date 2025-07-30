import pandas as pd
import numpy as np

# List of symbols to be used throughout the project.
SYMBOLS = ["BTC", "DOGE", "XRP", "ETH", "SOL"]


def preprocess_data(df):
    df.sort_values(by=["symbol", "timestamp"], inplace=True)
    # Ensure symbol format is consistent (e.g., 'BTC' not 'BTC/USDT')
    df["symbol"] = df["symbol"].str.split("/", n=1).str[0]

    # Ensure 0 volume does not result in division by 0
    df["volume"] = df["volume"].replace(0, 1e-9)
    df["price_change"] = df.groupby("symbol")["close"].transform(
        lambda x: x.pct_change()
    )
    df["volume_change"] = df.groupby("symbol")["volume"].transform(
        lambda x: x.pct_change()
    )

    df["rsi"] = calculate_rsi(df)
    macd = calculate_macd(df)
    bb = calculate_bollinger_bands(df)
    return pd.concat([df, macd, bb], axis=1).dropna()


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for each symbol in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame with 'timestamp', 'symbol', and 'close' columns.
        Should be pre-sorted by symbol and then by timestamp.
        window (int): The lookback period for the RSI calculation (default is 14).

    Returns:
        pd.Series: The RSI for each symbol.
    """
    delta = data["close"] - data["open"]
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.groupby(data["symbol"]).transform(
        lambda x: x.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    )
    avg_loss = loss.groupby(data["symbol"]).transform(
        lambda x: x.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    )

    relative_strength = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def calculate_macd(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD), its signal line,
    and histogram for each symbol in the DataFrame.

    Returns a new DataFrame with 'macd', 'macd_signal', and 'macd_hist' columns.

    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp', 'symbol', and 'close' columns.
                           Should be sorted by symbol and then by timestamp.
        short_window (int): The lookback period for the short-term EMA.
        long_window (int): The lookback period for the long-term EMA.
        signal_window (int): The lookback period for the signal line's EMA.

    Returns:
        pd.DataFrame: The DataFrame with the new MACD-related columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    data = df.copy()

    # Ensure the DataFrame is sorted correctly for window calculations
    data.sort_values(by=["symbol", "timestamp"], inplace=True)

    # --- Step 1: Calculate the short-term and long-term EMAs ---
    # We group by symbol to calculate the EMA for each symbol independently.
    # The `.transform()` method allows us to apply the calculation to each group
    # and return a result that is indexed identically to the original DataFrame.
    ema_short = data.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=short_window, adjust=False).mean()
    )

    ema_long = data.groupby("symbol")["close"].transform(
        lambda x: x.ewm(span=long_window, adjust=False).mean()
    )

    # --- Step 2: Calculate the MACD Line ---
    data["macd"] = ema_short - ema_long

    # --- Step 3: Calculate the Signal Line ---
    # The signal line is an EMA of the MACD line itself.
    data["macd_signal"] = data.groupby("symbol")["macd"].transform(
        lambda x: x.ewm(span=signal_window, adjust=False).mean()
    )

    # --- Step 4: Calculate the MACD Histogram ---
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["macd_hist_prev"] = data["macd_hist"].shift(1)

    return data[["macd", "macd_signal", "macd_hist", "macd_hist_prev"]]


def calculate_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: int = 2
) -> pd.DataFrame:
    """
    Calculates Bollinger Bands for each symbol in the DataFrame and returns a new DataFrame with the new Bollinger Band columns.

    Args:
        df (pd.DataFrame): The input DataFrame with 'timestamp', 'symbol', and 'close' columns.
                           Should be sorted by symbol and then by timestamp.
        window (int): The lookback period for the moving average and standard deviation (default is 20).
        num_std (int): The number of standard deviations to use for the upper and lower bands (default is 2).

    Returns:
        pd.DataFrame: A new DataFrame with the new Bollinger Band columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    data = df.copy()

    # Ensure the DataFrame is sorted correctly for the rolling calculations
    data.sort_values(by=["symbol", "timestamp"], inplace=True)

    # --- Step 1: Calculate the rolling mean (Middle Band) ---
    # We group by symbol to calculate the SMA for each symbol independently.
    # The .rolling() method creates a rolling window object.
    rolling_mean = data.groupby("symbol")["close"].rolling(window=window).mean()

    # --- Step 2: Calculate the rolling standard deviation ---
    rolling_std = data.groupby("symbol")["close"].rolling(window=window).std()

    # After a groupby().rolling(), the result has a MultiIndex. We need to
    # drop the 'symbol' level of the index to align it back with the original DataFrame.
    bb_middle = rolling_mean.reset_index(level=0, drop=True)

    # Use the calculated rolling standard deviation to compute the upper and lower bands.
    std = rolling_std.reset_index(level=0, drop=True) * num_std
    bb_upper = bb_middle + std
    bb_lower = bb_middle - std

    # Calculate the percent B and bandwidth
    bb_percent_b = ((data["close"] - bb_lower) / (bb_upper - bb_lower)).rename(
        "bb_percent_b"
    )
    bb_bandwidth = ((bb_upper - bb_lower) / bb_middle).rename("bb_bandwidth")

    return pd.concat([bb_percent_b, bb_bandwidth], axis=1)


def create_wide_format_data(
    long_df: pd.DataFrame, symbols: list[str], features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    # --- Create the wide observation DataFrame ---
    observation_df = long_df.pivot(index="timestamp", columns="symbol", values=features)

    # The pivot creates a multi-level column index, e.g., ('rsi', 'BTC').
    # We need to flatten this into a single-level index, e.g., 'BTC_rsi'.
    # We must also enforce the correct order: all of BTC's features, then all of ETH's, etc.
    new_obs_cols = []
    final_obs_cols = []
    for symbol in symbols:
        for feature in features:
            new_obs_cols.append(f"{symbol}_{feature}")
            final_obs_cols.append((feature, symbol))  # Original multi-level name

    observation_df = observation_df[final_obs_cols]  # Enforce column order
    observation_df.columns = new_obs_cols  # Rename to single-level

    # --- Create the wide prices DataFrame ---
    prices_df = long_df.pivot(index="timestamp", columns="symbol", values="close")
    prices_df = prices_df[symbols]  # Enforce column order

    # Drop any rows with NaN values that might result from pivoting
    observation_df.dropna(inplace=True)
    prices_df.dropna(inplace=True)

    # Align the indexes of both dataframes to ensure they match
    aligned_index = observation_df.index.intersection(prices_df.index)
    observation_df = observation_df.loc[aligned_index]
    prices_df = prices_df.loc[aligned_index]

    return observation_df, prices_df


def collate_results(
    model,
    train_steps,
    eval_steps,
    alpha,
    gamma,
    learming_rate,
    max_buy_price,
    max_sell_price,
    trade_fee,
    invalid_action_penalty,
    technical_features,
    test_return,
    eval_return,
    test_valid_action_rate,
    eval_valid_action_rate,
    test_optimal_trade_rate,
    eval_optimal_trade_rate,
):
    columns = [
        "timestamp",
        "model",
        "train_steps",
        "eval_steps",
        "alpha",
        "gamma",
        "learming_rate",
        "max_buy_price",
        "max_sell_price",
        "trade_fee",
        "invalid_action_penalty",
        "technical_features",
        "num_technical_features",
        "test_return",
        "eval_return",
        "test_valid_action_rate",
        "eval_valid_action_rate",
        "test_optimal_trade_rate",
        "eval_optimal_trade_rate",
    ]
    values = [
        [
            pd.Timestamp.now(),
            model,
            train_steps,
            eval_steps,
            alpha,
            gamma,
            learming_rate,
            max_buy_price,
            max_sell_price,
            trade_fee,
            invalid_action_penalty,
            '|'.join(technical_features),
            len(technical_features),
            test_return,
            eval_return,
            test_valid_action_rate,
            eval_valid_action_rate,
            test_optimal_trade_rate,
            eval_optimal_trade_rate,
        ]
    ]
    return pd.DataFrame(values, columns=columns)
