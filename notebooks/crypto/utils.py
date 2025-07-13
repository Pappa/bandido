import os
import pandas as pd

SYMBOLS = ['BTC', 'DOGE', 'XRP', 'ETH', 'SOL']


def load_and_prepare_data(filepath, symbols):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure the file exists.")
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, compression='gzip', parse_dates=['timestamp']).set_index('timestamp')
    # remove '/USDT' from symbol name
    df['symbol'] = df['symbol'].str.split('/', n=1).str[0]
    
    all_data = {}
    print("Processing symbols:", symbols)
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol][['close', 'volume']]
        symbol_df['close_return'] = symbol_df['close'].pct_change()
        symbol_df['volume_return'] = symbol_df['volume'].pct_change()
        all_data[symbol] = symbol_df[['close_return', 'volume_return', 'close']]
    combined_df = pd.concat(all_data, axis=1)
    combined_df.columns = ['_'.join(col).strip() for col in combined_df.columns.values]
    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True)
    print(f"Data prepared. Shape: {combined_df.shape}")
    return combined_df