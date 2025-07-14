import os
import pandas as pd

# --- Centralized Configuration ---
# List of symbols to be used throughout the project.
# The data loading function will handle stripping '/USDT' if present.
SYMBOLS = ['BTC', 'DOGE', 'XRP', 'ETH', 'SOL']

# Data and Model Paths
DATA_FILEPATH = 'data/ohlcv.csv.gz'
POLICY_SAVE_PATH = 'policy'

# Model Hyperparameters
CONTEXT_LENGTH = 10
NUM_TRAINING_STEPS = 1000  # Increased for more meaningful training
ALPHA = 1.0 # LinUCB exploration parameter


# --- Data Loading Function ---
def load_and_prepare_data(filepath, symbols):
    """Loads and prepares the data from the source CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Please ensure the file exists.")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, compression='gzip', parse_dates=['timestamp']).set_index('timestamp')
    
    # Ensure symbol format is consistent (e.g., 'BTC' not 'BTC/USDT')
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
    
    # Use forward-fill for missing values and then drop any remaining NaNs at the start
    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True)
    
    print(f"Data prepared. Shape: {combined_df.shape}")
    return combined_df