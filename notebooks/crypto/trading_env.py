import numpy as np
import pandas as pd
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class BasicCryptoTradingEnvironment(py_environment.PyEnvironment):
    """
    A TF-Agents environment for a crypto trading multi-armed bandit.

    This version is optimized for performance by accepting pre-calculated,
    "wide-format" DataFrames for observations and prices.
    """
    def __init__(self, observation_df: pd.DataFrame, prices_df: pd.DataFrame, symbols: list[str]):
        super().__init__()

        # --- Store pre-calculated data ---
        self._obs_data = observation_df
        self._price_data = prices_df
        self._symbols = symbols
        self._num_cryptos = len(symbols)
        
        # --- Define Action and Observation Specs ---
        num_actions = self._num_cryptos * 3  # BUY, HOLD, SELL for each crypto
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action'
        )

        # The observation shape is now simply the number of columns in the obs_df
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self._obs_data.columns),), dtype=np.float32, name='context'
        )
        
        # --- Environment State ---
        self._current_step_index = 0
        self._episode_ended = False

    @property
    def current_step(self):
        return self._current_step_index

    @property
    def symbols(self):
        return self._symbols

    @property
    def price_data(self):
        return self._price_data

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _get_observation(self) -> np.ndarray:
        """Returns the observation for the current step."""
        return self._obs_data.iloc[self._current_step_index].values.astype(np.float32)

    def _reset(self) -> ts.TimeStep:
        """Resets the environment to the first time step."""
        self._current_step_index = 0
        self._episode_ended = False
        return ts.restart(self._get_observation())

    def _step(self, action: int) -> ts.TimeStep:
        """Applies an action, calculates the reward, and steps the environment."""
        if self._episode_ended:
            return self.reset()

        # Decode the action to find the symbol and trade type
        crypto_index = action // 3
        trade_type_idx = action % 3  # 0: BUY, 1: HOLD, 2: SELL
        
        # Calculate the reward
        reward = 0.0
        if trade_type_idx != 1:  # No reward for HOLD
            symbol_to_trade = self._symbols[crypto_index]
            
            # Use the dedicated prices DataFrame for a fast lookup
            current_price = self._price_data.iloc[self._current_step_index][symbol_to_trade]
            next_price = self._price_data.iloc[self._current_step_index + 1][symbol_to_trade]
            
            if trade_type_idx == 0:  # BUY
                reward = (next_price - current_price) / current_price
            elif trade_type_idx == 2:  # SELL
                reward = (current_price - next_price) / current_price

        # Advance the environment state
        self._current_step_index += 1
        
        # Check if the episode has ended
        if self._current_step_index >= len(self._obs_data) - 1:
            self._episode_ended = True

        # Return the correct TimeStep object
        if self._episode_ended:
            # We must still provide an observation, even on the last step
            return ts.termination(self._get_observation(), reward=reward)
        else:
            return ts.transition(self._get_observation(), reward=reward, discount=1.0)
        

class CryptoTradingEnvironment(py_environment.PyEnvironment):
    """
    A TF-Agents environment that now includes portfolio management.
    - Tracks cash and asset holdings.
    - Overrides impossible actions (e.g., selling what you don't own).
    - Calculates reward based on the change in total portfolio value.
    """
    def __init__(self, observation_df: pd.DataFrame, prices_df: pd.DataFrame, symbols: list[str], seed_fund: float = 100.0, trade_size: float = 20.0, invalid_action_penalty: float = -1.0):
        super().__init__()

        # --- Store pre-calculated data ---
        self._obs_data = observation_df
        self._price_data = prices_df
        self._symbols = symbols
        self._num_cryptos = len(symbols)
        
        # --- Portfolio State ---
        self._seed_fund = seed_fund
        self._trade_size = trade_size
        self._cash_balance = seed_fund
        self._balance_history = [seed_fund]
        self._trade_history = []
        self._invalid_action_penalty = invalid_action_penalty
        # Use a pandas Series for easy lookups and calculations
        self._asset_holdings = pd.Series(0.0, index=self._symbols)
        
        # --- Define Action and Observation Specs ---
        num_actions = self._num_cryptos * 3  # BUY, HOLD, SELL for each crypto
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action'
        )

        # --- Update Observation Spec to include portfolio state ---
        market_features_size = len(self._obs_data.columns)
        # Add one feature for each crypto holding, plus one for the cash balance
        portfolio_features_size = self._num_cryptos + 1
        total_observation_size = market_features_size + portfolio_features_size
    
        self._observation_spec = array_spec.ArraySpec(
            shape=(total_observation_size,), dtype=np.float32, name='context'
        )
        
        # --- Environment State ---
        self._current_step_index = 0
        self._episode_ended = False

    @property
    def current_step(self):
        return self._current_step_index

    @property
    def symbols(self):
        return self._symbols

    @property
    def price_data(self):
        return self._price_data

    @property
    def balance_history(self):
        return np.array(self._balance_history)

    @property
    def trade_history(self):
        return np.array(self._trade_history)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _get_observation(self) -> np.ndarray:
        """
        Returns the observation, which is now a combination of market
        features and the agent's own portfolio state.
        """
        # 1. Get market features (same as before)
        market_features = self._obs_data.iloc[self._current_step_index].values
        
        # 2. Get portfolio features
        # Normalize cash by initial funds to keep it in a reasonable range (0-1+)
        normalized_cash = self._cash_balance / self._seed_fund
        
        # We need to normalize asset holdings as well. A simple way is to get their
        # current value in USD and divide by the total portfolio value.
        current_prices = self._price_data.iloc[self._current_step_index]
        total_value = self._calculate_portfolio_value(self._current_step_index)
        
        # Avoid division by zero at the very start
        if total_value == 0:
            normalized_holdings = self._asset_holdings.values
        else:
            normalized_holdings = (self._asset_holdings * current_prices).values / total_value
    
        portfolio_features = np.append(normalized_holdings, normalized_cash)
    
        # 3. Concatenate and return
        full_observation = np.concatenate([market_features, portfolio_features])
        return full_observation.astype(np.float32)

    def _calculate_portfolio_value(self, step_index: int) -> float:
        """Helper function to calculate the total value of the portfolio."""
        current_prices = self._price_data.iloc[step_index]
        asset_values = self._asset_holdings * current_prices
        return self._cash_balance + asset_values.sum()

    def _reset(self) -> ts.TimeStep:
        """Resets the environment, including the portfolio."""
        self._current_step_index = 0
        self._episode_ended = False
        self._cash_balance = self._seed_fund
        self._asset_holdings = pd.Series(0.0, index=self._symbols)
        return ts.restart(self._get_observation())

    def _step(self, action: int) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()

        # --- 1. Get current state and decode action ---
        current_prices = self._price_data.iloc[self._current_step_index]
        portfolio_value_before = self._calculate_portfolio_value(self._current_step_index)
        
        crypto_index = action // 3
        trade_type_idx = action % 3  # 0: BUY, 1: HOLD, 2: SELL
        symbol_to_trade = self._symbols[crypto_index]

        # --- 2. Check if the action is valid and override if not ---
        is_valid_action = True
        if trade_type_idx == 0 and self._cash_balance < self._trade_size:
            # Not enough cash to buy, override to HOLD
            trade_type_idx = 1
            is_valid_action = False
        elif trade_type_idx == 2 and self._asset_holdings[symbol_to_trade] <= 0:
            # No assets to sell, override to HOLD
            trade_type_idx = 1
            is_valid_action = False

        # --- 3. Execute the (potentially overridden) trade ---
        if trade_type_idx == 0:  # BUY
            amount_to_buy = self._trade_size / current_prices[symbol_to_trade]
            self._asset_holdings[symbol_to_trade] += amount_to_buy
            self._cash_balance -= self._trade_size
        elif trade_type_idx == 2:  # SELL (sell all holdings of this asset)
            amount_to_sell = self._asset_holdings[symbol_to_trade]
            self._cash_balance += amount_to_sell * current_prices[symbol_to_trade]
            self._asset_holdings[symbol_to_trade] = 0.0
        # For HOLD (trade_type_idx == 1), do nothing.

        # --- 4. Advance time and calculate the new portfolio value ---
        self._current_step_index += 1
        portfolio_value_after = self._calculate_portfolio_value(self._current_step_index)

        # --- 5. Update balance history and trade history ---
        self._balance_history.append(portfolio_value_after)
        if not is_valid_action or trade_type_idx == 1:
            self._trade_history.append((symbol_to_trade, int(is_valid_action), 1, 0))
        else:
            amount_to_trade = amount_to_buy if trade_type_idx == 0 else amount_to_sell
            self._trade_history.append((symbol_to_trade, int(is_valid_action), trade_type_idx, amount_to_trade))
        
        # --- 5. NEW: Calculate reward based on portfolio value change ---
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before

        if not is_valid_action:
            reward += self._invalid_action_penalty

        # --- 6. Check for end of episode ---
        if self._current_step_index >= len(self._obs_data) - 1:
            self._episode_ended = True

        # --- 7. Return TimeStep ---
        if self._episode_ended:
            return ts.termination(self._get_observation(), reward=reward)
        else:
            return ts.transition(self._get_observation(), reward=reward, discount=1.0)