import numpy as np
import pandas as pd
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class CryptoTradingEnvironment(py_environment.PyEnvironment):
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
        """
        PERFORMANCE WIN: This is now a simple, ultra-fast lookup.
        """
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
            return ts.termination(self._get_observation(), reward)
        else:
            return ts.transition(self._get_observation(), reward=reward, discount=1.0)