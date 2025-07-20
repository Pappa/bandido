import numpy as np
import pandas as pd
import tensorflow as tf
from enum import Enum

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class TradeType(Enum):
    BUY  = 0
    HOLD = 1
    SELL = 2
    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, TradeType):
            return self.value == other.value
        else:
            return False

class BaseCryptoTradingEnvironment(py_environment.PyEnvironment):
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
        self._num_market_features = self._obs_data.shape[1]

        # --- Define Action and Observation Specs ---
        self._init_action_spec()
        self._init_observation_spec()

        # --- Environment State ---
        self._current_step_index = 0
        self._episode_ended = False

    def _init_action_spec(self):
        """
        Initializes the action spec.
        """
        num_actions = self._num_cryptos * len(TradeType)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1, name='action'
        )

    def _init_observation_spec(self):
        """
        Initializes the observation spec.
        The observation shape is the number of columns in the observation_df
        """
        self._observation_spec = array_spec.ArraySpec(
            shape=(self._num_market_features,), dtype=np.float32, name='context'
        )

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
        crypto_index = action // len(TradeType)   # 0: BTC, 1: ETH, 2: SOL, etc.
        trade_type_idx = action % len(TradeType)  # 0: BUY, 1: HOLD, 2: SELL

        # Calculate the reward
        reward = 0.0
        if trade_type_idx != TradeType.HOLD:  # No reward for HOLD
            symbol = self._symbols[crypto_index]

            # Use the dedicated prices DataFrame for a fast lookup
            current_price = self._price_data.iloc[self._current_step_index][symbol]
            next_price = self._price_data.iloc[self._current_step_index + 1][symbol]

            if trade_type_idx == TradeType.BUY:  # BUY
                reward = (next_price - current_price) / current_price
            elif trade_type_idx == TradeType.SELL:  # SELL
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


class CryptoTradingEnvironment(BaseCryptoTradingEnvironment):
    """
    A TF-Agents environment that now includes portfolio management.
    - Tracks cash and asset holdings.
    - Overrides impossible actions (e.g., selling what you don't own).
    - Calculates reward based on the change in total portfolio value.
    """
    def __init__(self, observation_df: pd.DataFrame, prices_df: pd.DataFrame, symbols: list[str], seed_fund: float = 100.0, trade_size: float = 20.0, trade_fee: float = 0.005, invalid_action_penalty: float = -0.001):

        super().__init__(observation_df, prices_df, symbols)

        # --- Portfolio State ---
        self._seed_fund = seed_fund
        self._trade_size = trade_size
        self._trade_fee = trade_fee
        self._invalid_action_penalty = invalid_action_penalty
        self._cash_balance = seed_fund
        self._balance_history = [seed_fund]
        self._trade_history = []
        # Use a pandas Series for easy lookups and calculations
        self._asset_holdings = pd.Series(0.0, index=self._symbols)
        self._asset_value_history = [0]

    def _init_observation_spec(self):
        """
        Initializes the observation spec.
        The observation shape is the number of columns in the observation_df
        plus one feature for each crypto holding, plus one for the cash balance
        """
        num_portfolio_features = self._num_cryptos + 1
        observation_size = self._num_market_features + num_portfolio_features

        self._observation_spec = array_spec.ArraySpec(
            shape=(observation_size,), dtype=np.float32, name='context'
        )

    @property
    def balance_history(self):
        return np.array(self._balance_history)

    @property
    def asset_value_history(self):
        return np.array(self._asset_value_history)

    @property
    def portfolio_value_history(self):
        return np.array(self._balance_history) + np.array(self._asset_value_history)

    @property
    def trade_history(self):
        return np.array(self._trade_history)

    def _get_observation(self) -> np.ndarray:
        """
        Returns the observation, which is now a combination of market
        features and the agent's own portfolio state.
        """
        market_features = self._obs_data.iloc[self._current_step_index].values
        normalized_cash = self._cash_balance / self._seed_fund
        current_prices = self._price_data.iloc[self._current_step_index]
        total_value = self._calculate_asset_value(self._current_step_index) + self._cash_balance
        
        if total_value == 0:
            # Handle edge case at the beginning or if bankrupt
            normalized_holdings = np.zeros(self._num_cryptos)
        else:
            # Normalize each asset's value as a percentage of the total portfolio
            normalized_holdings = (self._asset_holdings * current_prices).values / total_value

        portfolio_features = np.append(normalized_holdings, normalized_cash)
        return np.concatenate([market_features, portfolio_features]).astype(np.float32)

    def _calculate_asset_value(self, step_index: int) -> float:
        current_prices = self._price_data.iloc[step_index]
        asset_values = self._asset_holdings * current_prices
        return asset_values.sum()
    
    
    def _is_valid_action(self, trade_type: TradeType, symbol: str) -> bool:
        if trade_type == TradeType.BUY and self._cash_balance < self._trade_size:
            return False
        elif trade_type == TradeType.SELL and self._asset_holdings[symbol] <= 0:
            return False
        return True
    

    def _calculate_pnl_reward(self, step_index: int, trade_type: TradeType, symbol: str) -> float:
        """
        Calculates the P&L reward for a given action at a specific step_index,
        accounting for trade fees. This is a simulation and does NOT change the
        environment's actual portfolio state.
        """
        # 1. No reward for HOLD
        if trade_type == TradeType.HOLD:
            return 0.0
        
        # 2. Get initial state for simulation
        current_prices = self._price_data.iloc[step_index]
        portfolio_value_before = self._calculate_asset_value(step_index) + self._cash_balance

        # 3. Simulate the trade to get the hypothetical next-step portfolio
        temp_cash = self._cash_balance
        temp_holdings = self._asset_holdings.copy()

        if trade_type == TradeType.BUY:
            fee = self._trade_size * self._trade_fee
            net_purchase_value = self._trade_size - fee
            amount_bought = net_purchase_value / current_prices[symbol]
            temp_holdings[symbol] += amount_bought
            temp_cash -= self._trade_size
            
        elif trade_type == TradeType.SELL:
            amount_to_sell = temp_holdings[symbol]
            if amount_to_sell > 0:
                total_sale_value = amount_to_sell * current_prices[symbol]
                fee = total_sale_value * self._trade_fee
                net_proceeds = total_sale_value - fee
                temp_cash += net_proceeds
                temp_holdings[symbol] = 0.0

        # 4. Calculate the portfolio value at the *next* time step
        next_prices = self._price_data.iloc[step_index + 1]
        simulated_asset_value_after = (temp_holdings * next_prices).sum()
        simulated_portfolio_value_after = simulated_asset_value_after + temp_cash

        # 5. Return the percentage change (the reward)
        # Avoid division by zero if portfolio value is 0
        if portfolio_value_before == 0:
            return 0.0
        reward = (simulated_portfolio_value_after - portfolio_value_before) / portfolio_value_before
        return reward
    

    def _step(self, action: int) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()
        
        # --- 1. Decode action and check for validity ---
        crypto_index = action // len(TradeType)
        trade_type = TradeType(action % len(TradeType))
        symbol = self._symbols[crypto_index]
        
        is_valid_action = self._is_valid_action(trade_type, symbol)
        if not is_valid_action:
            trade_type = TradeType.HOLD

        # --- 2. Calculate the reward BEFORE changing the state ---
        pnl_reward = self._calculate_pnl_reward(self._current_step_index, trade_type, symbol)

        # --- 3. Execute the effective trade to update the REAL state ---
        current_prices = self._price_data.iloc[self._current_step_index]
        trade_history_amount = 0.0
        
        if trade_type == TradeType.BUY:
            fee = self._trade_size * self._trade_fee
            net_purchase = self._trade_size - fee
            amount_bought = net_purchase / current_prices[symbol]
            self._asset_holdings[symbol] += amount_bought
            self._cash_balance -= self._trade_size
            trade_history_amount = self._trade_size
        elif trade_type == TradeType.SELL:
            amount_sold = self._asset_holdings[symbol]
            sale_value = amount_sold * current_prices[symbol]
            fee = sale_value * self._trade_fee
            self._cash_balance += sale_value - fee
            self._asset_holdings[symbol] = 0.0
            trade_history_amount = sale_value

        # --- 4. Advance time and update history ---
        self._current_step_index += 1
        asset_value_after = self._calculate_asset_value(self._current_step_index)
        self._balance_history.append(self._cash_balance)
        self._asset_value_history.append(asset_value_after)
        self._trade_history.append((symbol, int(is_valid_action), trade_type, trade_history_amount))

        # --- 5. Finalize the reward with any penalties ---
        final_reward = pnl_reward
        if not is_valid_action:
            final_reward += self._invalid_action_penalty

        # --- 6. Check for end of episode and return ---
        if self._current_step_index >= len(self._obs_data) - 1:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(self._get_observation(), reward=final_reward)
        else:
            return ts.transition(self._get_observation(), reward=final_reward, discount=1.0)
        

    def optimal_reward_oracle(self, trajectory) -> np.float32:
        """
        Calculates the best possible reward for the current step by looking ahead.
        This "perfect foresight" oracle is used for calculating regret.
        """
        # We can't look one step into the future.
        if trajectory.is_last():
            return 0.0
        
        current_step = self._current_step_index
        all_possible_rewards = []
        num_actions = self.action_spec().maximum + 1
        
        for action in range(num_actions):
            crypto_index = action // len(TradeType)
            trade_type = TradeType(action % len(TradeType))
            symbol = self._symbols[crypto_index]
            
            if not self._is_valid_action(trade_type, symbol):
                reward = self._calculate_pnl_reward(current_step, TradeType.HOLD, symbol) + self._invalid_action_penalty
            else:
                reward = self._calculate_pnl_reward(current_step, trade_type, symbol)

            all_possible_rewards.append(reward)
        
        return np.max(all_possible_rewards).astype(np.float32)