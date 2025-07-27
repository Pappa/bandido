import numpy as np

class ProgressObserver:
    def __init__(self, total, interval=25):
        self.counter = 0
        self.total = int(total)
        self.interval = interval
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % self.interval == 0:
            print(f"\r{self.counter}/{self.total}", end="")

class MetricsObserver:
    def __init__(self, env):
        self._env = env
        self._rewards = []
        self._regrets = []
        self._optimal_rewards = []
        self._balance_history = [self._env.current_balance]
        self._asset_value_history = [0]
        self._trade_history = []
        self._optimal_trade_history = []

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            reward = trajectory.reward.numpy()[0]
            self._rewards.append(reward)

            last_trade = self._env.last_trade
            current_balance = self._env.current_balance
            current_assets = self._env.current_assets

            self._trade_history.append(last_trade)
            self._balance_history.append(current_balance)
            self._asset_value_history.append(current_assets)
            
            optimal_reward = self._env.optimal_reward_oracle(trajectory)
            regret = optimal_reward - reward
            
            self._optimal_rewards.append(optimal_reward)
            self._regrets.append(regret)

            optimal_trade = self._env.last_optimal_trade

            self._optimal_trade_history.append(optimal_trade)

    @property
    def balance_history(self):
        return np.array(self._balance_history).astype(np.float32)

    @property
    def asset_value_history(self):
        return np.array(self._asset_value_history).astype(np.float32)

    @property
    def portfolio_value_history(self):
        return (np.array(self._balance_history) + np.array(self._asset_value_history)).astype(np.float32)

    @property
    def trade_history(self):
        return np.array(self._trade_history)

    @property
    def optimal_trade_history(self):
        return np.array(self._optimal_trade_history)

    def cum_rewards(self):
        cumulative_reward = np.cumsum(self._rewards)
        cumulative_optimal_reward = np.cumsum(self._optimal_rewards)
        return cumulative_reward, cumulative_optimal_reward

    def cum_averages(self):
        steps = np.arange(len(self._rewards)) + 1
        cumulative_avg_reward = np.cumsum(self._rewards) / steps
        cumulative_avg_optimal_reward = np.cumsum(self._optimal_rewards) / steps
        cumulative_avg_regret = np.cumsum(self._regrets) / steps
        return cumulative_avg_reward, cumulative_avg_optimal_reward, cumulative_avg_regret