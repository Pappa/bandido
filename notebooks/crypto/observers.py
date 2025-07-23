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
    def __init__(self, oracle_fn):
        self._oracle_fn = oracle_fn
        self.rewards = []
        self.regrets = []
        self.optimal_rewards = []

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            reward = trajectory.reward.numpy()[0]
            optimal_reward = self._oracle_fn(trajectory)
            regret = optimal_reward - reward
            
            self.rewards.append(reward)
            self.regrets.append(regret)
            self.optimal_rewards.append(optimal_reward)

    def cum_rewards(self):
        cumulative_reward = np.cumsum(self.rewards)
        cumulative_optimal_reward = np.cumsum(self.optimal_rewards)
        return cumulative_reward, cumulative_optimal_reward

    def cum_averages(self):
        steps = np.arange(len(self.rewards)) + 1
        cumulative_avg_reward = np.cumsum(self.rewards) / steps
        cumulative_avg_optimal_reward = np.cumsum(self.optimal_rewards) / steps
        cumulative_avg_regret = np.cumsum(self.regrets) / steps
        return cumulative_avg_reward, cumulative_avg_optimal_reward, cumulative_avg_regret