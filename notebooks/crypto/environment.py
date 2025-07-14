import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.replay_buffers import tf_uniform_replay_buffer

class CryptoTradingEnvironment(py_environment.PyEnvironment):
    """
    A TF-Agents environment for a crypto trading multi-armed bandit.
    Manages state using an internal replay buffer.
    """
    def __init__(self, data, symbols, context_len):
        super().__init__()
        self._data = data
        self._context_len = context_len
        self._symbols = symbols
        self._num_cryptos = len(self._symbols)
        
        self._minimal_obs_size = self._num_cryptos * 2 
        observation_size = self._minimal_obs_size * self._context_len

        # Define specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._num_cryptos * 2 - 1, name='action'
        )
        self._observation_spec = array_spec.ArraySpec(
            shape=(observation_size,), dtype=np.float32, name='context'
        )
        
        # Internal replay buffer to manage state history
        data_spec = tensor_spec.TensorSpec([self._minimal_obs_size], dtype=tf.float32, name='minimal_observation')
        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=data_spec, batch_size=1, max_length=self._context_len + 5
        )
        
        # Environment state
        self._current_step_index = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_minimal_observation(self, index):
        obs_slice = []
        for symbol in self._symbols:
            obs_slice.append(self._data.iloc[index][f'{symbol}_close_return'])
            obs_slice.append(self._data.iloc[index][f'{symbol}_volume_return'])
        return np.array(obs_slice, dtype=np.float32)

    def _observe(self):
        num_items = self._replay_buffer.num_frames()
        if num_items == 0:
            return np.zeros(self._observation_spec.shape, dtype=np.float32)

        dataset = self._replay_buffer.as_dataset(single_deterministic_pass=True)
        batched_items = next(iter(dataset.batch(num_items)))
        
        all_items_tensor = batched_items[0]
        
        context = tf.reshape(all_items_tensor[-self._context_len:], [-1])
        return context.numpy()

    def _reset(self):
        self._replay_buffer.clear()
        self._episode_ended = False
        self._current_step_index = self._context_len 
        for i in range(self._current_step_index):
            self._replay_buffer.add_batch(tf.expand_dims(self._get_minimal_observation(i), 0))
        return ts.restart(self._observe())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        crypto_index = action // 2
        is_buy_action = action % 2 == 0
        column_name = f'{self._symbols[crypto_index]}_close'
        current_price = self._data.iloc[self._current_step_index][column_name]
        next_price = self._data.iloc[self._current_step_index + 1][column_name]
        reward = ((next_price - current_price) / current_price) if is_buy_action else ((current_price - next_price) / current_price)
        
        self._replay_buffer.add_batch(tf.expand_dims(self._get_minimal_observation(self._current_step_index), 0))
        self._current_step_index += 1
        
        if self._current_step_index >= len(self._data) - 2:
            self._episode_ended = True
        
        observation = self._observe()
        return ts.termination(observation, reward) if self._episode_ended else ts.transition(observation, reward)