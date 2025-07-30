import tensorflow as tf
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent

from tf_agents.networks import encoding_network
from dataclasses import dataclass

@dataclass
class AgentConfig:
    alpha: float
    gamma: float
    fc_layer_params: tuple|None = None
    learning_rate: float|None = None
    encoding_dim: int|None = None

def create_agent(observation_spec, time_step_spec, action_spec, hp: AgentConfig, type="neural_linucb"):

    if type == "neural_linucb":

        encoder = encoding_network.EncodingNetwork(
            input_tensor_spec=observation_spec,
            fc_layer_params=hp.fc_layer_params,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        agent = neural_linucb_agent.NeuralLinUCBAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            encoding_network=encoder,
            encoding_dim=hp.encoding_dim,
            encoding_network_num_train_steps=1,
            optimizer=optimizer,
            alpha=hp.alpha,
            gamma=hp.gamma,
            dtype=tf.float32
        )

    elif type == "linucb":

        agent = lin_ucb_agent.LinearUCBAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            alpha=hp.alpha,
            gamma=hp.gamma,
            dtype=tf.float32
        )

    else:
        raise ValueError(f"Invalid agent type: {type}")

    return agent