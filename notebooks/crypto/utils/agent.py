import tensorflow as tf
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.networks import encoding_network
from keras_tuner.engine.trial import Trial
from dataclasses import dataclass

@dataclass
class AgentConfig:
    alpha: float
    gamma: float
    fc_layer_params: tuple|None = None
    learning_rate: float|None = None
    encoding_dim: int|None = None

def create_agent(specs: dict, config: AgentConfig, type="neural_linucb"):

    if type == "neural_linucb":

        encoder = encoding_network.EncodingNetwork(
            input_tensor_spec=specs['observation'],
            fc_layer_params=config.fc_layer_params,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        agent = neural_linucb_agent.NeuralLinUCBAgent(
            time_step_spec=specs['time_step'],
            action_spec=specs['action'],
            encoding_network=encoder,
            encoding_dim=config.encoding_dim,
            encoding_network_num_train_steps=1,
            optimizer=optimizer,
            alpha=config.alpha,
            gamma=config.gamma,
            dtype=tf.float32
        )

    elif type == "linucb":

        agent = lin_ucb_agent.LinearUCBAgent(
            time_step_spec=specs['time_step'],
            action_spec=specs['action'],
            alpha=config.alpha,
            gamma=config.gamma,
            dtype=tf.float32
        )

    else:
        raise ValueError(f"Invalid agent type: {type}")

    return agent


def create_agent_for_tuner(specs: dict, trial: Trial, hp_values: dict):
    hp = trial.hyperparameters
    config = AgentConfig(
        alpha=hp.Float('alpha', **hp_values['alpha']),
        gamma=hp.Float('gamma', **hp_values['gamma']),
        fc_layer_params=hp.Int('fc_layer_params', **hp_values['fc_layer_params']),
        learning_rate=hp.Float('learning_rate', **hp_values['learning_rate']),
        encoding_dim=hp.Int('encoding_dim', **hp_values['encoding_dim'])
    )
    type = hp.Choice('type', **hp_values['type'])
    agent = create_agent(specs, config, type)
    agent.initialize()
    return agent
