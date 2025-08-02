from keras_tuner import Tuner
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver

class TradingAgentTuner(Tuner):
    def __init__(self, train_env, val_env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_env = train_env
        self._val_env = val_env

    def run_trial(self, trial, train_obs_df, train_prices_df, val_obs_df, val_prices_df, symbols, **kwargs):
        """
        Executes one training and evaluation trial for a set of hyperparameters.
        """        
        # --- 2. Setup Training Environment and Agent ---
        train_tf_env = tf_py_environment.TFPyEnvironment(self._train_env)

        train_specs = {
            'observation': self._train_env.observation_spec(),
            'time_step': self._train_env.time_step_spec(),
            'action': self._train_env.action_spec(),
        }

        hp_values = {
            'alpha': { 'min_value': 1.0, 'max_value': 2.0, 'step': 0.2 },
            'gamma': { 'min_value': 0.9, 'max_value': 1.0, 'step': 0.1 },
            'fc_layer_params': { 'choices': [(52, 26)] }, # TODO: don't hardcode this
            'learning_rate': { 'min_value': 0.00001, 'max_value': 0.01, 'step': 10, "sampling": "log" },
            'encoding_dim': { 'choices': [self._train_env.observation_spec().shape[0]] },
            'type': { 'values': ['neural_linucb', 'neural_ucb'] }
        }
        
        # Build the agent with the current trial's hyperparameters
        agent = self.hypermodel.build(train_specs, trial, hp_values)

        # --- 3. Run the Training Driver ---
        # (This is your standard training driver setup)
        num_train_steps = len(train_obs_df) - 1
        # ... (define train_step observer) ...
        train_driver = dynamic_step_driver.DynamicStepDriver(
            env=train_tf_env,
            policy=agent.policy,
            num_steps=num_train_steps,
            observers=[train_step] # No need for metrics here, just training
        )
        train_driver.run()

        # --- 4. Evaluate the Trained Agent on the Validation Set ---
        val_py_env = CryptoTradingEnvironment(
            observation_df=val_obs_df,
            prices_df=val_prices_df,
            symbols=symbols,
            # ... other env params ...
        )
        
        # Run a simple evaluation loop
        time_step = val_py_env.reset()
        while not time_step.is_last():
            batched_time_step = time_step._replace(observation=tf.expand_dims(time_step.observation, 0))
            action_step = agent.policy.action(batched_time_step)
            time_step = val_py_env.step(action_step.action.numpy()[0])
            
        # --- 5. Report the Final Portfolio Value as the Objective ---
        final_portfolio_value = val_py_env.portfolio_value_history[-1]
        
        # KerasTuner needs a dictionary of metrics to log
        # We tell it to maximize 'final_portfolio_value'
        self.oracle.update_trial(trial.trial_id, {'final_portfolio_value': final_portfolio_value})