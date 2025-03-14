import importlib
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from bqr import nn
from bqr.data.fetch_data import *
from bqr.sim import env


class Simulator:

    def __init__(
        self,
        policy_name: str,
        feature_extractor: Union[ActorCriticPolicy, None],
        hidden_size: int = 64,
        num_episodes: int = 100,
        ticker: str = "ETHUSDT",
        start_date: str = "20240101",
        end_date: str = "20240630",
        verbose: int = 1,
        total_training_steps: int = 10000,
    ):
        self.policy_name = policy_name
        self.feature_extractor = feature_extractor
        self.hidden_size = hidden_size
        self.num_episodes = num_episodes
        self.total_training_steps = total_training_steps
        data = get_custom_data(ticker, start_date, end_date)
        self.verbose = verbose
        self.env = env.TradingSimulator(data)
        self.model = self._fetch_policy()

    def _fetch_policy(self):
        if self.policy_name == "PPO":
            policy = (
                importlib.import_module(f"nn.{self.feature_extractor}")
                if self.feature_extractor
                else nn.CustomTradingPPOPolicy
            )
            policy = "MlpPolicy"
            # policy_kwargs = dict(hidden_size=self.hidden_size)
            model = PPO(
                policy,
                self.env,
                verbose=self.verbose,  # policy_kwargs=policy_kwargs
            )

        else:
            raise NotImplementedError
        check_env(self.env)
        return model

    def _test_model(self):

        # Run 10 steps to see the model in action
        for episode in range(self.num_episodes):
            done = False
            print(f"Episode : {episode+1}")
            obs, _ = self.env.reset()
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _, info = self.env.step(action)
                self.env.render()
            self.env.save_history()

    def simulate(self):
        self.model.learn(self.total_training_steps)
        self._test_model()
