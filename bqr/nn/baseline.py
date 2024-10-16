import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLP(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, hidden_size=64
    ):  # You can customize features_dim here
        super(CustomMLP, self).__init__(observation_space, hidden_size)
        # Define the neural network layers for feature extraction
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),  # Output size is features_dim
            nn.ReLU(),
        )

        # For the value function output
        self.value_layer = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        return self.network(observations)

    def forward_critic(self, features):
        # Compute the value from the extracted features
        return self.value_layer(features)


class CustomTradingPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        hidden_size,
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs
    ):
        super(CustomTradingPPOPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,  # Will be passed as None if not specified
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
        # Custom feature extractor
        self.mlp_extractor = CustomMLP(observation_space, hidden_size)

        # Policy and value networks
        self.action_net = nn.Linear(
            self.mlp_extractor.features_dim, action_space.shape[0]
        )  # Output the 2 actions
        self.value_net = nn.Linear(
            self.mlp_extractor.features_dim, 1
        )  # Output a single value

    def forward(self, obs, deterministic=False):
        # Extract features
        features = self.mlp_extractor(obs)

        # Action distribution
        mean_actions = self.action_net(features)

        # Value function output
        value = self.value_net(features)

        # Create a normal distribution for continuous actions
        dist = th.distributions.Normal(mean_actions, th.ones_like(mean_actions))

        # Sample or return deterministic actions
        actions = dist.mean if deterministic else dist.sample()

        # Log probability of the action
        log_prob = dist.log_prob(actions)

        # Sum the log_probs of both actions to match (1,) expected shape
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return actions, value, log_prob

    def _predict(self, obs, deterministic=False):
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions
