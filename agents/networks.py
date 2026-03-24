"""
SwarmShield Neural Networks
===========================

Two simple MLPs:

1. Actor (policy network)
   Input : observation vector of length OBSERVATION_SIZE
   Output: raw action logits of length NUM_ACTIONS

2. Critic (value network)
   Input : observation vector of length OBSERVATION_SIZE
   Output: one scalar state-value estimate

Current finalized environment:
- OBSERVATION_SIZE = 77
- NUM_ACTIONS = 22

Architecture:
OBS -> 128 -> 64 -> output
with ReLU activations.

We do NOT apply softmax inside the Actor.
The PPO code will build a Categorical distribution from logits.
"""

import torch
import torch.nn as nn

from env.config import (
    OBSERVATION_SIZE,
    NUM_ACTIONS,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
)


def _init_linear_layer(layer: nn.Linear, std: float = 1.0) -> None:
    """
    Simple orthogonal initialization for better PPO stability.
    """
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """
    Policy network.

    Input:
        shape (..., OBSERVATION_SIZE)

    Output:
        shape (..., NUM_ACTIONS)
        raw logits, not probabilities
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE_1)
        self.layer2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.output_layer = nn.Linear(HIDDEN_SIZE_2, NUM_ACTIONS)

        self.relu = nn.ReLU()

        _init_linear_layer(self.layer1, std=1.0)
        _init_linear_layer(self.layer2, std=1.0)
        _init_linear_layer(self.output_layer, std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.layer1(obs)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        logits = self.output_layer(x)
        return logits


class Critic(nn.Module):
    """
    Value network.

    Input:
        shape (..., OBSERVATION_SIZE)

    Output:
        shape (..., 1)
        scalar state value
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE_1)
        self.layer2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.output_layer = nn.Linear(HIDDEN_SIZE_2, 1)

        self.relu = nn.ReLU()

        _init_linear_layer(self.layer1, std=1.0)
        _init_linear_layer(self.layer2, std=1.0)
        _init_linear_layer(self.output_layer, std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.layer1(obs)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        value = self.output_layer(x)
        return value