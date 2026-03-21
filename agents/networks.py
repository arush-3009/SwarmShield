"""
SwarmShield Neural Networks
============================

Two neural networks -> actor and critic

Actor (Policy Network):
    Takes observation (71 values) -> outputs action logits (11 values).
    No softmax inside the network. Softmax is applied externally when
    sampling actions or computing log-probabilities.

Critic (Value Network):
    Takes observation (71 values) -> outputs one number (state value).
    No activation on the output. The value can be any real number
    (positive or negative), representing how good this state is.

Both use two hidden layers with ReLU activations.
Architecture: 71 -> 128 -> 64 -> output
"""

import torch
import torch.nn as nn

from env.config import (
    OBSERVATION_SIZE,
    NUM_ACTIONS,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
)


class Actor(nn.Module):
    """
    Policy network. Takes an observation and outputs raw logits
    for each possible action. The training code applies softmax
    to get probabilities and samples from that distribution.

    Input:  observation vector (shape: OBSERVATION_SIZE = 71)
    Output: action logits (shape: NUM_ACTIONS = 11)
    """

    def __init__(self):
        super().__init__()

        # Layer 1: observation -> first hidden layer
        self.layer1 = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE_1)
        self.relu1 = nn.ReLU()

        # Layer 2: first hidden -> second hidden
        self.layer2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.relu2 = nn.ReLU()

        # Output layer: second hidden -> action logits
        # No activation here. Raw logits. Softmax applied externally.
        self.output_layer = nn.Linear(HIDDEN_SIZE_2, NUM_ACTIONS)

    def forward(self, obs):
        """
        Forward pass.

        obs: tensor of shape (batch_size, OBSERVATION_SIZE) or (OBSERVATION_SIZE,)
        returns: tensor of shape (batch_size, NUM_ACTIONS) — raw logits
        """
        x = self.layer1(obs)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.output_layer(x)
        return logits


class Critic(nn.Module):
    """
    Value network. Takes an observation and outputs a single number:
    the estimated value of being in this state.

    Input:  observation vector (shape: OBSERVATION_SIZE = 71)
    Output: state value (shape: 1)
    """

    def __init__(self):
        super().__init__()

        # Layer 1: observation -> first hidden layer
        self.layer1 = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE_1)
        self.relu1 = nn.ReLU()

        # Layer 2: first hidden -> second hidden
        self.layer2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.relu2 = nn.ReLU()

        # Output layer: second hidden -> single value
        # No activation. Value can be any real number.
        self.output_layer = nn.Linear(HIDDEN_SIZE_2, 1)

    def forward(self, obs):
        """
        Forward pass.

        obs: tensor of shape (batch_size, OBSERVATION_SIZE) or (OBSERVATION_SIZE,)
        returns: tensor of shape (batch_size, 1) — state value
        """
        x = self.layer1(obs)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        value = self.output_layer(x)
        return value