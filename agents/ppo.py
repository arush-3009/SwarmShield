"""
SwarmShield PPO Agent
=====================

One independent PPO agent.

Each defender in IPPO gets:
- its own Actor
- its own Critic
- its own optimizers
- its own rollout buffer

This class assumes the environment returns:
- obs: numpy array of shape (OBSERVATION_SIZE,)
- reward: float
- done: bool
where done should usually be:
    terminated OR truncated

Main pieces:
1. Experience buffer
2. GAE advantage computation
3. PPO clipped policy update
4. Critic regression update
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.networks import Actor, Critic
from env.config import (
    OBSERVATION_SIZE,
    NUM_ACTIONS,
    PPO_BATCH_SIZE,
    PPO_CLIP_EPSILON,
    PPO_ENTROPY_COEFF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE_ACTOR,
    PPO_LEARNING_RATE_CRITIC,
)


class PPOAgent:
    """
    One PPO agent with its own actor, critic, and rollout buffer.
    """

    def __init__(self, device):
        self.device = device

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=PPO_LEARNING_RATE_ACTOR,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=PPO_LEARNING_RATE_CRITIC,
        )

        self.clear_buffer()

    # -------------------------------------------------------------------------
    # Buffer management
    # -------------------------------------------------------------------------

    def clear_buffer(self) -> None:
        self.buffer_obs: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_log_probs: List[float] = []
        self.buffer_rewards: List[float] = []
        self.buffer_dones: List[bool] = []
        self.buffer_values: List[float] = []

    def buffer_size(self) -> int:
        return len(self.buffer_obs)

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def _obs_to_tensor(self, obs_numpy) -> torch.Tensor:
        obs_array = np.asarray(obs_numpy, dtype=np.float32)
        if obs_array.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"Observation must have shape ({OBSERVATION_SIZE},), got {obs_array.shape}"
            )
        return torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)

    def select_action(self, obs_numpy):
        """
        Sample one action from the current policy.

        Returns:
            action: int
            log_prob: float
            value: float
        """
        obs_tensor = self._obs_to_tensor(obs_numpy)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(obs_tensor).squeeze(-1)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def select_action_deterministic(self, obs_numpy) -> int:
        """
        Greedy action selection, useful for evaluation.
        """
        obs_tensor = self._obs_to_tensor(obs_numpy)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            action = torch.argmax(logits, dim=-1)

        return int(action.item())

    def get_value(self, obs_numpy) -> float:
        """
        Critic value estimate for one observation.
        Used for GAE bootstrapping after rollout collection.
        """
        obs_tensor = self._obs_to_tensor(obs_numpy)

        with torch.no_grad():
            value = self.critic(obs_tensor).squeeze(-1)

        return float(value.item())

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    def store_transition(self, obs, action, log_prob, reward, done, value) -> None:
        self.buffer_obs.append(np.asarray(obs, dtype=np.float32))
        self.buffer_actions.append(int(action))
        self.buffer_log_probs.append(float(log_prob))
        self.buffer_rewards.append(float(reward))
        self.buffer_dones.append(bool(done))
        self.buffer_values.append(float(value))

    # -------------------------------------------------------------------------
    # GAE
    # -------------------------------------------------------------------------

    def compute_gae(self, last_value: float):
        """
        Compute GAE advantages and returns.

        last_value:
            critic estimate of the state AFTER the last collected transition.
            For a terminal/truncated rollout end, the training loop should pass
            0.0 or pass the real value and rely on done=True masking.
        """
        n = len(self.buffer_rewards)

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = float(last_value)

        for t in range(n - 1, -1, -1):
            done = self.buffer_dones[t]
            mask = 0.0 if done else 1.0

            delta = (
                self.buffer_rewards[t]
                + PPO_GAMMA * next_value * mask
                - self.buffer_values[t]
            )

            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * mask * gae

            advantages[t] = gae
            returns[t] = gae + self.buffer_values[t]

            next_value = self.buffer_values[t]

        return advantages, returns

    # -------------------------------------------------------------------------
    # PPO update
    # -------------------------------------------------------------------------

    def update(self, last_value: float) -> Dict[str, float]:
        """
        Run one PPO update over the current buffer.

        Returns:
            dict with training stats
        """
        if self.buffer_size() == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "buffer_size": 0,
            }

        advantages, returns = self.compute_gae(last_value)

        obs_tensor = torch.as_tensor(
            np.asarray(self.buffer_obs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions_tensor = torch.as_tensor(
            np.asarray(self.buffer_actions, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        old_log_probs_tensor = torch.as_tensor(
            np.asarray(self.buffer_log_probs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )
        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        # Advantage normalization
        if len(advantages_tensor) > 1:
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantages_tensor.std(unbiased=False) + 1e-8)

        batch_size_total = self.buffer_size()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(PPO_EPOCHS):
            indices = np.arange(batch_size_total)
            np.random.shuffle(indices)

            for start in range(0, batch_size_total, PPO_BATCH_SIZE):
                end = min(start + PPO_BATCH_SIZE, batch_size_total)
                batch_indices = indices[start:end]

                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # -----------------------------
                # Actor update
                # -----------------------------
                logits = self.actor(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surrogate_1 = ratio * batch_advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - PPO_CLIP_EPSILON,
                    1.0 + PPO_CLIP_EPSILON,
                ) * batch_advantages

                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss = actor_loss - PPO_ENTROPY_COEFF * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # -----------------------------
                # Critic update
                # -----------------------------
                predicted_values = self.critic(batch_obs).squeeze(-1)
                critic_loss = F.mse_loss(predicted_values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1

        stats = {
            "actor_loss": total_actor_loss / max(num_updates, 1),
            "critic_loss": total_critic_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "buffer_size": self.buffer_size(),
        }

        self.clear_buffer()
        return stats

    # -------------------------------------------------------------------------
    # Save / load
    # -------------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])