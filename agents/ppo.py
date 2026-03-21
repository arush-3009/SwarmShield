"""
SwarmShield PPO Agent
======================

One independent PPO agent. IPPO just runs N of these in parallel.

1. EXPERIENCE BUFFER
   Stores (obs, action, log_prob, reward, done, value) tuples
   collected during rollout. Cleared after each update.

2. GAE (Generalized Advantage Estimation)
   Computes advantages using the critic's value estimates.
   GAE balances bias vs variance using lambda parameter.

3. CLIPPED SURROGATE UPDATE
   The PPO objective with clipping. Multiple epochs over the
   same batch. Separate actor and critic optimizer steps.
   Entropy bonus for exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.networks import Actor, Critic
from env.config import (
    PPO_LEARNING_RATE_ACTOR,
    PPO_LEARNING_RATE_CRITIC,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_EPSILON,
    PPO_ENTROPY_COEFF,
    PPO_EPOCHS,
    PPO_BATCH_SIZE,
    OBSERVATION_SIZE,
    NUM_ACTIONS,
)


class PPOAgent:
    """
    One PPO agent with its own actor, critic, and experience buffer.

    In IPPO, each of the 3 agents is a separate PPOAgent instance.
    They don't share weights or gradients. Each learns independently
    from its own observations and rewards.
    """

    def __init__(self, device):
        """
        device: torch device ('cpu', 'mps', or 'cuda')
        """
        self.device = device

        # Create actor and critic networks
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=PPO_LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=PPO_LEARNING_RATE_CRITIC
        )

        # Experience buffer -> gets filled during rollout
        # and cleared after each PPO update
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []

    def select_action(self, obs_numpy):
        """
        Given a numpy observation, select an action using the current policy.

        This is called every timestep during experience collection.
        The actor outputs logits, we apply softmax to get probabilities,
        sample an action from that distribution, and record the log-prob
        (needed later for the PPO ratio computation).

        obs_numpy: numpy array of shape (OBSERVATION_SIZE,)

        Returns:
            action: int, the chosen action (0-10)
            log_prob: float, log probability of the chosen action
            value: float, critic's estimate of state value
        """
        
        obs_tensor = torch.FloatTensor(obs_numpy).to(self.device)

        
        with torch.no_grad():
            
            logits = self.actor(obs_tensor)

            
            probs = torch.softmax(logits, dim=-1)

            # Sample an action from the probability distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            
            log_prob = dist.log_prob(action)

            # state value from critic
            value = self.critic(obs_tensor)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, obs, action, log_prob, reward, done, value):
        """
        Store one timestep of experience in the buffer.

        Called after each environment step during rollout.
        """
        self.buffer_obs.append(obs)
        self.buffer_actions.append(action)
        self.buffer_log_probs.append(log_prob)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)
        self.buffer_values.append(value)

    def compute_gae(self, last_value):
        """
        Compute Generalized Advantage Estimation (GAE) for the collected buffer.

        last_value: float, the critic's value estimate for the state
                    AFTER the last collected timestep. Needed for
                    bootstrapping the final advantage.

        Returns:
            advantages: numpy array of shape (buffer_size,)
            returns: numpy array of shape (buffer_size,)
        """
        buffer_size = len(self.buffer_rewards)
        advantages = np.zeros(buffer_size, dtype=np.float32)
        returns = np.zeros(buffer_size, dtype=np.float32)

        # GAE computation goes backwards through the buffer
        gae = 0.0
        next_value = last_value

        for t in range(buffer_size - 1, -1, -1):
            # If episode ended at this step, next_value should be 0
            # (no future rewards after terminal state)
            if self.buffer_dones[t]:
                next_value = 0.0
                gae = 0.0

            # TD error (delta)
            delta = self.buffer_rewards[t] + PPO_GAMMA * next_value - self.buffer_values[t]
            

            # GAE accumulation
            gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * gae

            advantages[t] = gae
            returns[t] = gae + self.buffer_values[t]

            # Move backwards
            next_value = self.buffer_values[t]

        return advantages, returns

    def update(self, last_value):
        """
        Run the PPO clipped surrogate update.

        1. Compute GAE advantages and returns from the buffer
        2. For K epochs, shuffle the buffer into minibatches
        3. For each minibatch, compute the clipped surrogate loss
           and update actor and critic

        last_value: float, critic's value estimate after the last
                    collected timestep (for GAE bootstrapping)

        Returns:
            dict with training stats (actor_loss, critic_loss, entropy)
        """
       
        advantages, returns = self.compute_gae(last_value)

        
        obs_tensor = torch.FloatTensor(np.array(self.buffer_obs)).to(self.device)
        actions_tensor = torch.LongTensor(self.buffer_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.buffer_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (zero mean, unit variance)
        
        if len(advantages_tensor) > 1:
            adv_mean = advantages_tensor.mean()
            adv_std = advantages_tensor.std() + 1e-8
            advantages_tensor = (advantages_tensor - adv_mean) / adv_std

       
        buffer_size = len(self.buffer_obs)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(PPO_EPOCHS):
            # random indices for minibatch sampling
            indices = np.arange(buffer_size)
            np.random.shuffle(indices)

            # Process minibatches
            for start in range(0, buffer_size, PPO_BATCH_SIZE):
                end = start + PPO_BATCH_SIZE
                if end > buffer_size:
                    end = buffer_size

                # Get minibatch indices
                batch_indices = indices[start:end]

                # Extract minibatch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # === ACTOR UPDATE ===

                # Get current policy's log probs for the batch actions
                logits = self.actor(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                
                surr1 = ratio * batch_advantages.detach()
    
                surr2 = (torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * 
                         batch_advantages.detach())
                
                # Take the minimum (pessimistic bound)
                actor_loss = -torch.min(surr1, surr2).mean()

                
                actor_loss = actor_loss - PPO_ENTROPY_COEFF * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # === CRITIC UPDATE ===

                
                values = self.critic(batch_obs).squeeze(-1)
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                # Track stats
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Clear buffer after update
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.buffer_values = []

        # Return average stats
        stats = {
            'actor_loss': total_actor_loss / max(num_updates, 1),
            'critic_loss': total_critic_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }
        return stats

    def get_value(self, obs_numpy):
        """
        Get the critic's value estimate for a single observation.
        Used for bootstrapping at the end of a rollout.
        """
        obs_tensor = torch.FloatTensor(obs_numpy).to(self.device)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value.item()

    def save(self, filepath):
        """Save actor and critic weights to a file."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load actor and critic weights from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])