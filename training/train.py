"""
SwarmShield Training Script
==============================

This is the main training loop. It does:

1. Create the environment and IPPO agents
2. Collect experience by running the environment for HORIZON steps
3. Run PPO update for each agent
4. Repeat for many iterations
5. Save checkpoints periodically
6. Print training stats
"""

import os
import time
import numpy as np
import torch

from env.swarmshield_env import SwarmShieldEnv
from agents.ippo import IPPO
from env.config import (
    PPO_HORIZON,
    NUM_AGENTS,
    MAX_TIMESTEPS,
)


def train():

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Total training timesteps. Start with 500k, increase if agents aren't
    # learning. With 200-step episodes and 3 agents, 500k timesteps is
    # roughly 800+ episodes.
    TOTAL_TIMESTEPS = 500_000

    # How often to print training stats (in PPO updates, not timesteps)
    LOG_INTERVAL = 1

    # How often to save checkpoints (in PPO updates)
    SAVE_INTERVAL = 10

    # Directory for saved model weights
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # =========================================================================
    # DEVICE SELECTION
    # =========================================================================
    # Use MPS (Apple Silicon GPU) if available, otherwise CPU.
    # CUDA for NVIDIA GPUs.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # =========================================================================
    # CREATE ENVIRONMENT AND AGENTS
    # =========================================================================
    env = SwarmShieldEnv()
    ippo = IPPO(device)

    print(f"Observation size: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Number of agents: {NUM_AGENTS}")
    print(f"Horizon (steps per update): {PPO_HORIZON}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Training starting...\n")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    total_steps = 0
    update_count = 0
    episode_count = 0
    start_time = time.time()

    # Episode tracking for logging
    episode_rewards = []        # Total reward per episode (summed across agents)
    episode_lengths = []        # Steps per episode
    episode_outcomes = []       # 'win', 'loss', or 'survive'
    current_episode_reward = 0.0
    current_episode_length = 0

    # Reset environment for first episode
    observations, infos = env.reset()

    while total_steps < TOTAL_TIMESTEPS:

        # =================================================================
        # COLLECT EXPERIENCE (ROLLOUT)
        # =================================================================
        # Run the environment for PPO_HORIZON steps, collecting
        # (obs, action, log_prob, reward, done, value) at each step.
        # This is the "on-policy" data that PPO trains on.

        for step in range(PPO_HORIZON):
            # Each agent selects an action from its observation
            actions, log_probs, values = ippo.select_actions(observations)

            # Step the environment with all 3 actions
            next_observations, rewards, dones, truncateds, infos = env.step(actions)

            # Combine terminated and truncated into a single done flag
            # for the experience buffer
            done_flags = []
            for i in range(NUM_AGENTS):
                done_flags.append(dones[i] or truncateds[i])

            # Store experience for all agents
            ippo.store_transitions(
                observations, actions, log_probs, rewards, done_flags, values
            )

            # Track episode stats (use agent 0's reward as representative,
            # since shared reward means all agents get similar totals)
            current_episode_reward += rewards[0]
            current_episode_length += 1
            total_steps += 1

            # Check if episode ended
            if done_flags[0]:
                # Record episode stats
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)


                if dones[0] and not truncateds[0]:
                    # Terminated: check termination reward to determine outcome
                    # Positive termination_reward = all contained = win
                    # Negative termination_reward = server compromised = loss
                    # Use the info dict to check server damage
                    info = infos[0]
                    server_dmg = info.get('server_damage', 0)
                    if server_dmg >= 500.0:  # SERVER_DAMAGE_THRESHOLD
                        episode_outcomes.append('loss')
                    else:
                        episode_outcomes.append('win')
                else:
                    episode_outcomes.append('survive')

                episode_count += 1
                current_episode_reward = 0.0
                current_episode_length = 0

                # Reset environment for next episode
                observations, infos = env.reset()
            else:
                observations = next_observations

            # Break if we've hit total timesteps
            if total_steps >= TOTAL_TIMESTEPS:
                break

        # =================================================================
        # PPO UPDATE
        # =================================================================
        # All agents update their policies using the collected experience.
        # Each agent runs its own independent PPO update.

        all_stats = ippo.update_all(observations)
        update_count += 1

        # =================================================================
        # LOGGING
        # =================================================================
        if update_count % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / max(elapsed, 1)

            # Compute recent episode stats (last 20 episodes)
            if len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-20:]
                recent_lengths = episode_lengths[-20:]
                recent_outcomes = episode_outcomes[-20:]

                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)

                win_count = 0
                loss_count = 0
                survive_count = 0
                for outcome in recent_outcomes:
                    if outcome == 'win':
                        win_count += 1
                    elif outcome == 'loss':
                        loss_count += 1
                    else:
                        survive_count += 1

                win_rate = win_count / len(recent_outcomes)

                # Agent 0 stats (representative)
                agent_stats = all_stats[0]

                print(
                    f"Update {update_count:4d} | "
                    f"Steps {total_steps:7d}/{TOTAL_TIMESTEPS} | "
                    f"Episodes {episode_count:4d} | "
                    f"Avg Reward {avg_reward:8.1f} | "
                    f"Avg Length {avg_length:5.1f} | "
                    f"Win/Loss/Surv {win_count}/{loss_count}/{survive_count} | "
                    f"WinRate {win_rate:.2f} | "
                    f"ActorLoss {agent_stats['actor_loss']:.4f} | "
                    f"CriticLoss {agent_stats['critic_loss']:.4f} | "
                    f"Entropy {agent_stats['entropy']:.4f} | "
                    f"SPS {steps_per_sec:.0f}"
                )

        # =================================================================
        # SAVE CHECKPOINT
        # =================================================================
        if update_count % SAVE_INTERVAL == 0:
            ippo.save_all(CHECKPOINT_DIR)
            print(f"  -> Checkpoint saved to {CHECKPOINT_DIR}/")

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {episode_count}")
    print(f"Total updates: {update_count}")

    # Final save
    ippo.save_all(CHECKPOINT_DIR)
    print(f"Final model saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    train()