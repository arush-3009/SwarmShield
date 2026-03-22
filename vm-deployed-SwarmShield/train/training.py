"""
SwarmShield Training Script
============================

This is the main training loop for IPPO (Independent PPO).

What it does each iteration:
1. Collect PPO_HORIZON timesteps of experience by running the environment.
2. Run one PPO update for each of the 3 agents independently.
3. Log training statistics.
4. Save checkpoints periodically.

Checkpoint structure:
    checkpoints/latest/  — most recent weights, used for resuming training
    checkpoints/best/    — highest win-rate weights, used for demo/evaluation

How to run:
    cd SwarmShield-attempt2-saturday
    python -m train.training

To resume from where you left off, just run the same command again.
The script auto-detects checkpoints/latest/ and loads from there.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path so imports work from any directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.swarmshield_env import SwarmShieldEnv
from agents.ippo import IPPO
from env.config import (
    PPO_HORIZON,
    NUM_AGENTS,
    MAX_TIMESTEPS,
    OBSERVATION_SIZE,
    NUM_ACTIONS,
)


def train():

    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================

    TOTAL_TIMESTEPS = 3_000_000

    LOG_INTERVAL = 1
    SAVE_INTERVAL = 10

    CHECKPOINT_ROOT = "checkpoints"
    LATEST_DIR = os.path.join(CHECKPOINT_ROOT, "latest")
    BEST_DIR = os.path.join(CHECKPOINT_ROOT, "best")
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    os.makedirs(LATEST_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    STATS_WINDOW = 20

    # Which device to train on:
    #   0 = CUDA GPU
    #   1 = Apple Silicon MPS
    #   2 = CPU
    DEVICE_TO_TRAIN_ON = 2

    # =========================================================================
    # DEVICE SELECTION
    # =========================================================================

    if DEVICE_TO_TRAIN_ON == 0:
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
    elif DEVICE_TO_TRAIN_ON == 1:
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # =========================================================================
    # CREATE ENVIRONMENT AND AGENTS
    # =========================================================================

    env = SwarmShieldEnv()
    ippo = IPPO(device)

    # =========================================================================
    # RESUME FROM CHECKPOINT IF AVAILABLE
    # =========================================================================

    if os.path.exists(os.path.join(LATEST_DIR, "agent_0.pt")):
        ippo.load_all(LATEST_DIR)
        print(f"Resumed model weights from {LATEST_DIR}/")
    else:
        print("No checkpoint found. Starting fresh.")

    print(f"Observation size: {OBSERVATION_SIZE}")
    print(f"Number of actions: {NUM_ACTIONS}")
    print(f"Number of agents: {NUM_AGENTS}")
    print(f"Horizon (steps per PPO update): {PPO_HORIZON}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Max episode length: {MAX_TIMESTEPS}")
    print(f"Device: {device}")
    print(f"Training starting...\n")

    # =========================================================================
    # TRACKING VARIABLES
    # =========================================================================

    total_steps = 0
    update_count = 0
    episode_count = 0
    start_time = time.time()

    episode_rewards = []
    episode_lengths = []
    episode_outcomes = []

    episode_infected_uncontained = []
    episode_infected_blocked = []
    episode_infected_quarantined = []
    episode_server_damage = []
    episode_false_positives = []

    current_episode_reward = 0.0
    current_episode_length = 0

    last_dones = [False] * NUM_AGENTS

    best_win_rate = -1.0
    best_server_damage = float("inf")

    # =========================================================================
    # RESET ENVIRONMENT
    # =========================================================================

    observations, infos = env.reset()

    # =========================================================================
    # MAIN TRAINING LOOP (with Ctrl+C safety)
    # =========================================================================

    try:
        while total_steps < TOTAL_TIMESTEPS:

            # =================================================================
            # COLLECT EXPERIENCE (ROLLOUT)
            # =================================================================

            for step in range(PPO_HORIZON):

                actions, log_probs, values = ippo.select_actions(observations)
                next_observations, rewards, dones, truncateds, infos = env.step(actions)

                done_flags = []
                for i in range(NUM_AGENTS):
                    done_flags.append(dones[i] or truncateds[i])

                ippo.store_transitions(
                    observations, actions, log_probs, rewards, done_flags, values
                )

                current_episode_reward += rewards[0]
                current_episode_length += 1
                total_steps += 1

                last_dones = list(done_flags)

                if done_flags[0]:

                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)

                    info = infos[0]
                    if dones[0] and not truncateds[0]:
                        if info.get('server_compromised', False):
                            episode_outcomes.append('loss')
                        elif info.get('all_infections_quarantined', False):
                            episode_outcomes.append('win')
                        else:
                            episode_outcomes.append('loss')
                    else:
                        episode_outcomes.append('survive')

                    episode_infected_uncontained.append(
                        info.get('infected_uncontained', 0)
                    )
                    episode_infected_blocked.append(
                        info.get('infected_blocked', 0)
                    )
                    episode_infected_quarantined.append(
                        info.get('infected_quarantined', 0)
                    )
                    episode_server_damage.append(
                        info.get('server_damage', 0.0)
                    )
                    episode_false_positives.append(
                        info.get('clean_blocked', 0) + info.get('clean_quarantined', 0)
                    )

                    episode_count += 1
                    current_episode_reward = 0.0
                    current_episode_length = 0

                    observations, infos = env.reset()
                    last_dones = [False] * NUM_AGENTS

                else:
                    observations = next_observations

                if total_steps >= TOTAL_TIMESTEPS:
                    break

            # =================================================================
            # PPO UPDATE
            # =================================================================

            all_stats = ippo.update_all(observations, last_dones=last_dones)
            update_count += 1

            # =================================================================
            # LOGGING
            # =================================================================

            if update_count % LOG_INTERVAL == 0 and len(episode_rewards) > 0:

                elapsed = time.time() - start_time
                steps_per_sec = total_steps / max(elapsed, 1.0)

                recent_rewards = episode_rewards[-STATS_WINDOW:]
                recent_lengths = episode_lengths[-STATS_WINDOW:]
                recent_outcomes = episode_outcomes[-STATS_WINDOW:]
                recent_inf_uc = episode_infected_uncontained[-STATS_WINDOW:]
                recent_inf_blk = episode_infected_blocked[-STATS_WINDOW:]
                recent_inf_quar = episode_infected_quarantined[-STATS_WINDOW:]
                recent_svr_dmg = episode_server_damage[-STATS_WINDOW:]
                recent_fp = episode_false_positives[-STATS_WINDOW:]

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

                avg_inf_uc = np.mean(recent_inf_uc)
                avg_inf_blk = np.mean(recent_inf_blk)
                avg_inf_quar = np.mean(recent_inf_quar)
                avg_svr_dmg = np.mean(recent_svr_dmg)
                avg_fp = np.mean(recent_fp)

                agent_stats = all_stats[0]

                print(
                    f"Update {update_count:4d} | "
                    f"Steps {total_steps:7d}/{TOTAL_TIMESTEPS} | "
                    f"Ep {episode_count:4d} | "
                    f"AvgRew {avg_reward:7.1f} | "
                    f"AvgLen {avg_length:5.1f} | "
                    f"W/L/S {win_count}/{loss_count}/{survive_count} | "
                    f"WR {win_rate:.2f} | "
                    f"InfUC {avg_inf_uc:.1f} | "
                    f"InfBlk {avg_inf_blk:.1f} | "
                    f"InfQ {avg_inf_quar:.1f} | "
                    f"SvrDmg {avg_svr_dmg:.0f} | "
                    f"FP {avg_fp:.1f} | "
                    f"ActL {agent_stats['actor_loss']:.4f} | "
                    f"CrtL {agent_stats['critic_loss']:.4f} | "
                    f"Ent {agent_stats['entropy']:.3f} | "
                    f"SPS {steps_per_sec:.0f}"
                )

                # ---- Check for new best ----
                is_new_best = False
                if win_rate > best_win_rate:
                    is_new_best = True
                elif win_rate == best_win_rate and avg_svr_dmg < best_server_damage:
                    is_new_best = True

                if is_new_best:
                    best_win_rate = win_rate
                    best_server_damage = avg_svr_dmg
                    ippo.save_all(BEST_DIR)
                    print(f"  -> NEW BEST saved! WR={best_win_rate:.2f} SvrDmg={best_server_damage:.0f}")

            # =================================================================
            # SAVE LATEST CHECKPOINT PERIODICALLY
            # =================================================================

            if update_count % SAVE_INTERVAL == 0:
                ippo.save_all(LATEST_DIR)
                print(f"  -> Latest saved to {LATEST_DIR}/")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by Ctrl+C.")

    # =========================================================================
    # ALWAYS SAVE ON EXIT (normal finish or Ctrl+C)
    # =========================================================================

    ippo.save_all(LATEST_DIR)
    print(f"\nLatest model saved to {LATEST_DIR}/")
    print(f"Best model remains in {BEST_DIR}/")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {episode_count}")
    print(f"Total PPO updates: {update_count}")

    if len(episode_rewards) > 0:
        print(f"\nFinal {STATS_WINDOW}-episode averages:")
        final_rewards = episode_rewards[-STATS_WINDOW:]
        final_outcomes = episode_outcomes[-STATS_WINDOW:]

        final_wins = 0
        for outcome in final_outcomes:
            if outcome == 'win':
                final_wins += 1

        print(f"  Average reward:  {np.mean(final_rewards):.1f}")
        print(f"  Win rate:        {final_wins / len(final_outcomes):.2f}")
        print(f"  Average length:  {np.mean(episode_lengths[-STATS_WINDOW:]):.1f}")

    # =========================================================================
    # TRAINING VISUALIZATION
    # =========================================================================

    if len(episode_rewards) > 5:

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle("SwarmShield IPPO Training", fontsize=14)

        window = min(STATS_WINDOW, len(episode_rewards))

        def running_avg(data, w):
            result = []
            for i in range(len(data)):
                start = max(0, i - w + 1)
                chunk = data[start:i+1]
                result.append(sum(chunk) / len(chunk))
            return result

        ax = axes[0][0]
        ax.plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(running_avg(episode_rewards, window), color='blue', label=f'{window}-ep avg')
        ax.set_title("Episode Reward (agent 0)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0][1]
        ax.plot(episode_lengths, alpha=0.3, color='green', label='Raw')
        ax.plot(running_avg(episode_lengths, window), color='green', label=f'{window}-ep avg')
        ax.axhline(y=MAX_TIMESTEPS, color='red', linestyle='--', alpha=0.5, label='Max')
        ax.set_title("Episode Length")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1][0]
        cumulative_wr = []
        for i, outcome in enumerate(episode_outcomes):
            start = max(0, i - window + 1)
            recent_wins = 0
            for j in range(start, i + 1):
                if episode_outcomes[j] == 'win':
                    recent_wins += 1
            cumulative_wr.append(recent_wins / (i - start + 1))
        ax.plot(cumulative_wr, color='purple')
        ax.set_title(f"Win Rate ({window}-ep rolling)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax = axes[1][1]
        ax.plot(running_avg(episode_infected_uncontained, window), label='Uncontained', color='red')
        ax.plot(running_avg(episode_infected_blocked, window), label='Blocked', color='orange')
        ax.plot(running_avg(episode_infected_quarantined, window), label='Quarantined', color='green')
        ax.set_title(f"Infected Hosts at Episode End ({window}-ep avg)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2][0]
        ax.plot(episode_server_damage, alpha=0.3, color='red', label='Raw')
        ax.plot(running_avg(episode_server_damage, window), color='red', label=f'{window}-ep avg')
        ax.axhline(y=300, color='black', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_title("Server Damage at Episode End")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Damage")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2][1]
        ax.plot(episode_false_positives, alpha=0.3, color='orange', label='Raw')
        ax.plot(running_avg(episode_false_positives, window), color='orange', label=f'{window}-ep avg')
        ax.set_title("False Positives at Episode End")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Clean hosts contained")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_ROOT, "training_plots.png"), dpi=150)
        print(f"Training plots saved to {CHECKPOINT_ROOT}/training_plots.png")
        plt.show()


if __name__ == "__main__":
    train()