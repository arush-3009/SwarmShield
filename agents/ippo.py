"""
SwarmShield IPPO
================

Independent PPO = one separate PPOAgent per defender.

There is no:
- parameter sharing
- centralized critic
- explicit communication

Coordination can only emerge because:
- all agents act in the same environment
- each agent sees the other agents' positions in its observation
- the reward includes a large shared component
"""

import os
from typing import Dict, List, Sequence

from agents.ppo import PPOAgent
from env.config import NUM_AGENTS


class IPPO:
    """
    Manager for N independent PPO agents.
    """

    def __init__(self, device):
        self.device = device
        self.num_agents = NUM_AGENTS
        self.agents: List[PPOAgent] = [PPOAgent(device) for _ in range(self.num_agents)]

    # -------------------------------------------------------------------------
    # Acting
    # -------------------------------------------------------------------------

    def select_actions(self, observations: Sequence):
        """
        Select one sampled action per agent.

        observations:
            list/sequence of length NUM_AGENTS

        Returns:
            actions: list[int]
            log_probs: list[float]
            values: list[float]
        """
        if len(observations) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} observations, got {len(observations)}"
            )

        actions = []
        log_probs = []
        values = []

        for agent_idx in range(self.num_agents):
            action, log_prob, value = self.agents[agent_idx].select_action(
                observations[agent_idx]
            )
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return actions, log_probs, values

    def select_actions_deterministic(self, observations: Sequence):
        """
        Greedy action selection, useful for evaluation / demo rollouts.
        """
        if len(observations) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} observations, got {len(observations)}"
            )

        actions = []
        for agent_idx in range(self.num_agents):
            action = self.agents[agent_idx].select_action_deterministic(
                observations[agent_idx]
            )
            actions.append(action)
        return actions

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    def store_transitions(
        self,
        observations: Sequence,
        actions: Sequence[int],
        log_probs: Sequence[float],
        rewards: Sequence[float],
        dones: Sequence[bool],
        values: Sequence[float],
    ) -> None:
        """
        Store one timestep for all agents.

        Each argument should be length NUM_AGENTS.
        """
        if not (
            len(observations)
            == len(actions)
            == len(log_probs)
            == len(rewards)
            == len(dones)
            == len(values)
            == self.num_agents
        ):
            raise ValueError("All transition components must have length NUM_AGENTS.")

        for agent_idx in range(self.num_agents):
            self.agents[agent_idx].store_transition(
                observations[agent_idx],
                actions[agent_idx],
                log_probs[agent_idx],
                rewards[agent_idx],
                dones[agent_idx],
                values[agent_idx],
            )

    # -------------------------------------------------------------------------
    # Updating
    # -------------------------------------------------------------------------

    def update_all(self, last_observations: Sequence, last_dones: Sequence[bool] | None = None):
        """
        Run PPO update for all agents.

        last_observations:
            observations after the final collected step, one per agent.

        last_dones:
            optional done flags for those last observations.
            If last_dones[i] is True, we bootstrap with 0.0 for agent i.
            Otherwise we bootstrap from that agent's critic.
        """
        if len(last_observations) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} last observations, got {len(last_observations)}"
            )

        if last_dones is not None and len(last_dones) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} last_dones flags, got {len(last_dones)}"
            )

        all_stats: List[Dict[str, float]] = []

        for agent_idx in range(self.num_agents):
            if last_dones is not None and last_dones[agent_idx]:
                last_value = 0.0
            else:
                last_value = self.agents[agent_idx].get_value(last_observations[agent_idx])

            stats = self.agents[agent_idx].update(last_value)
            all_stats.append(stats)

        return all_stats

    # -------------------------------------------------------------------------
    # Save / load
    # -------------------------------------------------------------------------

    def save_all(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        for agent_idx in range(self.num_agents):
            filepath = os.path.join(directory, f"agent_{agent_idx}.pt")
            self.agents[agent_idx].save(filepath)

    def load_all(self, directory: str) -> None:
        for agent_idx in range(self.num_agents):
            filepath = os.path.join(directory, f"agent_{agent_idx}.pt")
            self.agents[agent_idx].load(filepath)

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def clear_all_buffers(self) -> None:
        for agent in self.agents:
            agent.clear_buffer()

    def get_buffer_sizes(self) -> List[int]:
        return [agent.buffer_size() for agent in self.agents]