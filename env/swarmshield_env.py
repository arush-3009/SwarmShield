"""
SwarmShield Gymnasium Environment
===================================

This is the main environment class that ties everything together.
It inherits from gymnasium.Env and implements the standard reset() and step()
interface that PPO training loops expect.

WHAT THIS FILE DOES:
- Creates the network, traffic manager, and attacker on initialization
- reset(): randomizes everything for a new episode, returns initial observations
- step(): takes actions from all 3 agents, simulates one timestep, returns
  observations, rewards, done flags, and info dicts

HOW ONE TIMESTEP WORKS (inside step()):
1. Process each agent's action (move, block, quarantine, unblock, or observe)
2. Run the attacker logic (beaconing, scanning, infection, attack)
3. Generate normal traffic for all clean hosts
4. Prune old traffic records and decay suspicious scores
5. Compute observations for each agent at their current node
6. Compute rewards based on network health and agent actions
7. Check if the episode is over (server compromised, all contained, or timeout)

MULTI-AGENT DESIGN:
This is a multi-agent environment with 3 independent agents (IPPO).
step() takes a list of 3 actions and returns lists of 3 observations/rewards.
Each agent has its own observation (traffic at its node + global info) but
they all share the same reward signal.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.config import (
    NUM_HOSTS,
    NUM_SUBNETS,
    NUM_AGENTS,
    NUM_ACTIONS,
    OBSERVATION_SIZE,
    NUM_TRAFFIC_FEATURES,
    MAX_TIMESTEPS,
    INITIAL_INFECTIONS,
    OBSERVATION_WINDOW,
    SUBNET_HOSTS,
    HOST_TO_SUBNET,
    SERVER_HOST_ID,
    SERVER_DAMAGE_THRESHOLD,
    # Action IDs
    ACTION_OBSERVE,
    ACTION_MOVE_WITHIN_SUBNET,
    ACTION_MOVE_TO_SUBNET_0,
    ACTION_MOVE_TO_SUBNET_5,
    ACTION_BLOCK,
    ACTION_QUARANTINE,
    ACTION_UNBLOCK,
    # Movement costs
    MOVE_WITHIN_SUBNET_COST,
    MOVE_CROSS_SUBNET_COST,
    # Rewards
    REWARD_HEALTHY_HOST,
    REWARD_UNCONTAINED_INFECTION,
    REWARD_FALSE_BLOCK_PER_STEP,
    REWARD_AGENT_OVERLAP,
    REWARD_CORRECT_QUARANTINE,
    REWARD_CORRECT_BLOCK,
    REWARD_FALSE_QUARANTINE,
    REWARD_FALSE_BLOCK_EVENT,
    REWARD_CORRECT_UNBLOCK,
    REWARD_BAD_UNBLOCK,
    REWARD_MOVE_WITHIN_SUBNET,
    REWARD_MOVE_CROSS_SUBNET,
    REWARD_SERVER_COMPROMISED,
    REWARD_ALL_CONTAINED,
    REWARD_SURVIVED,
)
from env.network import Network, STATUS_CLEAN, STATUS_INFECTED, STATUS_BLOCKED, STATUS_QUARANTINED
from env.traffic import TrafficManager
from env.attacker import Attacker


class SwarmShieldEnv(gym.Env):
    """
    The SwarmShield multi-agent Gymnasium environment.

    3 defensive RL agents patrol the Dunder Mifflin network, observing
    traffic at their current node and taking actions to contain a botnet.

    This environment is designed for IPPO (Independent PPO) training:
    each agent gets its own observation and picks its own action independently.
    All agents share the same reward
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        """
        Initialize the environment.

        seed: optional int for reproducible episodes. If None, episodes
              are random (used during training). If set, same seed
              produces identical episodes.
        """
        super().__init__()

        
        self.rng = np.random.default_rng(seed)

        # Create the three core components
        self.network = Network()
        self.traffic_manager = TrafficManager()
        self.attacker = Attacker()

        # Agent state tracking
        # positions[i] = which host (0-17) agent i is currently at
        # transit_remaining[i] = how many timesteps until agent i arrives
        #   0 means agent is at its node and can observe/act
        #   1+ means agent is moving and is blind
        # transit_destination[i] = where agent i is heading (only valid if in transit)
        self.agent_positions = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.agent_transit_remaining = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.agent_transit_destination = np.zeros(NUM_AGENTS, dtype=np.int32)

        # Episode tracking
        self.current_timestep = 0
        self.done = False

        # Gymnasium spaces
        # Action space: each agent picks one of 11 discrete actions
        # We define it for a single agent — the training loop handles multi-agent
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: each agent gets a 71-dimensional float vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        This is called at the start of every episode during training.
        Everything gets randomized: vulnerability scores, initial infections,
        agent starting positions.

        Returns:
            observations: list of 3 numpy arrays, one per agent (each shape (71,))
            infos: list of 3 info dicts (empty for now)
        """
        # Optionally update the random seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset all components
        self.network.reset(self.rng)
        self.traffic_manager.reset()
        self.attacker.reset()

        # Reset episode tracking
        self.current_timestep = 0
        self.done = False

        # Place agents at random nodes
        # Each agent starts at a different random host
        starting_positions = self.rng.choice(
            NUM_HOSTS, size=NUM_AGENTS, replace=False
        )
        self.agent_positions = starting_positions.astype(np.int32)
        self.agent_transit_remaining = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.agent_transit_destination = np.zeros(NUM_AGENTS, dtype=np.int32)

        # Infect initial hosts (1-2 random hosts, never the server)
        self.attacker.infect_initial_hosts(
            self.network, self.current_timestep, INITIAL_INFECTIONS, self.rng
        )

        # Generate initial normal traffic so agents have something to observe
        self.traffic_manager.generate_normal_traffic(
            self.network, self.current_timestep, self.rng
        )

        # Also run one step of attacker so initial infected hosts produce
        # their first beacon (gives agents something to detect from step 1)
        self.attacker.step(
            self.network, self.traffic_manager, self.current_timestep, self.rng
        )

        # Build observations for all agents
        observations = self._build_observations()

        # Info dicts (one per agent, empty for now but could contain debug data)
        infos = [{} for _ in range(NUM_AGENTS)]

        return observations, infos

    def step(self, actions):
        """
        Execute one timestep of the environment.

        This is the core simulation loop. It takes actions from all 3 agents,
        simulates everything that happens in one timestep, and returns the
        results.

        actions: list of 3 ints, one action per agent (each in range [0, 10])

        Returns:
            observations: list of 3 numpy arrays (shape (71,) each)
            rewards: list of 3 floats (same reward for all agents)
            dones: list of 3 bools (all True or all False — episode ends for everyone)
            truncateds: list of 3 bools (True if episode hit max timesteps)
            infos: list of 3 dicts with debug/logging information
        """
        if self.done:
            # Episode already ended — return zeros
            obs = [np.zeros(OBSERVATION_SIZE, dtype=np.float32) for _ in range(NUM_AGENTS)]
            return obs, [0.0] * NUM_AGENTS, [True] * NUM_AGENTS, [False] * NUM_AGENTS, [{} for _ in range(NUM_AGENTS)]

        self.current_timestep += 1

        # =====================================================================
        # STEP 1: Process agent actions
        # =====================================================================
        # Each agent's action is processed independently.
        # Agents in transit have their action ignored (they're still moving).
        # Track one-time event rewards from actions (block/quarantine/unblock).
        event_rewards = [0.0 for _ in range(NUM_AGENTS)]

        for agent_idx in range(NUM_AGENTS):
            action = actions[agent_idx]

            # If agent is in transit (moving between nodes), decrement counter
            # and skip their action — they can't do anything while moving.
            if self.agent_transit_remaining[agent_idx] > 0:
                self.agent_transit_remaining[agent_idx] -= 1

                # Check if agent just arrived at destination
                if self.agent_transit_remaining[agent_idx] == 0:
                    self.agent_positions[agent_idx] = self.agent_transit_destination[agent_idx]

                continue  # Skip action processing — agent is blind

            # Agent is at a node and can act
            current_node = self.agent_positions[agent_idx]

            # --- ACTION: OBSERVE ---
            if action == ACTION_OBSERVE:
                # Do nothing. Free. Agent stays and watches.
                pass

            # --- ACTION: MOVE WITHIN SUBNET ---
            elif action == ACTION_MOVE_WITHIN_SUBNET:
                current_subnet = HOST_TO_SUBNET[current_node]
                same_subnet_hosts = SUBNET_HOSTS[current_subnet]

                # Find other hosts in same subnet (not current node)
                other_hosts = []
                for h in same_subnet_hosts:
                    if h != current_node:
                        other_hosts.append(h)

                if len(other_hosts) > 0:
                    dest = self.rng.choice(other_hosts)
                    self.agent_transit_remaining[agent_idx] = MOVE_WITHIN_SUBNET_COST
                    self.agent_transit_destination[agent_idx] = dest
                    event_rewards[agent_idx] += REWARD_MOVE_WITHIN_SUBNET

            # --- ACTION: MOVE TO SPECIFIC SUBNET ---
            elif ACTION_MOVE_TO_SUBNET_0 <= action <= ACTION_MOVE_TO_SUBNET_5:
                target_subnet = action - ACTION_MOVE_TO_SUBNET_0
                target_hosts = SUBNET_HOSTS[target_subnet]

                # Pick a random host in the target subnet
                dest = self.rng.choice(target_hosts)

                # Determine if this is within-subnet or cross-subnet move
                current_subnet = HOST_TO_SUBNET[current_node]
                if target_subnet == current_subnet:
                    # Moving within same subnet (might land on same node — that's just a wasted move)
                    self.agent_transit_remaining[agent_idx] = MOVE_WITHIN_SUBNET_COST
                    self.agent_transit_destination[agent_idx] = dest
                    event_rewards[agent_idx] += REWARD_MOVE_WITHIN_SUBNET
                else:
                    # Cross-subnet move — takes longer
                    self.agent_transit_remaining[agent_idx] = MOVE_CROSS_SUBNET_COST
                    self.agent_transit_destination[agent_idx] = dest
                    event_rewards[agent_idx] += REWARD_MOVE_CROSS_SUBNET

            # --- ACTION: BLOCK HOST ---
            elif action == ACTION_BLOCK:
                host = self.network.get_host(current_node)

                # Can only block clean or infected hosts (not already blocked/quarantined)
                if host.is_clean or host.is_infected:
                    was_infected = host.is_infected
                    host.block()

                    if was_infected:
                        event_rewards[agent_idx] += REWARD_CORRECT_BLOCK
                    else:
                        event_rewards[agent_idx] += REWARD_FALSE_BLOCK_EVENT

            # --- ACTION: QUARANTINE HOST ---
            elif action == ACTION_QUARANTINE:
                host = self.network.get_host(current_node)

                if host.is_clean or host.is_infected:
                    was_infected = host.is_infected
                    host.quarantine()

                    if was_infected:
                        event_rewards[agent_idx] += REWARD_CORRECT_QUARANTINE
                    else:
                        event_rewards[agent_idx] += REWARD_FALSE_QUARANTINE

            # --- ACTION: UNBLOCK HOST ---
            elif action == ACTION_UNBLOCK:
                
                host = self.network.get_host(current_node)

                if host.is_blocked or host.is_quarantined:
                    was_correctly_contained = (host.timestep_infected >= 0)
                    host.unblock()

                    if was_correctly_contained:
                        # Unblocking an infected host is very bad — it's now
                        # free to beacon, scan, and attack again.
                        # Heavy penalty to prevent unblock/reblock farming.
                        event_rewards[agent_idx] += REWARD_SERVER_COMPROMISED * 0.25  # -50.0
                    else:
                        event_rewards[agent_idx] += REWARD_CORRECT_UNBLOCK

        # =====================================================================
        # STEP 2: Run attacker logic
        # =====================================================================
        # All active infected hosts beacon, scan, attempt infections, and
        # potentially attack the server. Returns list of newly infected host IDs.
        newly_infected = self.attacker.step(
            self.network, self.traffic_manager, self.current_timestep, self.rng
        )

        # =====================================================================
        # STEP 3: Generate normal traffic
        # =====================================================================
        # Every clean, operational host generates background traffic.
        # This mixes with the malicious traffic from Step 2.
        self.traffic_manager.generate_normal_traffic(
            self.network, self.current_timestep, self.rng
        )

        # =====================================================================
        # STEP 4: Maintenance — prune old records and decay suspicious scores
        # =====================================================================
        self.traffic_manager.prune_old_records(self.current_timestep)
        self.traffic_manager.decay_long_term_scores()

        # =====================================================================
        # STEP 5: Check termination conditions
        # =====================================================================
        terminated = False
        truncated = False
        termination_reward = 0.0

        # Check if server is compromised -> attacker wins
        
        # Two ways the attacker wins:
        # 1. Accumulated DDoS damage exceeds threshold
        # 2. Server itself gets directly infected by a scan
        server_host = self.network.get_host(SERVER_HOST_ID)
        
        if ((self.network.is_server_compromised(self.attacker.server_damage, SERVER_DAMAGE_THRESHOLD)) or
            (server_host.is_infected)):
            
            terminated = True
            termination_reward = REWARD_SERVER_COMPROMISED

        # Check if all infections are contained (defenders win)
        elif self.network.all_infections_contained():
            terminated = True
            termination_reward = REWARD_ALL_CONTAINED

        # Check if episode hit max timesteps
        elif self.current_timestep >= MAX_TIMESTEPS:
            truncated = True
            # Survived the full episode with server safe — partial victory
            termination_reward = REWARD_SURVIVED

        self.done = terminated or truncated

        # =====================================================================
        # STEP 6: Compute rewards
        # =====================================================================
        shared_reward = self._compute_shared_reward()
        shared_reward += termination_reward

        # Each agent gets: shared reward + their individual event reward
        rewards = []
        for agent_idx in range(NUM_AGENTS):
            total = shared_reward + event_rewards[agent_idx]
            rewards.append(total)

        # =====================================================================
        # STEP 7: Build observations
        # =====================================================================
        observations = self._build_observations()

        # =====================================================================
        # STEP 8: Build info dicts
        # =====================================================================
        status_counts = self.network.count_by_status()
        info = {
            'timestep': self.current_timestep,
            'infected_count': status_counts['infected'],
            'blocked_count': status_counts['blocked'],
            'quarantined_count': status_counts['quarantined'],
            'clean_count': status_counts['clean'],
            'server_damage': self.attacker.server_damage,
            'newly_infected': newly_infected,
            'false_blocks': status_counts['clean_blocked'],
            'false_quarantines': status_counts['clean_quarantined'],
        }
        infos = [info.copy() for _ in range(NUM_AGENTS)]

        # Return in Gymnasium format
        dones = [terminated] * NUM_AGENTS
        truncateds = [truncated] * NUM_AGENTS

        return observations, rewards, dones, truncateds, infos

    def _compute_shared_reward(self):
        """
        Compute the per-timestep reward shared by all agents.

        This includes:
        - Positive reward for each healthy, operational host
        - Negative reward for each uncontained infection
        - Negative reward for each falsely blocked/quarantined clean host
        - Negative reward for agents at the same node
        """
        reward = 0.0
        status_counts = self.network.count_by_status()

        # +1.0 per clean, unblocked host (incentivize keeping office running)
        reward += REWARD_HEALTHY_HOST * status_counts['clean']

        # -3.0 per infected, uncontained host (pressure to act)
        reward += REWARD_UNCONTAINED_INFECTION * status_counts['infected']

        # -0.5 per falsely blocked/quarantined clean host
        false_positives = status_counts['clean_blocked'] + status_counts['clean_quarantined']
        reward += REWARD_FALSE_BLOCK_PER_STEP * false_positives

        # -2.0 per pair of agents at the same node (discourage clustering)
        # With 3 agents, possible pairs: (0,1), (0,2), (1,2) = 3 pairs max
        for i in range(NUM_AGENTS):
            for j in range(i + 1, NUM_AGENTS):
                # Only count if both agents are actually at their nodes (not in transit)
                if self.agent_transit_remaining[i] == 0 and self.agent_transit_remaining[j] == 0:
                    if self.agent_positions[i] == self.agent_positions[j]:
                        reward += REWARD_AGENT_OVERLAP

        return reward

    def _build_observations(self):
        """
        Build the observation vector for each agent.

        Each agent's observation has two parts:
        1. Traffic features (13 values) from the agent's current node
        2. Global features (58 values) — agent position, other agents, network status

        Total: 71 values per agent, all normalized to [0, 1].

        If an agent is in transit (moving between nodes), its traffic features
        are zeros (it can't observe anything while running between desks).
        """
        observations = []

        for agent_idx in range(NUM_AGENTS):
            obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

            # === TRAFFIC FEATURES (indices 0-12) ===
            if self.agent_transit_remaining[agent_idx] == 0:
                # Agent is at a node — compute traffic features for that node
                current_node = self.agent_positions[agent_idx]
                traffic_features = self.traffic_manager.compute_features(current_node)
                obs[0:NUM_TRAFFIC_FEATURES] = traffic_features
            else:
                # Agent is in transit — can't observe, traffic features are zeros
                pass

            # === GLOBAL FEATURES (indices 13-70) ===

            # This agent's position — one-hot encoded (18 values)
            # Even if in transit, we report the current position (where agent IS,
            # not where it's going) so the network knows the agent's general area.
            offset = NUM_TRAFFIC_FEATURES
            obs[offset + self.agent_positions[agent_idx]] = 1.0

            # Other agents' positions — one-hot encoded (18 values each)
            # Agent sees where its teammates are, enabling implicit coordination.
            other_agents = []
            for other_idx in range(NUM_AGENTS):
                if other_idx != agent_idx:
                    other_agents.append(other_idx)

            for i, other_idx in enumerate(other_agents):
                other_offset = offset + NUM_HOSTS + (i * NUM_HOSTS)
                obs[other_offset + self.agent_positions[other_idx]] = 1.0

            # Network status features (4 values)
            status_offset = offset + (NUM_HOSTS * 3)
            status_counts = self.network.count_by_status()

            # Number of quarantined hosts (normalized by total hosts)
            obs[status_offset] = status_counts['quarantined'] / NUM_HOSTS

            # Number of blocked hosts (normalized by total hosts)
            obs[status_offset + 1] = status_counts['blocked'] / NUM_HOSTS

            # Episode progress (0.0 at start, 1.0 at end)
            obs[status_offset + 2] = self.current_timestep / MAX_TIMESTEPS

            # Is this agent currently in transit? (0 or 1)
            obs[status_offset + 3] = 1.0 if self.agent_transit_remaining[agent_idx] > 0 else 0.0

            observations.append(obs)

        return observations

    def get_state_for_visualization(self):
        """
        Get the full environment state for rendering in Godot.

        This returns EVERYTHING — infection status, agent positions,
        traffic data, server damage — that the visualization needs.
        The RL agents never call this; it's only for the demo display.

        Returns a dict that gets sent to Godot over WebSocket.
        """
        host_states = []
        for host in self.network.hosts:
            host_states.append({
                'id': host.host_id,
                'name': host.name,
                'subnet': host.subnet_id,
                'status': host.status,
                'is_server': host.is_server,
            })

        agent_states = []
        for agent_idx in range(NUM_AGENTS):
            agent_states.append({
                'position': int(self.agent_positions[agent_idx]),
                'in_transit': int(self.agent_transit_remaining[agent_idx]) > 0,
                'destination': int(self.agent_transit_destination[agent_idx]),
            })

        return {
            'timestep': self.current_timestep,
            'hosts': host_states,
            'agents': agent_states,
            'server_damage': self.attacker.server_damage,
            'server_damage_threshold': SERVER_DAMAGE_THRESHOLD,
            'done': self.done,
        }

    def __str__(self):
        """Pretty-print the environment state for debugging."""
        lines = []
        lines.append(f"\n=== SwarmShield Environment (Step {self.current_timestep}/{MAX_TIMESTEPS}) ===")
        lines.append(f"Server Damage: {self.attacker.server_damage:.1f} / {SERVER_DAMAGE_THRESHOLD}")

        # Agent info
        for i in range(NUM_AGENTS):
            pos = self.agent_positions[i]
            host_name = self.network.get_host(pos).name
            transit = self.agent_transit_remaining[i]
            if transit > 0:
                dest = self.agent_transit_destination[i]
                dest_name = self.network.get_host(dest).name
                lines.append(f"Agent {i}: In transit to {dest_name} ({transit} steps remaining)")
            else:
                lines.append(f"Agent {i}: At {host_name} (host {pos})")

        # Network state
        lines.append(str(self.network))


        return "\n".join(lines)