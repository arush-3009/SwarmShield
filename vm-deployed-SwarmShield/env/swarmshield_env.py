"""
SwarmShield Multi-Agent Environment
===================================

This is the top-level environment class used by IPPO training.

It owns:
- the Network (host state + containment logic)
- the TrafficManager (traffic history + 16 traffic features)
- the Attacker (scripted malicious behavior)
- the 3 defender agent runtime states (positions + transit)

Interface required by the project spec:
- reset() -> list of 3 observations, list of 3 info dicts
- step(actions) -> list of 3 observations, list of 3 rewards,
                   list of 3 terminated, list of 3 truncated,
                   list of 3 info dicts

Design notes:
- We do NOT subclass gym.Env here. The spec only requires reset/step with
  per-agent lists, and keeping this plain makes it simpler for rushed training.
- Movement transit is handled here, not in network.py.
- The firewall / containment semantics are centralized in network.py.
- Malicious traffic generation is centralized in attacker.py.
- Legitimate traffic generation and traffic-feature computation are centralized
  in traffic.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.attacker import Attacker
from env.config import (
    ACTION_BLOCK,
    ACTION_MOVE_BASE,
    ACTION_MOVE_LAST,
    ACTION_OBSERVE,
    ACTION_QUARANTINE,
    ACTION_UNBLOCK,
    FEAT_CURRENT_HOST_BLOCKED,
    FEAT_CURRENT_HOST_QUARANTINED,
    FEAT_EPISODE_PROGRESS,
    FEAT_FRACTION_BLOCKED,
    FEAT_FRACTION_QUARANTINED,
    FEAT_IN_TRANSIT,
    FEAT_OTHER_AGENT_1_POS_START,
    FEAT_OTHER_AGENT_2_POS_START,
    FEAT_SELF_POS_START,
    FEAT_SERVER_DAMAGE,
    INITIAL_INFECTIONS,
    MAX_TIMESTEPS,
    MOVE_CROSS_SUBNET_COST,
    MOVE_WITHIN_SUBNET_COST,
    NUM_ACTIONS,
    NUM_AGENTS,
    NUM_HOSTS,
    NUM_TRAFFIC_FEATURES,
    OBSERVATION_SIZE,
    REWARD_AGENT_OVERLAP,
    REWARD_ALL_QUARANTINED,
    REWARD_BAD_UNBLOCK,
    REWARD_CORRECT_BLOCK,
    REWARD_CORRECT_QUARANTINE,
    REWARD_CORRECT_UNBLOCK,
    REWARD_FALSE_BLOCK,
    REWARD_FALSE_BLOCK_PER_STEP,
    REWARD_FALSE_QUARANTINE,
    REWARD_FALSE_QUARANTINE_PER_STEP,
    REWARD_FALSE_QUARANTINE_UPGRADE,
    REWARD_HEALTHY_HOST,
    REWARD_INFECTED_BLOCKED,
    REWARD_INFECTED_QUARANTINED,
    REWARD_INFECTED_UNCONTAINED,
    REWARD_MOVE_CROSS_SUBNET,
    REWARD_MOVE_WITHIN_SUBNET,
    REWARD_NEW_INFECTION,
    REWARD_QUARANTINE_UPGRADE,
    REWARD_SERVER_COMPROMISED,
    REWARD_SERVER_DAMAGE,
    REWARD_SURVIVED,
    SERVER_DAMAGE_THRESHOLD,
)
from env.network import CONTAINMENT_BLOCKED, CONTAINMENT_NONE, Network
from env.traffic import TrafficManager


@dataclass
class AgentRuntimeState:
    """
    Runtime state for one defender agent.

    current_host:
        The last host the agent is on / was on.
        While in transit, we keep this at the last occupied host.

    in_transit:
        True if the agent is currently moving and therefore blind.

    transit_target:
        Destination host if moving, else None.

    transit_remaining:
        Number of future step-start transit decrements left before arrival.
    """

    agent_id: int
    current_host: int
    in_transit: bool = False
    transit_target: Optional[int] = None
    transit_remaining: int = 0

    def begin_transit(self, target_host: int, travel_time: int) -> None:
        self.in_transit = True
        self.transit_target = target_host
        self.transit_remaining = travel_time

    def advance_transit(self) -> None:
        """
        Advance movement by one step-start tick.

        This timing choice means:
        - move with cost 1 -> current step's returned observation is blind,
          then agent arrives at the start of next step and can act there.
        - move with cost 2 -> two blind returned observations.
        """
        if not self.in_transit:
            return

        self.transit_remaining -= 1
        if self.transit_remaining <= 0:
            self.current_host = int(self.transit_target)
            self.transit_target = None
            self.in_transit = False
            self.transit_remaining = 0


class SwarmShieldEnv:
    """
    Final multi-agent SwarmShield environment.

    High-level step order:
    1. Advance ongoing transit.
    2. Process current agent actions.
    3. Decay long-memory suspicion scores.
    4. Run attacker.
    5. Generate legitimate traffic.
    6. Prune old traffic records.
    7. Check termination.
    8. Compute rewards.
    9. Build observations.
    """

    # The spec does not define initial defender positions.
    # These are stable defaults across 3 different subnets.
    DEFAULT_INITIAL_AGENT_POSITIONS = (0, 5, 12)

    def __init__(
        self,
        seed: Optional[int] = None,
        initial_agent_positions: Optional[Sequence[int]] = None,
    ) -> None:
        self.network = Network()
        self.traffic_manager = TrafficManager()
        self.attacker = Attacker()

        self.rng = np.random.default_rng(seed)
        self._initial_agent_positions = self._validate_initial_agent_positions(
            initial_agent_positions
        )

        self.agent_states: List[AgentRuntimeState] = []
        self.current_timestep = 0
        self.done = False

        # Cached step-level bookkeeping for infos/debugging.
        self.last_newly_infected: List[int] = []
        self.last_server_damage_delta = 0.0
        self.last_shared_reward = 0.0
        self.last_event_rewards = [0.0 for _ in range(NUM_AGENTS)]

    # -------------------------------------------------------------------------
    # Reset / Step
    # -------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """
        Reset the full environment.

        Returns:
        - observations: list of 3 np.ndarray, each shape (77,), dtype float32
        - infos: list of 3 dicts
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.network.reset(self.rng)
        self.traffic_manager.reset()
        self.attacker.reset()

        self.current_timestep = 0
        self.done = False

        self.last_newly_infected = []
        self.last_server_damage_delta = 0.0
        self.last_shared_reward = 0.0
        self.last_event_rewards = [0.0 for _ in range(NUM_AGENTS)]

        # Initial infection happens at reset per spec.
        self.network.seed_initial_infections(
            self.rng,
            INITIAL_INFECTIONS,
            current_timestep=0,
        )

        self.agent_states = [
            AgentRuntimeState(agent_id=i, current_host=host_id)
            for i, host_id in enumerate(self._initial_agent_positions)
        ]

        observations = self._build_all_observations()
        infos = [
            self._build_info_for_agent(
                agent_id=i,
                action_taken=None,
                action_result=None,
                local_event_reward=0.0,
                shared_reward=0.0,
                terminal_reward=0.0,
                ignored_due_to_transit=False,
                move_started=False,
                arrived_this_step=False,
            )
            for i in range(NUM_AGENTS)
        ]
        return observations, infos

    def step(self, actions: Sequence[int]):
        """
        Run one environment timestep.

        Input:
        - actions: sequence of 3 ints

        Returns:
        - observations: list[3] of np.ndarray shape (77,)
        - rewards: list[3] of float
        - terminated: list[3] of bool
        - truncated: list[3] of bool
        - infos: list[3] of dict
        """
        if self.done:
            raise RuntimeError(
                "Cannot call step() on a finished episode. Call reset() first."
            )

        self._validate_actions(actions)

        action_results: List[Optional[object]] = [None] * NUM_AGENTS
        local_event_rewards = [0.0 for _ in range(NUM_AGENTS)]
        ignored_due_to_transit = [False for _ in range(NUM_AGENTS)]
        move_started = [False for _ in range(NUM_AGENTS)]
        arrived_this_step = [False for _ in range(NUM_AGENTS)]

        # ---------------------------------------------------------------------
        # 1) Advance ongoing transit first
        # ---------------------------------------------------------------------
        for agent_id, agent_state in enumerate(self.agent_states):
            if agent_state.in_transit:
                agent_state.advance_transit()
                if not agent_state.in_transit:
                    arrived_this_step[agent_id] = True

        # ---------------------------------------------------------------------
        # 2) Process agent actions
        # ---------------------------------------------------------------------
        for agent_id, action in enumerate(actions):
            agent_state = self.agent_states[agent_id]

            if agent_state.in_transit:
                ignored_due_to_transit[agent_id] = True
                continue

            reward, result, did_start_move = self._process_single_action(
                agent_state=agent_state,
                action=int(action),
            )
            local_event_rewards[agent_id] = reward
            action_results[agent_id] = result
            move_started[agent_id] = did_start_move

        # ---------------------------------------------------------------------
        # 3) Decay long-memory suspicion scores
        # ---------------------------------------------------------------------
        self.network.decay_all_long_memory()

        # ---------------------------------------------------------------------
        # 4) Run attacker
        # ---------------------------------------------------------------------
        pre_damage = self.attacker.server_damage
        newly_infected = self.attacker.step(
            self.network,
            self.traffic_manager,
            self.current_timestep,
            self.rng,
        )
        self.last_newly_infected = list(newly_infected)
        self.last_server_damage_delta = self.attacker.server_damage - pre_damage

        # ---------------------------------------------------------------------
        # 5) Generate legitimate traffic
        # ---------------------------------------------------------------------
        self.traffic_manager.generate_normal_traffic(
            self.network,
            self.current_timestep,
            self.rng,
        )

        # ---------------------------------------------------------------------
        # 6) Prune old traffic records
        # ---------------------------------------------------------------------
        self.traffic_manager.prune_old_records(self.current_timestep)

        # ---------------------------------------------------------------------
        # 7) Check termination / truncation
        # ---------------------------------------------------------------------
        next_timestep = self.current_timestep + 1

        terminated = False
        truncated = False
        terminal_reward = 0.0

        # If both ever somehow occur together, attacker win takes precedence.
        if self.network.is_server_compromised(
            self.attacker.server_damage,
            SERVER_DAMAGE_THRESHOLD,
        ):
            terminated = True
            terminal_reward = REWARD_SERVER_COMPROMISED
        elif self.network.all_infections_quarantined():
            terminated = True
            terminal_reward = REWARD_ALL_QUARANTINED
        elif next_timestep >= MAX_TIMESTEPS:
            truncated = True
            terminal_reward = REWARD_SURVIVED

        # ---------------------------------------------------------------------
        # 8) Compute rewards
        # ---------------------------------------------------------------------
        shared_reward = self._compute_shared_reward(
            num_new_infections=len(newly_infected),
            server_damage_delta=self.last_server_damage_delta,
        )
        self.last_shared_reward = shared_reward

        rewards = [
            float(shared_reward + local_event_rewards[i] + terminal_reward)
            for i in range(NUM_AGENTS)
        ]
        self.last_event_rewards = list(local_event_rewards)

        # Update timestep after this step is fully accounted for.
        self.current_timestep = next_timestep
        self.done = terminated or truncated

        # ---------------------------------------------------------------------
        # 9) Build observations
        # ---------------------------------------------------------------------
        observations = self._build_all_observations()
        terminated_list = [terminated for _ in range(NUM_AGENTS)]
        truncated_list = [truncated for _ in range(NUM_AGENTS)]

        infos = [
            self._build_info_for_agent(
                agent_id=i,
                action_taken=actions[i],
                action_result=action_results[i],
                local_event_reward=local_event_rewards[i],
                shared_reward=shared_reward,
                terminal_reward=terminal_reward,
                ignored_due_to_transit=ignored_due_to_transit[i],
                move_started=move_started[i],
                arrived_this_step=arrived_this_step[i],
            )
            for i in range(NUM_AGENTS)
        ]

        return observations, rewards, terminated_list, truncated_list, infos

    # -------------------------------------------------------------------------
    # Action processing
    # -------------------------------------------------------------------------

    def _process_single_action(
        self,
        agent_state: AgentRuntimeState,
        action: int,
    ) -> Tuple[float, object, bool]:
        """
        Process one action for one non-transit agent.

        Returns:
        - local_event_reward: float
        - action_result: object for info/debugging
        - did_start_move: bool
        """
        if action == ACTION_OBSERVE:
            return 0.0, {"type": "observe"}, False

        # Move actions: 1..18 -> host 0..17
        if ACTION_MOVE_BASE <= action <= ACTION_MOVE_LAST:
            target_host = action - ACTION_MOVE_BASE

            if target_host == agent_state.current_host:
                return 0.0, {
                    "type": "move_noop_same_host",
                    "target_host": target_host,
                }, False

            if self.network.are_same_subnet(agent_state.current_host, target_host):
                travel_time = MOVE_WITHIN_SUBNET_COST
                reward = REWARD_MOVE_WITHIN_SUBNET
            else:
                travel_time = MOVE_CROSS_SUBNET_COST
                reward = REWARD_MOVE_CROSS_SUBNET

            agent_state.begin_transit(target_host, travel_time)
            return reward, {
                "type": "move_started",
                "target_host": target_host,
                "travel_time": travel_time,
            }, True

        current_host = agent_state.current_host

        if action == ACTION_BLOCK:
            result = self.network.apply_block(current_host)
            return self._event_reward_from_containment_result(result), result, False

        if action == ACTION_QUARANTINE:
            result = self.network.apply_quarantine(current_host)
            return self._event_reward_from_containment_result(result), result, False

        if action == ACTION_UNBLOCK:
            result = self.network.apply_unblock(current_host)
            return self._event_reward_from_containment_result(result), result, False

        raise ValueError(f"Unsupported action {action}")

    def _event_reward_from_containment_result(self, result) -> float:
        """
        Translate one containment action result into the acting agent's
        local event reward.
        """
        if not result.changed:
            return 0.0

        if result.action == "block":
            if result.was_infected and result.old_state == CONTAINMENT_NONE:
                return REWARD_CORRECT_BLOCK
            if result.was_clean and result.old_state == CONTAINMENT_NONE:
                return REWARD_FALSE_BLOCK
            return 0.0

        if result.action == "quarantine":
            if result.was_infected and result.old_state == CONTAINMENT_NONE:
                return REWARD_CORRECT_QUARANTINE
            if result.was_infected and result.old_state == CONTAINMENT_BLOCKED:
                return REWARD_QUARANTINE_UPGRADE
            if result.was_clean and result.old_state == CONTAINMENT_NONE:
                return REWARD_FALSE_QUARANTINE
            if result.was_clean and result.old_state == CONTAINMENT_BLOCKED:
                return REWARD_FALSE_QUARANTINE_UPGRADE
            return 0.0

        if result.action == "unblock":
            if result.was_clean:
                return REWARD_CORRECT_UNBLOCK
            if result.was_infected:
                return REWARD_BAD_UNBLOCK
            return 0.0

        return 0.0

    # -------------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------------

    def _compute_shared_reward(
        self,
        num_new_infections: int,
        server_damage_delta: float,
    ) -> float:
        counts = self.network.count_by_status()

        reward = 0.0
        reward += REWARD_HEALTHY_HOST * counts["clean_total"]
        reward += REWARD_INFECTED_UNCONTAINED * counts["infected_uncontained"]
        reward += REWARD_INFECTED_BLOCKED * counts["infected_blocked"]
        reward += REWARD_INFECTED_QUARANTINED * counts["infected_quarantined"]
        reward += REWARD_FALSE_BLOCK_PER_STEP * counts["clean_blocked"]
        reward += REWARD_FALSE_QUARANTINE_PER_STEP * counts["clean_quarantined"]
        reward += REWARD_AGENT_OVERLAP * self._count_overlap_pairs()
        reward += REWARD_NEW_INFECTION * num_new_infections
        reward += REWARD_SERVER_DAMAGE * server_damage_delta
        return float(reward)

    def _count_overlap_pairs(self) -> int:
        """
        Count overlap pairs among agents that are NOT in transit.

        If 2 agents share a host -> 1 pair
        If 3 agents share a host -> 3 pairs
        """
        occupancy: Dict[int, int] = {}

        for agent_state in self.agent_states:
            if agent_state.in_transit:
                continue
            occupancy[agent_state.current_host] = (
                occupancy.get(agent_state.current_host, 0) + 1
            )

        pairs = 0
        for count in occupancy.values():
            if count >= 2:
                pairs += (count * (count - 1)) // 2
        return pairs

    # -------------------------------------------------------------------------
    # Observation building
    # -------------------------------------------------------------------------

    def _build_all_observations(self) -> List[np.ndarray]:
        return [self._build_observation_for_agent(i) for i in range(NUM_AGENTS)]

    def _build_observation_for_agent(self, agent_id: int) -> np.ndarray:
        """
        Build one 77-dim observation.

        In-transit convention:
        - traffic features 0-15 are zeroed
        - self position remains at last occupied host
        - self in-transit flag is 1
        """
        obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        agent_state = self.agent_states[agent_id]

        # 0-15: traffic features
        if not agent_state.in_transit:
            obs[:NUM_TRAFFIC_FEATURES] = self.traffic_manager.compute_features(
                agent_state.current_host,
                self.network,
            )

        # 16-33: self position one-hot
        obs[FEAT_SELF_POS_START + agent_state.current_host] = 1.0

        # 34-69: other agent positions one-hot
        other_ids = [i for i in range(NUM_AGENTS) if i != agent_id]
        other_1 = self.agent_states[other_ids[0]]
        other_2 = self.agent_states[other_ids[1]]

        obs[FEAT_OTHER_AGENT_1_POS_START + other_1.current_host] = 1.0
        obs[FEAT_OTHER_AGENT_2_POS_START + other_2.current_host] = 1.0

        # 70-76: global scalars / flags
        counts = self.network.count_by_status()
        obs[FEAT_FRACTION_QUARANTINED] = counts["quarantined_total"] / float(NUM_HOSTS)
        obs[FEAT_FRACTION_BLOCKED] = counts["blocked_total"] / float(NUM_HOSTS)
        obs[FEAT_SERVER_DAMAGE] = (
            self.attacker.server_damage / float(SERVER_DAMAGE_THRESHOLD)
        )
        obs[FEAT_EPISODE_PROGRESS] = min(
            self.current_timestep / float(MAX_TIMESTEPS),
            1.0,
        )

        current_host_obj = self.network.get_host(agent_state.current_host)
        obs[FEAT_CURRENT_HOST_BLOCKED] = 1.0 if current_host_obj.is_blocked else 0.0
        obs[FEAT_CURRENT_HOST_QUARANTINED] = (
            1.0 if current_host_obj.is_quarantined else 0.0
        )
        obs[FEAT_IN_TRANSIT] = 1.0 if agent_state.in_transit else 0.0

        np.clip(obs, 0.0, 1.0, out=obs)
        return obs

    # -------------------------------------------------------------------------
    # Info helpers
    # -------------------------------------------------------------------------

    def _build_info_for_agent(
        self,
        agent_id: int,
        action_taken,
        action_result,
        local_event_reward: float,
        shared_reward: float,
        terminal_reward: float,
        ignored_due_to_transit: bool,
        move_started: bool,
        arrived_this_step: bool,
    ) -> Dict[str, object]:
        agent_state = self.agent_states[agent_id]
        counts = self.network.count_by_status()

        return {
            "agent_id": agent_id,
            "timestep": self.current_timestep,
            "current_host": agent_state.current_host,
            "in_transit": agent_state.in_transit,
            "transit_target": agent_state.transit_target,
            "transit_remaining": agent_state.transit_remaining,
            "action_taken": None if action_taken is None else int(action_taken),
            "action_result": self._serialize_action_result(action_result),
            "action_ignored_due_to_transit": ignored_due_to_transit,
            "move_started": move_started,
            "arrived_this_step": arrived_this_step,
            "local_event_reward": float(local_event_reward),
            "shared_reward": float(shared_reward),
            "terminal_reward": float(terminal_reward),
            "server_damage": float(self.attacker.server_damage),
            "server_damage_delta": float(self.last_server_damage_delta),
            "newly_infected_this_step": list(self.last_newly_infected),
            "num_new_infections_this_step": len(self.last_newly_infected),
            "infected_total": counts["infected_total"],
            "infected_uncontained": counts["infected_uncontained"],
            "infected_blocked": counts["infected_blocked"],
            "infected_quarantined": counts["infected_quarantined"],
            "clean_total": counts["clean_total"],
            "clean_blocked": counts["clean_blocked"],
            "clean_quarantined": counts["clean_quarantined"],
            "blocked_total": counts["blocked_total"],
            "quarantined_total": counts["quarantined_total"],
            "overlap_pairs": self._count_overlap_pairs(),
            "all_infections_quarantined": self.network.all_infections_quarantined(),
            "server_compromised": self.network.is_server_compromised(
                self.attacker.server_damage,
                SERVER_DAMAGE_THRESHOLD,
            ),
        }

    def _serialize_action_result(self, action_result):
        if action_result is None:
            return None
        if isinstance(action_result, dict):
            return dict(action_result)
        if hasattr(action_result, "__dict__"):
            return dict(action_result.__dict__)
        return str(action_result)

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def _validate_actions(self, actions: Sequence[int]) -> None:
        if len(actions) != NUM_AGENTS:
            raise ValueError(
                f"step(actions) expects {NUM_AGENTS} actions, got {len(actions)}"
            )

        for action in actions:
            action_int = int(action)
            if not (0 <= action_int < NUM_ACTIONS):
                raise ValueError(
                    f"Each action must be in [0, {NUM_ACTIONS - 1}], got {action_int}"
                )

    def _validate_initial_agent_positions(
        self,
        positions: Optional[Sequence[int]],
    ) -> Tuple[int, int, int]:
        if positions is None:
            positions = self.DEFAULT_INITIAL_AGENT_POSITIONS

        if len(positions) != NUM_AGENTS:
            raise ValueError(
                f"initial_agent_positions must have length {NUM_AGENTS}, got {len(positions)}"
            )

        pos_list = [int(x) for x in positions]
        if len(set(pos_list)) != NUM_AGENTS:
            raise ValueError("initial_agent_positions must be unique")

        for host_id in pos_list:
            if not (0 <= host_id < NUM_HOSTS):
                raise ValueError(
                    f"Agent start host must be in [0, {NUM_HOSTS - 1}], got {host_id}"
                )

        return tuple(pos_list)

    # -------------------------------------------------------------------------
    # Optional debug helper
    # -------------------------------------------------------------------------

    def get_state_summary(self) -> Dict[str, object]:
        """
        Small snapshot for quick debugging / printing.
        """
        counts = self.network.count_by_status()
        return {
            "timestep": self.current_timestep,
            "server_damage": float(self.attacker.server_damage),
            "counts": counts,
            "agent_states": [
                {
                    "agent_id": agent_state.agent_id,
                    "current_host": agent_state.current_host,
                    "in_transit": agent_state.in_transit,
                    "transit_target": agent_state.transit_target,
                    "transit_remaining": agent_state.transit_remaining,
                }
                for agent_state in self.agent_states
            ],
            "infected_hosts": self.network.get_all_infected_host_ids(),
        }