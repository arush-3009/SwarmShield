"""
SwarmShield Network Topology
============================

This file defines the office network structure and the per-host state
that persists across an episode.

What belongs here:
- Fixed topology (host identity, subnet membership, server host)
- Mutable host state (infection, containment, vulnerability, timers)
- Decayed long-memory suspicion features stored per host
- Helper methods for topology queries
- Canonical connection-success logic from the environment spec
- Canonical containment-action application (including server no-op rules)
- Win/loss related host-state queries
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from env.config import (
    NUM_HOSTS,
    NUM_SUBNETS,
    SUBNET_HOSTS,
    SUBNET_NAMES,
    HOST_TO_SUBNET,
    HOST_NAMES,
    REGULAR_HOST_IDS,
    SERVER_HOST_ID,
    EXTERNAL_C2_ID,
    VULN_RANGE_REGULAR,
    CONTAINMENT_NONE,
    CONTAINMENT_BLOCKED,
    CONTAINMENT_QUARANTINED,
    SUSPICIOUS_DECAY_FACTOR,
)


# =============================================================================
# RESULT OBJECTS
# =============================================================================


@dataclass(frozen=True)
class ConnectionDecision:
    """
    Result of checking whether a connection attempt succeeds or fails
    under the canonical containment rules.

    success:
        True if the connection completed.
        False if containment blocked it.
        Non-firewall failures (for example exploit failure after arrival)
        are handled later by attacker/traffic code, not here.

    reached_destination:
        True if the packet reached the internal destination host.
        This controls whether the destination gets an incoming record.

    blocked_by_firewall:
        True only when containment rules stopped the packet.

    reason:
        Short debug string describing what happened.
    """

    success: bool
    reached_destination: bool
    blocked_by_firewall: bool
    reason: str


@dataclass(frozen=True)
class ContainmentActionResult:
    """
    Result of applying one containment action to one host.

    This is used later by reward logic to decide whether the action was:
    - a correct block
    - a correct quarantine
    - a quarantine upgrade
    - a false positive
    - a bad unblock
    - or a no-op
    """

    action: str
    host_id: int
    changed: bool
    old_state: int
    new_state: int
    was_infected: bool
    was_clean: bool
    noop_reason: Optional[str] = None


# =============================================================================
# HOST
# =============================================================================


class Host:
    """
    One computer on the network.

    Fixed identity:
    - host_id
    - name
    - subnet_id
    - is_server

    Mutable episode state:
    - vulnerability
    - infected
    - containment_state
    - timestep_infected
    - beacon_timer

    Decayed long-memory suspicion counters:
    - decayed_failed_outgoing
    - decayed_unique_peers
    - decayed_incoming_scans
    - decayed_server_contacts

    Important rule:
    infection and containment are independent axes.
    Infection never changes containment.
    Only defender actions change containment.
    """

    def __init__(self, host_id: int):
        self.host_id = host_id
        self.name = HOST_NAMES[host_id]
        self.subnet_id = HOST_TO_SUBNET[host_id]
        self.is_server = host_id == SERVER_HOST_ID

        self.vulnerability = 0.0
        self.infected = False
        self.containment_state = CONTAINMENT_NONE
        self.timestep_infected = -1
        self.beacon_timer = 0

        self.decayed_failed_outgoing = 0.0
        self.decayed_unique_peers = 0.0
        self.decayed_incoming_scans = 0.0
        self.decayed_server_contacts = 0.0

    # -------------------------------------------------------------------------
    # Reset / lifecycle
    # -------------------------------------------------------------------------

    def reset(self, rng) -> None:
        """
        Reset all mutable state for a fresh episode.

        Regular hosts get a vulnerability in VULN_RANGE_REGULAR.
        The file server is never a scan target, so its vulnerability is 0.
        """
        if self.is_server:
            self.vulnerability = 0.0
        else:
            low, high = VULN_RANGE_REGULAR
            self.vulnerability = float(rng.uniform(low, high))

        self.infected = False
        self.containment_state = CONTAINMENT_NONE
        self.timestep_infected = -1

        # 0 means beacon is due immediately on the next attacker pass.
        self.beacon_timer = 0

        self.reset_long_memory()

    def reset_long_memory(self) -> None:
        """Clear all four decayed long-memory counters."""
        self.decayed_failed_outgoing = 0.0
        self.decayed_unique_peers = 0.0
        self.decayed_incoming_scans = 0.0
        self.decayed_server_contacts = 0.0

    def infect(self, current_timestep: int) -> bool:
        """
        Mark this host as infected.

        Returns True only if the host changed from clean to infected.
        Returns False if it was already infected.

        Infection never changes containment.
        """
        if self.infected:
            return False

        self.infected = True
        self.timestep_infected = current_timestep
        self.beacon_timer = 0
        return True

    def infection_age(self, current_timestep: int) -> int:
        """
        Number of timesteps since infection.
        Returns -1 if the host is not infected.
        """
        if not self.infected:
            return -1
        return current_timestep - self.timestep_infected

    # -------------------------------------------------------------------------
    # Long-memory suspicion counters
    # -------------------------------------------------------------------------

    def decay_long_memory(self, decay_factor: float = SUSPICIOUS_DECAY_FACTOR) -> None:
        """Apply exponential decay to all four counters."""
        self.decayed_failed_outgoing *= decay_factor
        self.decayed_unique_peers *= decay_factor
        self.decayed_incoming_scans *= decay_factor
        self.decayed_server_contacts *= decay_factor

    def note_failed_outgoing(self, amount: float = 1.0) -> None:
        self.decayed_failed_outgoing += amount

    def note_unique_peer_contact(self, amount: float = 1.0) -> None:
        self.decayed_unique_peers += amount

    def note_incoming_scan(self, amount: float = 1.0) -> None:
        self.decayed_incoming_scans += amount

    def note_server_contact(self, amount: float = 1.0) -> None:
        self.decayed_server_contacts += amount

    # -------------------------------------------------------------------------
    # Low-level containment setters
    # -------------------------------------------------------------------------

    def _set_blocked(self) -> None:
        self.containment_state = CONTAINMENT_BLOCKED

    def _set_quarantined(self) -> None:
        self.containment_state = CONTAINMENT_QUARANTINED

    def _set_uncontained(self) -> None:
        self.containment_state = CONTAINMENT_NONE

    # -------------------------------------------------------------------------
    # Read-only state helpers
    # -------------------------------------------------------------------------

    @property
    def is_clean(self) -> bool:
        return not self.infected

    @property
    def is_infected(self) -> bool:
        return self.infected

    @property
    def is_uncontained(self) -> bool:
        return self.containment_state == CONTAINMENT_NONE

    @property
    def is_blocked(self) -> bool:
        return self.containment_state == CONTAINMENT_BLOCKED

    @property
    def is_quarantined(self) -> bool:
        return self.containment_state == CONTAINMENT_QUARANTINED

    @property
    def is_contained(self) -> bool:
        return self.containment_state != CONTAINMENT_NONE

    @property
    def is_infected_uncontained(self) -> bool:
        return self.infected and self.containment_state == CONTAINMENT_NONE

    @property
    def is_infected_blocked(self) -> bool:
        return self.infected and self.containment_state == CONTAINMENT_BLOCKED

    @property
    def is_infected_quarantined(self) -> bool:
        return self.infected and self.containment_state == CONTAINMENT_QUARANTINED

    @property
    def status_label(self) -> str:
        if self.is_infected_blocked:
            return "INFECTED+BLOCKED"
        if self.is_infected_quarantined:
            return "INFECTED+QUAR"
        if self.is_clean and self.is_blocked:
            return "CLEAN+BLOCKED"
        if self.is_clean and self.is_quarantined:
            return "CLEAN+QUAR"
        if self.is_infected:
            return "INFECTED"
        return "CLEAN"


# =============================================================================
# NETWORK
# =============================================================================


class Network:
    """
    The full office network.

    Responsibilities:
    - own all Host objects
    - provide topology helpers
    - centralize containment action semantics
    - centralize connection-success logic
    - provide counts for rewards and observations
    - seed initial infections
    - decay host long-memory each timestep
    """

    def __init__(self):
        self.hosts: List[Host] = []
        for host_id in range(NUM_HOSTS):
            self.hosts.append(Host(host_id))

    # -------------------------------------------------------------------------
    # Basic access
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.hosts)

    def __iter__(self):
        return iter(self.hosts)

    def reset(self, rng) -> None:
        for host in self.hosts:
            host.reset(rng)

    def get_host(self, host_id: int) -> Host:
        self._validate_internal_host_id(host_id)
        return self.hosts[host_id]

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def _validate_internal_host_id(self, host_id: int) -> None:
        if not (0 <= host_id < NUM_HOSTS):
            raise ValueError(
                f"Internal host_id must be in [0, {NUM_HOSTS - 1}], got {host_id}"
            )

    def _validate_dest_id(self, dest_id: int) -> None:
        if dest_id != EXTERNAL_C2_ID and not (0 <= dest_id < NUM_HOSTS):
            raise ValueError(
                f"dest_id must be in [0, {NUM_HOSTS - 1}] or "
                f"EXTERNAL_C2_ID={EXTERNAL_C2_ID}, got {dest_id}"
            )

    def is_external_dest(self, dest_id: int) -> bool:
        return dest_id == EXTERNAL_C2_ID

    # -------------------------------------------------------------------------
    # Topology helpers
    # -------------------------------------------------------------------------

    def are_same_subnet(self, host_a_id: int, host_b_id: int) -> bool:
        """
        True only if both are internal hosts in the same subnet.
        External C2 is outside all subnets.
        """
        self._validate_internal_host_id(host_a_id)

        if self.is_external_dest(host_b_id):
            return False

        self._validate_internal_host_id(host_b_id)
        return HOST_TO_SUBNET[host_a_id] == HOST_TO_SUBNET[host_b_id]

    def get_subnet_host_ids(self, subnet_id: int) -> List[int]:
        return list(SUBNET_HOSTS[subnet_id])

    def get_subnet_hosts(self, subnet_id: int) -> List[Host]:
        result: List[Host] = []
        for host_id in SUBNET_HOSTS[subnet_id]:
            result.append(self.hosts[host_id])
        return result

    def get_same_subnet_host_ids(self, host_id: int, include_self: bool = False) -> List[int]:
        self._validate_internal_host_id(host_id)
        subnet_id = HOST_TO_SUBNET[host_id]

        result: List[int] = []
        for other_id in SUBNET_HOSTS[subnet_id]:
            if not include_self and other_id == host_id:
                continue
            result.append(other_id)
        return result

    def get_same_subnet_hosts(self, host_id: int, include_self: bool = False) -> List[Host]:
        ids = self.get_same_subnet_host_ids(host_id, include_self=include_self)

        result: List[Host] = []
        for other_id in ids:
            result.append(self.hosts[other_id])
        return result

    def get_cross_subnet_host_ids(self, host_id: int, include_server: bool = True) -> List[int]:
        self._validate_internal_host_id(host_id)
        my_subnet = HOST_TO_SUBNET[host_id]

        result: List[int] = []
        for other_id in range(NUM_HOSTS):
            if HOST_TO_SUBNET[other_id] == my_subnet:
                continue
            if not include_server and other_id == SERVER_HOST_ID:
                continue
            result.append(other_id)
        return result

    def get_cross_subnet_hosts(self, host_id: int, include_server: bool = True) -> List[Host]:
        ids = self.get_cross_subnet_host_ids(host_id, include_server=include_server)

        result: List[Host] = []
        for other_id in ids:
            result.append(self.hosts[other_id])
        return result

    def get_scan_target_ids_same_subnet(self, host_id: int) -> List[int]:
        """
        Same-subnet scan targets exclude self.
        The file server cannot appear here because it is alone in subnet 5.
        """
        return self.get_same_subnet_host_ids(host_id, include_self=False)

    def get_scan_target_ids_cross_subnet(self, host_id: int) -> List[int]:
        """
        Cross-subnet scan targets exclude the file server by spec.
        """
        return self.get_cross_subnet_host_ids(host_id, include_server=False)

    # -------------------------------------------------------------------------
    # Initial infection seeding
    # -------------------------------------------------------------------------

    def seed_initial_infections(self, rng, num_infections: int, current_timestep: int = 0) -> List[int]:
        """
        Infect random regular (non-server) hosts at episode start.

        Supports either:
        - NumPy-style RNG with choice(..., replace=False)
        - Python random.Random-style RNG with sample(...)
        """
        if num_infections < 1:
            raise ValueError(f"num_infections must be >= 1, got {num_infections}")
        if num_infections > len(REGULAR_HOST_IDS):
            raise ValueError(
                f"num_infections={num_infections} exceeds "
                f"number of regular hosts={len(REGULAR_HOST_IDS)}"
            )

        if hasattr(rng, "choice"):
            chosen = rng.choice(REGULAR_HOST_IDS, size=num_infections, replace=False)
            if num_infections == 1:
                chosen_list = [int(chosen[0])]
            else:
                chosen_list = []
                for host_id in chosen:
                    chosen_list.append(int(host_id))
        elif hasattr(rng, "sample"):
            chosen_list = []
            for host_id in rng.sample(REGULAR_HOST_IDS, k=num_infections):
                chosen_list.append(int(host_id))
        else:
            raise TypeError("rng must support choice(...) or sample(...)")

        newly_infected: List[int] = []
        for host_id in chosen_list:
            changed = self.hosts[host_id].infect(current_timestep)
            if changed:
                newly_infected.append(host_id)

        return newly_infected

    # -------------------------------------------------------------------------
    # Canonical connection-success logic
    # -------------------------------------------------------------------------

    def decide_connection(self, source_id: int, dest_id: int) -> ConnectionDecision:
        """
        Apply the canonical Section 6 containment logic.

        Order:
        1. source quarantined -> fail
        2. source blocked and dest is cross-subnet or external -> fail
        3. external destination that passed sender checks -> success, no incoming record
        4. internal destination quarantined -> fail
        5. internal destination blocked and sender is cross-subnet -> fail
        6. otherwise -> success
        """
        self._validate_internal_host_id(source_id)
        self._validate_dest_id(dest_id)

        source = self.hosts[source_id]

        if source.is_quarantined:
            return ConnectionDecision(
                success=False,
                reached_destination=False,
                blocked_by_firewall=True,
                reason="source_quarantined",
            )

        if source.is_blocked:
            if self.is_external_dest(dest_id):
                return ConnectionDecision(
                    success=False,
                    reached_destination=False,
                    blocked_by_firewall=True,
                    reason="source_blocked_external",
                )

            if not self.are_same_subnet(source_id, dest_id):
                return ConnectionDecision(
                    success=False,
                    reached_destination=False,
                    blocked_by_firewall=True,
                    reason="source_blocked_cross_subnet",
                )

        if self.is_external_dest(dest_id):
            return ConnectionDecision(
                success=True,
                reached_destination=False,
                blocked_by_firewall=False,
                reason="success_external",
            )

        dest = self.hosts[dest_id]

        if dest.is_quarantined:
            return ConnectionDecision(
                success=False,
                reached_destination=False,
                blocked_by_firewall=True,
                reason="dest_quarantined",
            )

        if dest.is_blocked and not self.are_same_subnet(source_id, dest_id):
            return ConnectionDecision(
                success=False,
                reached_destination=False,
                blocked_by_firewall=True,
                reason="dest_blocked_cross_subnet",
            )

        return ConnectionDecision(
            success=True,
            reached_destination=True,
            blocked_by_firewall=False,
            reason="success_internal",
        )

    # -------------------------------------------------------------------------
    # Canonical containment actions
    # -------------------------------------------------------------------------

    def apply_block(self, host_id: int) -> ContainmentActionResult:
        """
        Apply BLOCK to a host.

        No-op cases:
        - server host
        - already blocked
        - already quarantined (no downgrade)
        """
        host = self.get_host(host_id)
        old_state = host.containment_state

        if host.is_server:
            return ContainmentActionResult(
                action="block",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="server_noop",
            )

        if host.is_blocked:
            return ContainmentActionResult(
                action="block",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="already_blocked",
            )

        if host.is_quarantined:
            return ContainmentActionResult(
                action="block",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="already_quarantined_no_downgrade",
            )

        host._set_blocked()
        return ContainmentActionResult(
            action="block",
            host_id=host_id,
            changed=True,
            old_state=old_state,
            new_state=host.containment_state,
            was_infected=host.is_infected,
            was_clean=host.is_clean,
            noop_reason=None,
        )

    def apply_quarantine(self, host_id: int) -> ContainmentActionResult:
        """
        Apply QUARANTINE to a host.

        No-op cases:
        - server host
        - already quarantined
        """
        host = self.get_host(host_id)
        old_state = host.containment_state

        if host.is_server:
            return ContainmentActionResult(
                action="quarantine",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="server_noop",
            )

        if host.is_quarantined:
            return ContainmentActionResult(
                action="quarantine",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="already_quarantined",
            )

        host._set_quarantined()
        return ContainmentActionResult(
            action="quarantine",
            host_id=host_id,
            changed=True,
            old_state=old_state,
            new_state=host.containment_state,
            was_infected=host.is_infected,
            was_clean=host.is_clean,
            noop_reason=None,
        )

    def apply_unblock(self, host_id: int) -> ContainmentActionResult:
        """
        Remove containment from a host.

        No-op cases:
        - server host
        - already uncontained
        """
        host = self.get_host(host_id)
        old_state = host.containment_state

        if host.is_server:
            return ContainmentActionResult(
                action="unblock",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="server_noop",
            )

        if host.is_uncontained:
            return ContainmentActionResult(
                action="unblock",
                host_id=host_id,
                changed=False,
                old_state=old_state,
                new_state=old_state,
                was_infected=host.is_infected,
                was_clean=host.is_clean,
                noop_reason="already_uncontained",
            )

        host._set_uncontained()
        return ContainmentActionResult(
            action="unblock",
            host_id=host_id,
            changed=True,
            old_state=old_state,
            new_state=host.containment_state,
            was_infected=host.is_infected,
            was_clean=host.is_clean,
            noop_reason=None,
        )

    # -------------------------------------------------------------------------
    # Infection / attack-phase queries
    # -------------------------------------------------------------------------

    def get_all_infected_hosts(self) -> List[Host]:
        result: List[Host] = []
        for host in self.hosts:
            if host.is_infected:
                result.append(host)
        return result

    def get_all_infected_host_ids(self) -> List[int]:
        result: List[int] = []
        for host in self.hosts:
            if host.is_infected:
                result.append(host.host_id)
        return result

    def get_infected_uncontained_hosts(self) -> List[Host]:
        result: List[Host] = []
        for host in self.hosts:
            if host.is_infected_uncontained:
                result.append(host)
        return result

    def get_infected_uncontained_host_ids(self) -> List[int]:
        result: List[int] = []
        for host in self.hosts:
            if host.is_infected_uncontained:
                result.append(host.host_id)
        return result

    def count_infected_uncontained(self) -> int:
        count = 0
        for host in self.hosts:
            if host.is_infected_uncontained:
                count += 1
        return count

    def count_active_uncontained_infections(self) -> int:
        """
        Alias with a name that matches the spec wording.
        """
        return self.count_infected_uncontained()

    # -------------------------------------------------------------------------
    # Win/loss checks
    # -------------------------------------------------------------------------

    def all_infections_quarantined(self) -> bool:
        """
        Defender win condition:
        - at least one infection exists
        - every infected host is quarantined
        """
        infected_hosts = self.get_all_infected_hosts()

        if len(infected_hosts) == 0:
            return False

        for host in infected_hosts:
            if not host.is_quarantined:
                return False

        return True

    def is_server_compromised(self, server_damage: float, threshold: float) -> bool:
        return server_damage >= threshold

    # -------------------------------------------------------------------------
    # Counts for rewards / observations
    # -------------------------------------------------------------------------

    def count_by_status(self) -> Dict[str, int]:
        counts = {
            "clean_total": 0,
            "clean_uncontained": 0,
            "clean_blocked": 0,
            "clean_quarantined": 0,
            "infected_total": 0,
            "infected_uncontained": 0,
            "infected_blocked": 0,
            "infected_quarantined": 0,
            "blocked_total": 0,
            "quarantined_total": 0,
        }

        for host in self.hosts:
            if host.is_clean:
                counts["clean_total"] += 1

                if host.is_uncontained:
                    counts["clean_uncontained"] += 1
                elif host.is_blocked:
                    counts["clean_blocked"] += 1
                elif host.is_quarantined:
                    counts["clean_quarantined"] += 1
            else:
                counts["infected_total"] += 1

                if host.is_uncontained:
                    counts["infected_uncontained"] += 1
                elif host.is_blocked:
                    counts["infected_blocked"] += 1
                elif host.is_quarantined:
                    counts["infected_quarantined"] += 1

            if host.is_blocked:
                counts["blocked_total"] += 1
            elif host.is_quarantined:
                counts["quarantined_total"] += 1

        return counts

    # -------------------------------------------------------------------------
    # Long-memory decay
    # -------------------------------------------------------------------------

    def decay_all_long_memory(self, decay_factor: float = SUSPICIOUS_DECAY_FACTOR) -> None:
        for host in self.hosts:
            host.decay_long_memory(decay_factor)

    # -------------------------------------------------------------------------
    # Debug formatting
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        lines: List[str] = []

        for subnet_id in range(NUM_SUBNETS):
            lines.append(f"\n--- {SUBNET_NAMES[subnet_id]} (Subnet {subnet_id}) ---")

            for host in self.get_subnet_hosts(subnet_id):
                vuln_str = f"vuln={host.vulnerability:.3f}"

                if host.is_infected:
                    infected_at = f"infected_at={host.timestep_infected}"
                else:
                    infected_at = "infected_at=-"

                lines.append(
                    f"  Host {host.host_id:2d} ({host.name:12s}): "
                    f"{host.status_label:18s} {vuln_str} {infected_at}"
                )

        return "\n".join(lines)


# =============================================================================
# SANITY CHECKS
# =============================================================================

assert len(HOST_NAMES) == NUM_HOSTS
assert len(SUBNET_NAMES) == NUM_SUBNETS
assert set(REGULAR_HOST_IDS) == set(range(NUM_HOSTS)) - {SERVER_HOST_ID}
assert HOST_TO_SUBNET[SERVER_HOST_ID] == 5
assert SUBNET_HOSTS[5] == [SERVER_HOST_ID]