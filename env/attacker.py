"""
SwarmShield Attacker: Scripted Botnet Logic
===========================================

The attacker is not learned. It follows a fixed script:

1. Beaconing
   Every infected host periodically sends small messages to the external
   C2 server (dest_id = -1). The first beacon fires immediately when the
   host is infected. After that, the beacon timer resets to random(2, 4)
   steps.

2. Scanning
   After SCAN_DELAY timesteps since infection, an infected host starts
   scanning for new targets. It sends 3-5 probes per step, 80% within
   its own subnet and 20% cross-subnet. The file server is never a
   scan target. If a probe reaches a clean host, infection succeeds
   with probability = target vulnerability.

3. Server Attack
   Once the number of infected+uncontained hosts reaches
   ATTACK_INFECTION_THRESHOLD, every infected host attempts attack
   connections to the file server. Uncontained hosts succeed.
   Blocked and quarantined hosts still attempt them, but containment
   makes them fail.

Important:
- ALL infected hosts attempt malicious traffic every timestep.
- Containment determines what succeeds via network.decide_connection().
- Failed malicious attempts still generate outgoing traffic records.
- Newly infected hosts do NOT act again later in the same timestep.
  They start their own lifecycle on the next timestep.
"""

from typing import List, Sequence

from env.config import (
    EXTERNAL_C2_ID,
    SERVER_HOST_ID,
    BEACON_INTERVAL_BASE,
    BEACON_JITTER,
    BEACON_PORT,
    BEACON_BYTES_SENT,
    BEACON_BYTES_RECEIVED,
    SCAN_DELAY,
    SCAN_PROBES_PER_STEP,
    SCAN_SAME_SUBNET_PROB,
    SCAN_PROBE_PORT_RANGE,
    SCAN_PROBE_BYTES,
    SCAN_PROBE_RECV_ALREADY_INFECTED,
    INFECTION_PAYLOAD_BYTES_SENT,
    INFECTION_PAYLOAD_BYTES_RECEIVED,
    ATTACK_INFECTION_THRESHOLD,
    ATTACK_CONNECTIONS_PER_STEP,
    ATTACK_DEST_PORT,
    ATTACK_BYTES_PER_CONNECTION,
    SERVER_DAMAGE_PER_CONNECTION,
)


class Attacker:
    """
    Scripted botnet controller.

    Tracks cumulative server damage and runs the three attacker phases
    each timestep.

    Initial infections are NOT handled here.
    Those are created by Network.seed_initial_infections() at reset.
    """

    def __init__(self):
        self.server_damage = 0.0

    def reset(self) -> None:
        """Reset attacker-side episode state."""
        self.server_damage = 0.0

    # -------------------------------------------------------------------------
    # RNG helpers
    # -------------------------------------------------------------------------

    def _rand_int_inclusive(self, rng, low: int, high: int) -> int:
        """
        Sample uniformly from [low, high], inclusive.

        Supports:
        - numpy.random.Generator via integers(...)
        - python random.Random via randint(...)
        - numpy RandomState-like objects via randint(...)
        """
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high + 1))

        if hasattr(rng, "sample"):
            return int(rng.randint(low, high))

        if hasattr(rng, "randint"):
            return int(rng.randint(low, high + 1))

        raise TypeError("rng must support integers(...), randint(...), or sample(...).")

    def _rand_float_01(self, rng) -> float:
        """Sample one float in [0, 1)."""
        if hasattr(rng, "random"):
            return float(rng.random())

        if hasattr(rng, "random_sample"):
            return float(rng.random_sample())

        raise TypeError("rng must support random() or random_sample().")

    def _choice(self, rng, seq: Sequence[int]) -> int:
        """Choose one element from a non-empty sequence."""
        if len(seq) == 0:
            raise ValueError("Cannot choose from an empty sequence.")

        if hasattr(rng, "choice"):
            return int(rng.choice(seq))

        if hasattr(rng, "sample"):
            return int(rng.sample(list(seq), k=1)[0])

        raise TypeError("rng must support choice(...) or sample(...).")

    def _sample_beacon_interval(self, rng) -> int:
        """
        Sample the next beacon interval from [2, 4].

        Config uses:
        BEACON_INTERVAL_BASE = 3
        BEACON_JITTER = 1
        """
        low = BEACON_INTERVAL_BASE - BEACON_JITTER
        high = BEACON_INTERVAL_BASE + BEACON_JITTER
        return max(1, self._rand_int_inclusive(rng, low, high))

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(self, network, traffic_manager, current_timestep: int, rng) -> List[int]:
        """
        Run one timestep of malicious activity.

        Important timestep semantics:
        - We take a snapshot of infected hosts at the START of the step.
        - Only that snapshot acts this timestep.
        - Hosts infected during this step begin acting next timestep.

        Returns:
        - newly_infected: list of host IDs newly infected this timestep
        """
        newly_infected: List[int] = []

        infected_hosts = list(network.get_all_infected_hosts())

        attack_phase_active = (
            network.count_active_uncontained_infections()
            >= ATTACK_INFECTION_THRESHOLD
        )

        for host in infected_hosts:
            self._do_beaconing(
                host=host,
                network=network,
                traffic_manager=traffic_manager,
                current_timestep=current_timestep,
                rng=rng,
            )

            new_from_scan = self._do_scanning(
                host=host,
                network=network,
                traffic_manager=traffic_manager,
                current_timestep=current_timestep,
                rng=rng,
            )
            for host_id in new_from_scan:
                newly_infected.append(host_id)

            if attack_phase_active:
                self._do_server_attack(
                    host=host,
                    network=network,
                    traffic_manager=traffic_manager,
                    current_timestep=current_timestep,
                    rng=rng,
                )

        return newly_infected

    # -------------------------------------------------------------------------
    # Phase 1: Beaconing
    # -------------------------------------------------------------------------

    def _do_beaconing(self, host, network, traffic_manager, current_timestep: int, rng) -> None:
        """
        Process one host's beacon timer and possibly send a C2 beacon.

        Timer semantics:
        - Host.infect() sets beacon_timer = 0.
        - At attacker step, timer decrements first.
        - If timer <= 0, beacon fires now and timer resets to random(2, 4).

        This yields:
        - immediate beacon on the first attacker pass after infection
        - later beacons every 2-4 timesteps
        """
        host.beacon_timer -= 1

        if host.beacon_timer > 0:
            return

        decision = network.decide_connection(host.host_id, EXTERNAL_C2_ID)

        bytes_sent = self._rand_int_inclusive(
            rng,
            BEACON_BYTES_SENT[0],
            BEACON_BYTES_SENT[1],
        )

        if decision.success:
            bytes_received = self._rand_int_inclusive(
                rng,
                BEACON_BYTES_RECEIVED[0],
                BEACON_BYTES_RECEIVED[1],
            )
        else:
            bytes_received = 0

        traffic_manager.record_connection(
            network=network,
            source_id=host.host_id,
            dest_id=EXTERNAL_C2_ID,
            dest_port=BEACON_PORT,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            success=decision.success,
            timestamp=current_timestep,
            is_malicious=True,
            reached_destination=decision.reached_destination,
        )

        host.beacon_timer = self._sample_beacon_interval(rng)

    # -------------------------------------------------------------------------
    # Phase 2: Scanning / lateral spread
    # -------------------------------------------------------------------------

    def _choose_scan_target_id(self, host, network, rng) -> int | None:
        """
        Choose one scan target.

        By spec:
        - 80% same-subnet
        - 20% cross-subnet
        - file server is never a scan target
        """
        same_subnet_ids = network.get_scan_target_ids_same_subnet(host.host_id)
        cross_subnet_ids = network.get_scan_target_ids_cross_subnet(host.host_id)

        if len(same_subnet_ids) == 0 and len(cross_subnet_ids) == 0:
            return None

        choose_same = self._rand_float_01(rng) < SCAN_SAME_SUBNET_PROB

        if choose_same and len(same_subnet_ids) > 0:
            return self._choice(rng, same_subnet_ids)

        if (not choose_same) and len(cross_subnet_ids) > 0:
            return self._choice(rng, cross_subnet_ids)

        if len(same_subnet_ids) > 0:
            return self._choice(rng, same_subnet_ids)

        return self._choice(rng, cross_subnet_ids)

    def _do_scanning(self, host, network, traffic_manager, current_timestep: int, rng) -> List[int]:
        """
        Handle one host's scan traffic and possible infections.

        Rules:
        - scanning starts at age >= SCAN_DELAY
        - probes per step = random(3, 5)
        - if containment blocks delivery, record outgoing-only failed probe
        - if delivery succeeds:
            * target already infected -> small success probe, no state change
            * target clean, exploit fails -> small failed exploit, reached target
            * target clean, exploit succeeds -> large payload + infection
        """
        newly_infected: List[int] = []

        if host.infection_age(current_timestep) < SCAN_DELAY:
            return newly_infected

        num_probes = self._rand_int_inclusive(
            rng,
            SCAN_PROBES_PER_STEP[0],
            SCAN_PROBES_PER_STEP[1],
        )

        for _ in range(num_probes):
            target_id = self._choose_scan_target_id(host, network, rng)
            if target_id is None:
                continue

            if target_id == SERVER_HOST_ID:
                raise RuntimeError("Server host must never be selected as a scan target.")

            target_host = network.get_host(target_id)

            scan_port = self._rand_int_inclusive(
                rng,
                SCAN_PROBE_PORT_RANGE[0],
                SCAN_PROBE_PORT_RANGE[1],
            )

            decision = network.decide_connection(host.host_id, target_id)

            # Firewall / containment blocked the probe before arrival
            if not decision.reached_destination:
                bytes_sent = self._rand_int_inclusive(
                    rng,
                    SCAN_PROBE_BYTES[0],
                    SCAN_PROBE_BYTES[1],
                )

                traffic_manager.record_connection(
                    network=network,
                    source_id=host.host_id,
                    dest_id=target_id,
                    dest_port=scan_port,
                    bytes_sent=bytes_sent,
                    bytes_received=0,
                    success=False,
                    timestamp=current_timestep,
                    is_malicious=True,
                    reached_destination=False,
                )
                continue

            # Probe reached the target host
            if target_host.is_infected:
                bytes_sent = self._rand_int_inclusive(
                    rng,
                    SCAN_PROBE_BYTES[0],
                    SCAN_PROBE_BYTES[1],
                )
                bytes_received = self._rand_int_inclusive(
                    rng,
                    SCAN_PROBE_RECV_ALREADY_INFECTED[0],
                    SCAN_PROBE_RECV_ALREADY_INFECTED[1],
                )

                traffic_manager.record_connection(
                    network=network,
                    source_id=host.host_id,
                    dest_id=target_id,
                    dest_port=scan_port,
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    success=True,
                    timestamp=current_timestep,
                    is_malicious=True,
                    reached_destination=True,
                )
                continue

            # Target is clean: exploit roll happens only after delivery
            exploit_success = self._rand_float_01(rng) < target_host.vulnerability

            if exploit_success:
                bytes_sent = self._rand_int_inclusive(
                    rng,
                    INFECTION_PAYLOAD_BYTES_SENT[0],
                    INFECTION_PAYLOAD_BYTES_SENT[1],
                )
                bytes_received = self._rand_int_inclusive(
                    rng,
                    INFECTION_PAYLOAD_BYTES_RECEIVED[0],
                    INFECTION_PAYLOAD_BYTES_RECEIVED[1],
                )

                changed = target_host.infect(current_timestep)
                if changed:
                    newly_infected.append(target_id)

                traffic_manager.record_connection(
                    network=network,
                    source_id=host.host_id,
                    dest_id=target_id,
                    dest_port=scan_port,
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    success=True,
                    timestamp=current_timestep,
                    is_malicious=True,
                    reached_destination=True,
                )
            else:
                bytes_sent = self._rand_int_inclusive(
                    rng,
                    SCAN_PROBE_BYTES[0],
                    SCAN_PROBE_BYTES[1],
                )

                # Non-firewall failure: exploit failed after delivery.
                # Must be recorded as reached_destination=True.
                traffic_manager.record_connection(
                    network=network,
                    source_id=host.host_id,
                    dest_id=target_id,
                    dest_port=scan_port,
                    bytes_sent=bytes_sent,
                    bytes_received=0,
                    success=False,
                    timestamp=current_timestep,
                    is_malicious=True,
                    reached_destination=True,
                )

        return newly_infected

    # -------------------------------------------------------------------------
    # Phase 3: Server attack
    # -------------------------------------------------------------------------

    def _do_server_attack(self, host, network, traffic_manager, current_timestep: int, rng) -> None:
        """
        Handle attack traffic from one infected host to the file server.

        By spec:
        - once the attack phase is active, each infected host sends 1-3
          attack connections per step
        - blocked/quarantined hosts still attempt them
        - only successful connections add damage
        - bytes_received is always 0
        """
        num_connections = self._rand_int_inclusive(
            rng,
            ATTACK_CONNECTIONS_PER_STEP[0],
            ATTACK_CONNECTIONS_PER_STEP[1],
        )

        for _ in range(num_connections):
            decision = network.decide_connection(host.host_id, SERVER_HOST_ID)

            bytes_sent = self._rand_int_inclusive(
                rng,
                ATTACK_BYTES_PER_CONNECTION[0],
                ATTACK_BYTES_PER_CONNECTION[1],
            )

            traffic_manager.record_connection(
                network=network,
                source_id=host.host_id,
                dest_id=SERVER_HOST_ID,
                dest_port=ATTACK_DEST_PORT,
                bytes_sent=bytes_sent,
                bytes_received=0,
                success=decision.success,
                timestamp=current_timestep,
                is_malicious=True,
                reached_destination=decision.reached_destination,
            )

            if decision.success:
                self.server_damage += SERVER_DAMAGE_PER_CONNECTION