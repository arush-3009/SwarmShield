"""
SwarmShield Attacker: Automated Botnet Logic
===============================================

This is the botnet that the RL agents defend against.
The attacker is not an RL agent, it follows scripted rules representing
the real-world C2 (Command and Control) botnet lifecycle:

Phase 1 — BEACONING: Every infected host periodically contacts the C2 server.
    Small TCP connection, same destination every time, semi-regular timing
    with random jitter.

Phase 2 — SCANNING: After being infected for a few timesteps, the zombie
    starts probing other hosts. Many connection attempts to random targets,
    most fail. This is how the botnet discovers new machines to infect.

Phase 3 — INFECTION: When a scan reaches a vulnerable host, infection is
    attempted. Success probability = target's vulnerability score.
    Newly infected hosts join the botnet and start their own beaconing/scanning.

Phase 4 — ATTACK: Once enough hosts are infected, ALL zombies start flooding
    the file server with massive traffic. This is the DDoS / data exfiltration
    phase. If the traffic reaches the server and isn't blocked, the server
    accumulates damage and eventually gets compromised.

The attacker generates TrafficRecords for all malicious activity, which get
mixed into the same traffic history that normal traffic uses. The RL agents
see both normal and malicious records mixed together and must learn to
distinguish them based on statistical patterns (entropy, flow counts, etc.)
"""

import numpy as np
from env.config import (
    NUM_HOSTS,
    SERVER_HOST_ID,
    HOST_TO_SUBNET,
    SUBNET_HOSTS,
    BEACON_INTERVAL_BASE,
    BEACON_JITTER,
    BEACON_BYTES,
    BEACON_PORT,
    SCAN_DELAY,
    SCAN_PROBES_PER_STEP,
    SCAN_SAME_SUBNET_PROB,
    ATTACK_INFECTION_THRESHOLD,
    ATTACK_CONNECTIONS_PER_STEP,
    ATTACK_BYTES_PER_CONNECTION,
    SERVER_DAMAGE_PER_CONNECTION,
)
from env.traffic import TrafficRecord, TrafficManager
from env.network import Network


# The C2 server is outside the network -> use -1 as its host_ID to distinguish it from internal hosts (0-17).
C2_SERVER_ID = -1


class Attacker:
    """
    The automated botnet controller.

    This class manages all infected hosts and runs the botnet logic
    each timestep. It does not learn, it follows fixed heuristics.

    The attacker tracks:
    - Which hosts are infected (via the Network object)
    - Each infected host's beacon timer (when to next phone home)
    - Accumulated damage on the file server
    """

    def __init__(self):
        """Initialize attacker state"""
        
        # Server damage accumulates when attack traffic reaches the server.
        # If it exceeds SERVER_DAMAGE_THRESHOLD, the server is compromised.
        self.server_damage = 0.0

    def reset(self):
        """Reset attacker state for a new episode."""
        self.server_damage = 0.0

    def step(self, network: Network, traffic_manager: TrafficManager, current_timestep, rng):
        """
        Run one timestep of attacker logic.

        For every infected, non-contained host:
        1. Handle beaconing
        2. Handle scanning
        3. Handle server attack

        Also attempts to infect hosts that were successfully scanned.

        network:          Network object with all host states
        traffic_manager:  TrafficManager to add malicious traffic records
        current_timestep: int, current simulation step
        rng:              numpy random generator

        Returns:
            list of newly infected host IDs (can be returned empty)
        """
        newly_infected = []

        # Get all infected hosts that are still active (not blocked/quarantined).
        # Blocked hosts can't send cross-subnet but CAN still do some things.
        # Quarantined hosts can't do anything.
        active_infected = network.get_infected_hosts()

        # Count total active infected for phase transition decisions
        total_active_infected = len(active_infected)

        for host in active_infected:

            # =================================================================
            # PHASE 1: BEACONING
            # =================================================================
            # Every infected host periodically contacts the C2 server.
            # The beacon is a small TCP connection to the same external IP
            # on port 443 (disguised as normal HTTPS traffic).
            #
            # The beacon timer counts down each timestep. When it reaches 0,
            # the host sends a beacon and the timer resets with jitter.
            #
            # Jitter means the interval isn't perfectly regular:
            #   base interval = 3 timesteps
            #   jitter = ±1 timestep
            #   actual interval = random between 2 and 4 timesteps
            # This makes detection harder — perfectly periodic traffic
            # is trivially detectable, but jittered periodicity is not.

            host.beacon_timer -= 1

            if host.beacon_timer <= 0:
                # Time to beacon — but only if the host can send cross-subnet.
                # Blocked hosts have their cross-subnet traffic dropped at
                # the router, so the beacon never reaches the C2 server.
                # The host still TRIES (it doesn't know it's blocked),
                # which generates a failed traffic record.

                beacon_succeeds = host.can_send_cross_subnet

                beacon_bytes = rng.integers(BEACON_BYTES[0], BEACON_BYTES[1] + 1)

                record = TrafficRecord(
                    source_id=host.host_id,
                    dest_id=C2_SERVER_ID,
                    dest_port=BEACON_PORT,
                    bytes_sent=beacon_bytes if beacon_succeeds else rng.integers(40, 80),
                    bytes_received=rng.integers(50, 150) if beacon_succeeds else 0,
                    success=beacon_succeeds,
                    timestamp=current_timestep,
                    is_malicious=True,
                )
                traffic_manager.add_record(record)

                # Reset beacon timer with jitter
                # rng.integers(low, high+1) gives uniform random in [low, high]
                jitter = rng.integers(-BEACON_JITTER, BEACON_JITTER + 1)
                host.beacon_timer = max(1, BEACON_INTERVAL_BASE + jitter)

            # =================================================================
            # PHASE 2: SCANNING
            # =================================================================
            # After being infected for SCAN_DELAY timesteps, the zombie
            # starts probing other hosts to find new infection targets.
            #
            # Each timestep, it sends 2-5 connection attempts to random hosts.
            # 70% of probes target the same subnet (faster, harder to detect
            # because traffic stays local and doesn't cross the router).
            # 30% target other subnets (crosses the router, easier for
            # agents to spot).
            #
            # Most scan probes FAIL — the target either doesn't have the
            # probed port open, or isn't vulnerable. This high failure rate
            # is a key detection signal.

            timesteps_since_infection = current_timestep - host.timestep_infected

            if timesteps_since_infection >= SCAN_DELAY:
                host.has_been_scanning = True

                num_probes = rng.integers(
                    SCAN_PROBES_PER_STEP[0],
                    SCAN_PROBES_PER_STEP[1] + 1
                )

                for i in range(num_probes):
                    
                    #  decide to scan same subnet or cross-subnet
                    scan_local = rng.random() < SCAN_SAME_SUBNET_PROB

                    if scan_local:
                        # Scan within same subnet
                        targets = network.get_hosts_in_same_subnet(host.host_id)
                    else:
                        # Scan cross-subnet — host tries regardless of block status
                        targets = network.get_hosts_in_different_subnets(host.host_id)

                    if len(targets) == 0:
                        continue

                    # Pick a random target to probe
                    target = rng.choice(targets)

                    # Skip if target already infected
                    if target.timestep_infected >= 0:
                        # Still generates a connection record (the scanner doesn't
                        # know the target is already infected until it connects)
                        record = TrafficRecord(
                            source_id=host.host_id,
                            dest_id=target.host_id,
                            dest_port=rng.integers(1, 1024),
                            bytes_sent=rng.integers(40, 80),
                            bytes_received=rng.integers(40, 80),
                            success=True,
                            timestamp=current_timestep,
                            is_malicious=True,
                        )
                        traffic_manager.add_record(record)
                        continue

                    # Determine if the probe reaches the target
                    
                    same_subnet = (HOST_TO_SUBNET[host.host_id] == HOST_TO_SUBNET[target.host_id])
                    
                    if target.is_quarantined:
                        # Target is offline, probe fails
                        probe_reaches = False
                    elif not same_subnet and not host.can_send_cross_subnet:
                        # Source is blocked, cross-subnet probe gets dropped at router
                        probe_reaches = False
                    else:
                        probe_reaches = True

                    # Probe reaches the target — does the exploit succeed
                    # Success probability = target's vulnerability score.
                    # A well-patched machine (vuln=0.2) resists 80% of attacks.
                    # An unpatched machine (vuln=0.7) falls 70% of the time.
                    exploit_roll = rng.random()
                    exploit_succeeds = exploit_roll < target.vulnerability

                    if exploit_succeeds:
                        # Infection successful!
                        target.infect(current_timestep)
                        newly_infected.append(target.host_id)

                        # Generate a successful connection record with data transfer
                        # (the malware payload being installed)
                        record = TrafficRecord(
                            source_id=host.host_id,
                            dest_id=target.host_id,
                            dest_port=rng.integers(1, 1024),
                            bytes_sent=rng.integers(5000, 20000),  # Malware payload
                            bytes_received=rng.integers(100, 500),
                            success=True,
                            timestamp=current_timestep,
                            is_malicious=True,
                        )
                        traffic_manager.add_record(record)

                    else:
                        # Exploit failed — target resisted.
                        
                        # probe still shows up as a connection attempt.
                        # Most scan probes end up here (target not vulnerable
                        # enough, or port not open). This is why scanning
                        # produces a high failed connection rate.
                        record = TrafficRecord(
                            source_id=host.host_id,
                            dest_id=target.host_id,
                            dest_port=rng.integers(1, 1024),
                            bytes_sent=rng.integers(40, 80),
                            bytes_received=0,
                            success=False,
                            timestamp=current_timestep,
                            is_malicious=True,
                        )
                        traffic_manager.add_record(record)

            # =================================================================
            # PHASE 4: ATTACK (server flooding)
            # =================================================================
            # Once enough hosts are infected (>= threshold), ALL active
            # infected hosts start flooding the file server with massive traffic.
            #
            # Each infected host sends 10-30 connections per timestep to the
            # server, each carrying 5000-50000 bytes. This creates a massive
            # spike in:
            #   - Bytes sent (feature 8)
            #   - Connections to one destination (low entropy on dest IPs, feature 5)
            #   - Fano factor (sudden burst of activity, feature 11)
            #
            # If the traffic reaches the server (not blocked), it accumulates
            # damage. The server is compromised when damage exceeds the threshold.

            if total_active_infected >= ATTACK_INFECTION_THRESHOLD:

                # see if host can reach server
                # Server is cross-subnet from everyone, so the host needs
                # to be able to send cross-subnet traffic.
                if not host.can_send_cross_subnet:
                    continue

                # Is the server still reachable? (not quarantined)
                server = network.get_host(SERVER_HOST_ID)
                if server.is_quarantined:
                    continue

                num_attack_connections = rng.integers(
                    ATTACK_CONNECTIONS_PER_STEP[0],
                    ATTACK_CONNECTIONS_PER_STEP[1] + 1
                )

                for _ in range(num_attack_connections):
                    attack_bytes = rng.integers(
                        ATTACK_BYTES_PER_CONNECTION[0],
                        ATTACK_BYTES_PER_CONNECTION[1] + 1
                    )

                    # Does the attack traffic actually reach the server?
                    # If server is blocked, its responses get dropped but
                    # the incoming attack traffic still arrives (blocking
                    # affects outbound from the blocked host, not inbound).
                    # Here, blocking a host blocks its
                    # FORWARDED traffic at the router. Attack traffic going
                    # TO the server passes through the router too, but the
                    # block rule is on the SOURCE (infected host), not the
                    # destination (server). So if the infected host is not
                    # blocked, attack traffic reaches the server regardless
                    # of the server's block status.
                    #
                    # The server takes damage from received attack traffic.
                    # A blocked server can still receive traffic (blocking
                    # only stops its outbound cross-subnet responses).

                    attack_succeeds = True  # Host already checked can_send_cross_subnet

                    record = TrafficRecord(
                        source_id=host.host_id,
                        dest_id=SERVER_HOST_ID,
                        dest_port=443,
                        bytes_sent=attack_bytes,
                        bytes_received=rng.integers(50, 200) if attack_succeeds else 0,
                        success=attack_succeeds,
                        timestamp=current_timestep,
                        is_malicious=True,
                    )
                    traffic_manager.add_record(record)

                    # Accumulate damage on the server
                    if attack_succeeds:
                        self.server_damage += SERVER_DAMAGE_PER_CONNECTION

        return newly_infected

    def infect_initial_hosts(self, network, current_timestep, num_infections, rng):
        """
        Infect the initial hosts at the start of an episode.

        Picks random hosts to infect, excluding the server (it's the target,
        not a starting point for infection).

        network:          Network object
        current_timestep: int (usually 0)
        num_infections:   int, how many hosts to infect initially
        rng:              numpy random generator

        Returns:
            list of initially infected host IDs
        """
        # Build list of candidates (all hosts except the server)
        candidates = []
        for host_id in range(NUM_HOSTS):
            if host_id != SERVER_HOST_ID:
                candidates.append(host_id)

        # Pick random hosts to infect
        chosen = rng.choice(candidates, size=num_infections, replace=False)

        infected_ids = []
        for host_id in chosen:
            host = network.get_host(host_id)
            host.infect(current_timestep)
            infected_ids.append(host_id)

        return infected_ids