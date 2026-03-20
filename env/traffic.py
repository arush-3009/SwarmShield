"""
SwarmShield Traffic Generation & Feature Computation
=====================================================

This file does two jobs:

JOB 1 — GENERATE TRAFFIC
Every timestep, every operational host generates traffic. Clean hosts generate
normal, non-malicious traffic (browsing, file access, etc.). Infected hosts generate both
normal traffic && malicious traffic (beacons, scans, attacks). This does not
simulate actual packets, instead it simulates traffic records, each describing one
connection (source, destination, port, bytes, success/failure, timestamp).

JOB 2 — COMPUTE FEATURES
When an RL agent stands at a host and observes, it needs a numerical vector
describing the traffic at that host. We compute 13 features from the rolling
window of recent traffic records like entropy of destinations, flow counts, failed
connection rates, etc. They capture the statistical signatures that distinguish
normal traffic from beaconing, scanning, and DDoS attacks.

"""

import numpy as np
from collections import defaultdict
from env.config import (
    NUM_HOSTS,
    OBSERVATION_WINDOW,
    NORMAL_FLOWS_PER_STEP,
    NORMAL_BYTES_RANGE,
    NORMAL_SAME_SUBNET_PROB,
    NORMAL_SERVER_PROB,
    NORMAL_CROSS_SUBNET_PROB,
    NORMAL_PORTS,
    SERVER_HOST_ID,
    SUBNET_HOSTS,
    HOST_TO_SUBNET,
    SUSPICIOUS_DECAY_FACTOR,
    EPSILON,
    NORM_TOTAL_FLOWS,
    NORM_CONN_ATTEMPTS,
    NORM_UNIQUE_DEST_IPS,
    NORM_UNIQUE_DEST_PORTS,
    NORM_ENTROPY,
    NORM_BYTES,
    NORM_FANO,
    NORM_SUSPICIOUS,
    NUM_TRAFFIC_FEATURES,
)


class TrafficRecord:
    """
    One network connection record -> basically a monitoring log for each connection.
    
    In real networking terms, this corresponds to one TCP 
    flow, i.e., SYN → SYN-ACK → ACK → data → FIN sequence between two hosts.
    
    source_id:      int, host ID that initiated the connection (0-17)
    dest_id:        int, host ID that received the connection (0-17, or -1 for external C2)
    dest_port:      int, port number on the destination (80, 443, 445, etc.)
    bytes_sent:     int, bytes the source sent TO the destination
    bytes_received: int, bytes the source received FROM the destination
    success:        bool, indicates whether the connection got completed (True = SYN-ACK received,
                    False = RST or timeout —> connection failed)
    timestamp:      int, the timestep when this connection happened
    is_malicious:   bool, True if this is beacon/scan/attack traffic.
                    The agent never sees this flag. It's for internal
                    bookkeeping and reward computation only.
    """

    def __init__(self, source_id, dest_id, dest_port, bytes_sent,
                 bytes_received, success, timestamp, is_malicious=False):
        self.source_id = source_id
        self.dest_id = dest_id
        self.dest_port = dest_port
        self.bytes_sent = bytes_sent
        self.bytes_received = bytes_received
        self.success = success
        self.timestamp = timestamp
        self.is_malicious = is_malicious


class TrafficManager:
    """
    Manages all traffic generation and feature computation.

    This class:
    1. Maintains a rolling window of traffic records for each host.
    2. Generates normal (legitimate) traffic for clean hosts each timestep.
    3. Provides methods for the attacker to add malicious traffic records.
    4. Computes the 13-feature observation vector for any host on demand.
    5. Tracks a decayed cumulative suspicious-activity score per host.
    """

    def __init__(self):
        """
        Initialize traffic storage for all hosts.

        traffic_history[host_id] is a list of TrafficRecord objects.
        We keep only the last OBSERVATION_WINDOW timesteps of records.

        suspicious_scores[host_id] is the decayed cumulative suspicious
        activity score. It accumulates over the entire episode (not just
        the window) and old events gradually fade via exponential decay.
        """
        # Per-host list of traffic records within the rolling window
        self.traffic_history = {}
        for host_id in range(NUM_HOSTS):
            self.traffic_history[host_id] = []

        # Per-host decayed suspicious activity score
        # This is the "long-term memory" feature — it remembers that a
        # host was suspicious even after the rolling window has moved past.
        self.suspicious_scores = {}
        for host_id in range(NUM_HOSTS):
            self.suspicious_scores[host_id] = 0.0

    def reset(self):
        """Clear all traffic history for a new episode."""
        for host_id in range(NUM_HOSTS):
            self.traffic_history[host_id] = []
            self.suspicious_scores[host_id] = 0.0

    def add_record(self, record):
        """
        Add a traffic record to a host's history.

        The record is added to the SOURCE host's history, because the
        agent observes traffic at the host it's standing at, and we
        care about what traffic this host is GENERATING (sending out).

        Also updates the suspicious activity score if the traffic is malicious.
        """
        host_id = record.source_id
        self.traffic_history[host_id].append(record)

        # If this is malicious traffic, bump the suspicious score.
        # The score uses exponential decay — old events fade over time.
        # We add 1.0 for each malicious connection.
        if record.is_malicious:
            self.suspicious_scores[host_id] += 1.0

    def prune_old_records(self, current_timestep):
        """
        Remove traffic records that are older than the rolling window.

        We keep records from the last OBSERVATION_WINDOW timesteps.
        Everything older gets deleted. This keeps memory usage constant
        and ensures features reflect recent activity, not ancient history.
        """
        cutoff = current_timestep - OBSERVATION_WINDOW

        for host_id in range(NUM_HOSTS):
            # Keep only records with timestamp > cutoff
            fresh_records = []
            for record in self.traffic_history[host_id]:
                if record.timestamp > cutoff:
                    fresh_records.append(record)
            self.traffic_history[host_id] = fresh_records

    def decay_suspicious_scores(self):
        """
        Apply exponential decay to all suspicious activity scores.

        Called once per timestep. Each step, every host's suspicious score
        is multiplied by the decay factor (0.995). This means:
        - After 10 steps: score retains 95% (barely faded)
        - After 100 steps: score retains 61% (noticeably faded)
        - After 500 steps: score retains 8% (mostly gone)

        This gives agents a "long-term memory" of suspicious behavior that
        naturally fades, so hosts that were suspicious long ago don't
        permanently look bad.
        """
        for host_id in range(NUM_HOSTS):
            self.suspicious_scores[host_id] *= SUSPICIOUS_DECAY_FACTOR

    def generate_normal_traffic(self, network, current_timestep, rng):
        """
        Generate normal (legitimate) traffic for all clean, operational hosts.

        Each clean, non-quarantined host makes 1-4 random connections per timestep.
        Destinations are chosen to mimic real office behavior:
        - 50% to coworkers in the same department (same subnet)
        - 30% to the file server (accessing shared files)
        - 20% to other departments (cross-subnet)

        All normal connections succeed (no failed connections).
        Bytes are random within a realistic range.
        Ports are chosen from common office services (HTTP, HTTPS, SMB, etc.).

        This creates the "background noise" that malicious traffic hides in.
        Without this, any traffic at all would be suspicious and detection
        would be trivially easy.

        Args:
            network: Network object with current host states
            current_timestep: int, current simulation timestep
            rng: numpy random generator
        """
        for host_id in range(NUM_HOSTS):
            host = network.get_host(host_id)

            # Skip hosts that can't generate traffic
            # Quarantined hosts are fully offline — no traffic at all.
            # Blocked hosts CAN still generate within-subnet traffic.
            if host.is_quarantined:
                continue

            # Decide how many connections this host makes this timestep.
            # Random between 1 and 4, mimicking variable human activity.
            num_flows = rng.integers(
                NORMAL_FLOWS_PER_STEP[0],
                NORMAL_FLOWS_PER_STEP[1] + 1
            )

            for _ in range(num_flows):
                # Pick a destination based on probability distribution.
                # This models real office behavior: mostly local traffic,
                # some file server access, occasional cross-department.
                roll = rng.random()

                if roll < NORMAL_SAME_SUBNET_PROB:
                    # Same subnet — talking to a coworker
                    same_subnet_hosts = network.get_hosts_in_same_subnet(host_id)
                    # Filter to only operational hosts
                    reachable = []
                    for h in same_subnet_hosts:
                        if h.is_operational:
                            reachable.append(h)
                    if len(reachable) == 0:
                        continue  # Nobody to talk to in this subnet
                    target = rng.choice(reachable)
                    dest_id = target.host_id

                elif roll < NORMAL_SAME_SUBNET_PROB + NORMAL_SERVER_PROB:
                    # File server — accessing shared files
                    server = network.get_host(SERVER_HOST_ID)
                    if not server.is_operational:
                        continue  # Server is down
                    # If host is blocked, it can't reach the server (cross-subnet)
                    if host.is_blocked:
                        continue
                    dest_id = SERVER_HOST_ID

                else:
                    # Cross-subnet — talking to another department
                    if host.is_blocked:
                        continue  # Blocked hosts can't go cross-subnet
                    cross_hosts = network.get_hosts_in_different_subnets(host_id)
                    reachable = []
                    for h in cross_hosts:
                        if h.is_operational:
                            reachable.append(h)
                    if len(reachable) == 0:
                        continue
                    target = rng.choice(reachable)
                    dest_id = target.host_id

                # Generate random bytes and pick a random normal port
                bytes_sent = rng.integers(
                    NORMAL_BYTES_RANGE[0],
                    NORMAL_BYTES_RANGE[1] + 1
                )
                # Normal traffic has a response — you send a request, you get data back.
                # Received bytes are typically larger than sent (you request a file,
                # you get the file back — more data coming in than going out).
                bytes_received = rng.integers(
                    bytes_sent,
                    bytes_sent * 3 + 1
                )
                dest_port = rng.choice(NORMAL_PORTS)

                # Create the traffic record
                record = TrafficRecord(
                    source_id=host_id,
                    dest_id=dest_id,
                    dest_port=dest_port,
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    success=True,           # Normal connections always succeed
                    timestamp=current_timestep,
                    is_malicious=False,
                )
                self.add_record(record)

    def compute_features(self, host_id):
        """
        Compute the 13-feature observation vector for a given host.

        This is what the RL agent "sees" when it stands at this host.
        All features are computed from the traffic records in the rolling
        window and normalized to approximately [0, 1] range.

        The 13 features are:
         0: Total flows (connections) in the window
         1: Total connection attempts (including failed ones)
         2: Failed connection rate (fraction of attempts that failed)
         3: Unique destination IPs contacted
         4: Unique destination ports targeted
         5: Entropy of destination IPs (diversity of who this host talks to)
         6: Entropy of inter-arrival times (regularity of connection timing)
         7: SYN-to-ACK ratio (proxy: attempt-to-success ratio)
         8: Bytes sent in window
         9: Bytes received in window
        10: Sent/received byte ratio
        11: Fano factor of packets per timestep (burstiness measure)
        12: Decayed cumulative suspicious activity score

        Args:
            host_id: int, which host to compute features for (0-17)

        Returns:
            numpy array of 13 float values, each normalized to ~[0, 1]
        """
        records = self.traffic_history[host_id]
        features = np.zeros(NUM_TRAFFIC_FEATURES, dtype=np.float32)

        # If no traffic records exist yet (early in episode), return zeros.
        # The agent sees "nothing happening here" which is accurate.
        if len(records) == 0:
            # Still include the decayed suspicious score even with no current traffic
            features[12] = self.suspicious_scores[host_id] / NORM_SUSPICIOUS
            return features

        # =====================================================================
        # Feature 0: Total flows in the rolling window
        # =====================================================================
        # Count all connections (both successful and failed).
        # Normal host: 10-30 over 10 timesteps.
        # Scanning host: 50+ (lots of probes).
        total_flows = len(records)
        features[0] = total_flows / NORM_TOTAL_FLOWS

        # =====================================================================
        # Feature 1: Total connection attempts
        # =====================================================================
        # Same as total flows in our simulation (each record IS an attempt).
        # In a real system, you might count SYN packets separately.
        total_attempts = len(records)
        features[1] = total_attempts / NORM_CONN_ATTEMPTS

        # =====================================================================
        # Feature 2: Failed connection rate
        # =====================================================================
        # What fraction of connections failed?
        # Normal: near 0 (you connect to known, working destinations).
        # Scanning: near 1 (blindly probing, most targets don't respond).
        failed_count = 0
        for record in records:
            if not record.success:
                failed_count += 1
        if total_attempts > 0:
            features[2] = failed_count / total_attempts
        else:
            features[2] = 0.0

        # =====================================================================
        # Feature 3: Unique destination IPs
        # =====================================================================
        # How many different hosts did this machine contact?
        # Normal: 3-6 (coworkers + server).
        # Scanning: 15+ (probing many targets).
        unique_dests = set()
        for record in records:
            unique_dests.add(record.dest_id)
        features[3] = len(unique_dests) / NORM_UNIQUE_DEST_IPS

        # =====================================================================
        # Feature 4: Unique destination ports
        # =====================================================================
        # How many different services did this machine target?
        # Normal: 1-3 (HTTP, HTTPS, file sharing).
        # Port scanning: many different ports.
        unique_ports = set()
        for record in records:
            unique_ports.add(record.dest_port)
        features[4] = len(unique_ports) / NORM_UNIQUE_DEST_PORTS

        # =====================================================================
        # Feature 5: Entropy of destination IPs
        # =====================================================================
        # Entropy measures diversity/randomness of a distribution.
        # Formula: E = -sum(p_i * log(p_i)) for each unique value i
        # where p_i = fraction of connections going to destination i.
        #
        # High entropy = diverse destinations (scanning — hitting many targets).
        # Low entropy = concentrated destinations (beaconing — always same C2 server).
        # Moderate entropy = normal behavior (a few regular destinations).
        #
        # Why this catches beaconing: if a host makes 20 connections and 8 of
        # them go to the same C2 IP, that IP's probability is 0.4 — much higher
        # than any single destination in normal traffic. This concentration
        # pulls the entropy down.
        features[5] = self._compute_entropy_of_destinations(records) / NORM_ENTROPY

        # =====================================================================
        # Feature 6: Entropy of inter-arrival times
        # =====================================================================
        # Inter-arrival time = gap between consecutive connections.
        # Beaconing: gaps are semi-regular (e.g. all around 2-3 timesteps).
        #   Even with jitter, this is MORE regular than human behavior.
        #   Low entropy.
        # Normal: gaps are wildly variable (0.1s, 3s, 15s, 0.5s).
        #   High entropy.
        #
        # We discretize the inter-arrival times into bins to compute entropy.
        features[6] = self._compute_entropy_of_inter_arrivals(records) / NORM_ENTROPY

        # =====================================================================
        # Feature 7: SYN-to-ACK ratio (attempt-to-success ratio)
        # =====================================================================
        # In real networking: SYN packets = connection attempts,
        # ACK packets = successful connections.
        # In our simulation: we use the ratio of total attempts to successful ones.
        #
        # Normal: ratio near 1 (almost all connections succeed).
        # Scanning: ratio >> 1 (many attempts, few successes).
        # SYN flood: ratio very high (all attempts, no completions).
        success_count = 0
        for record in records:
            if record.success:
                success_count += 1
        if success_count > 0:
            features[7] = total_attempts / (success_count + EPSILON)
        else:
            # All failed — maximum suspicion
            features[7] = total_attempts
        # Normalize: normal ratio is ~1, scanning might be 5-10
        features[7] = min(features[7] / 10.0, 1.0)

        # =====================================================================
        # Feature 8: Bytes sent in rolling window
        # =====================================================================
        # Total outbound data from this host.
        # Normal: 1000-10000 bytes over 10 timesteps.
        # DDoS/exfiltration: 100000+ bytes (massive spike).
        total_bytes_sent = 0
        for record in records:
            total_bytes_sent += record.bytes_sent
        features[8] = total_bytes_sent / NORM_BYTES

        # =====================================================================
        # Feature 9: Bytes received in rolling window
        # =====================================================================
        # Total inbound data to this host.
        # A spike could mean receiving a malware payload during infection.
        total_bytes_received = 0
        for record in records:
            total_bytes_received += record.bytes_received
        features[9] = total_bytes_received / NORM_BYTES

        # =====================================================================
        # Feature 10: Sent/received byte ratio
        # =====================================================================
        # Normal browsing: you send small requests, receive large responses.
        #   Ratio < 1 (receive more than you send).
        # Exfiltration: you're uploading stolen data TO the C2 server.
        #   Ratio > 1 (send more than you receive).
        # A flip in this ratio is a strong signal that something changed.
        if total_bytes_received > 0:
            ratio = total_bytes_sent / (total_bytes_received + EPSILON)
        else:
            ratio = total_bytes_sent / (1.0 + EPSILON)
        # Normalize: normal is ~0.3-0.5, exfiltration could be 2-5
        features[10] = min(ratio / 5.0, 1.0)

        # =====================================================================
        # Feature 11: Fano factor of connections per timestep
        # =====================================================================
        # Fano factor = variance / mean of connection counts per timestep.
        #
        # It measures "burstiness" — how steady or spiky the traffic is.
        # Near 0: traffic arrives at a perfectly steady rate (beaconing —
        #         same number of connections every timestep, like a metronome).
        # Around 1: traffic arrives randomly (normal human behavior).
        # Much > 1: traffic arrives in bursts (DDoS — quiet then sudden flood).
        #
        # We count how many connections happened in each timestep within
        # the window, then compute variance / mean of those counts.
        features[11] = self._compute_fano_factor(records) / NORM_FANO

        # =====================================================================
        # Feature 12: Decayed cumulative suspicious activity score
        # =====================================================================
        # This is the "long-term memory" that the rolling window can't provide.
        # It accumulates over the entire episode. Each malicious connection
        # adds 1.0 to the score. Each timestep, the score is multiplied by
        # 0.995 (exponential decay — old events gradually fade).
        #
        # A host that had suspicious activity 50 timesteps ago will have a
        # faded but non-zero score. A host that's actively suspicious right
        # now will have a high score. A host that's never been suspicious
        # will have a score of 0.
        features[12] = self.suspicious_scores[host_id] / NORM_SUSPICIOUS

        # Clip all features to [0, 1] range to be safe.
        # Some features could theoretically exceed their normalization constant
        # in extreme scenarios. Clipping prevents the neural network from
        # seeing unexpectedly large values.
        features = np.clip(features, 0.0, 1.0)

        return features

    def _compute_entropy_of_destinations(self, records):
        """
        Compute Shannon entropy of destination IP distribution.

        Entropy formula: E = -sum(p_i * log(p_i))
        where p_i is the fraction of connections going to destination i.

        Example:
        - 10 connections, all to the same IP: p = [1.0] → E = 0 (no diversity)
        - 10 connections, each to a different IP: p = [0.1]*10 → E = 2.3 (max diversity)
        - 10 connections, 6 to IP_A, 2 to IP_B, 2 to IP_C: p = [0.6, 0.2, 0.2] → E ≈ 1.0

        Beaconing produces low entropy (one C2 destination dominates).
        Scanning produces high entropy (many different targets).
        """
        if len(records) == 0:
            return 0.0

        # Count how many times each destination appears
        dest_counts = defaultdict(int)
        for record in records:
            dest_counts[record.dest_id] += 1

        # Convert counts to probabilities
        total = len(records)
        entropy = 0.0
        for dest_id, count in dest_counts.items():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p + EPSILON)

        return entropy

    def _compute_entropy_of_inter_arrivals(self, records):
        """
        Compute Shannon entropy of inter-arrival time distribution.

        Inter-arrival time = the gap (in timesteps) between consecutive
        connections from this host. We collect all these gaps, bin them
        into discrete categories, and compute the entropy of the bin
        distribution.

        Beaconing: gaps are semi-regular (all around 2-3 timesteps).
          Most gaps fall in the same bin → low entropy.
        Normal: gaps are wildly variable (0, 1, 3, 0, 5, 1, ...).
          Gaps spread across many bins → high entropy.

        We use the timestep field of records as the timing signal.
        """
        if len(records) < 2:
            return 0.0

        # Sort records by timestamp to get chronological order
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Compute inter-arrival times (gap between consecutive records)
        gaps = []
        for i in range(1, len(sorted_records)):
            gap = sorted_records[i].timestamp - sorted_records[i - 1].timestamp
            gaps.append(gap)

        if len(gaps) == 0:
            return 0.0

        # Bin the gaps: 0, 1, 2, 3, 4, 5+
        # This discretization is necessary because entropy requires
        # discrete categories. Gaps of 0 and 1 are "fast bursts",
        # gaps of 2-3 are "regular pacing", gaps of 5+ are "long pauses".
        bin_counts = defaultdict(int)
        for gap in gaps:
            if gap >= 5:
                bin_counts[5] += 1  # All long gaps in one bin
            else:
                bin_counts[gap] += 1

        # Compute entropy of the bin distribution
        total = len(gaps)
        entropy = 0.0
        for bin_id, count in bin_counts.items():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p + EPSILON)

        return entropy

    def _compute_fano_factor(self, records):
        """
        Compute the Fano factor (variance / mean) of connections per timestep.

        We count how many connections happened in each timestep within the
        rolling window, producing an array like [3, 2, 4, 1, 3, 2, 2, 3, 4, 2].
        Then: Fano = variance(counts) / mean(counts).

        Near 0: perfectly steady (same count every timestep — bot-like).
        Around 1: random Poisson-like arrivals (normal traffic).
        Much > 1: bursty (quiet periods interrupted by floods — DDoS).

        The Fano factor is also known as the Index of Dispersion.
        """
        if len(records) == 0:
            return 0.0

        # Find the range of timesteps in the records
        timesteps = set()
        for record in records:
            timesteps.add(record.timestamp)

        if len(timesteps) == 0:
            return 0.0

        min_ts = min(timesteps)
        max_ts = max(timesteps)

        # Count connections per timestep
        counts_per_step = []
        for ts in range(min_ts, max_ts + 1):
            count = 0
            for record in records:
                if record.timestamp == ts:
                    count += 1
            counts_per_step.append(count)

        if len(counts_per_step) < 2:
            return 0.0

        mean_count = np.mean(counts_per_step)
        if mean_count < EPSILON:
            return 0.0

        variance_count = np.var(counts_per_step)
        fano = variance_count / (mean_count + EPSILON)

        return fano