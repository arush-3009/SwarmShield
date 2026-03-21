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
from env.network import Network, Host


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
    2. Generates normal/unmalicious traffic for clean hosts each timestep.
    3. Provides methods for the attacker to add malicious traffic records.
    4. Computes the 13-feature observation vector for any host on demand.
    5. Tracks a decayed cumulative suspicious-activity score per host.
    """

    def __init__(self):
        """
        Initialize traffic storage for all hosts.

        traffic_history[host_id] -> list of TrafficRecord objects.
            - We keep only the last OBSERVATION_WINDOW timesteps of records.

        suspicious_scores[host_id] -> decayed cumulative suspicious activity score. 
            - It accumulates over the entire episode (not just
            the window) => old events gradually fade via exponential decay.
        """
        # Per-host list of traffic records within the rolling window
        self.traffic_history = {}
        for host_id in range(NUM_HOSTS):
            self.traffic_history[host_id] = []

        # Per-host decayed suspicious activity score
        self.suspicious_scores = {}
        for host_id in range(NUM_HOSTS):
            self.suspicious_scores[host_id] = 0.0

    def reset(self):
        """Clear all traffic history for a new episode"""
        for host_id in range(NUM_HOSTS):
            self.traffic_history[host_id] = []
            self.suspicious_scores[host_id] = 0.0

    def add_record(self, record: TrafficRecord):
        """
        Add a traffic record to a host's history.

        The record is added to the host's history, because the
        agent observes traffic at the host it's standing at, and we
        care about what traffic this host is generating i.e., sending out.

        Also updates the suspicious activity score if the traffic is malicious.
        """
        host_id = record.source_id
        self.traffic_history[host_id].append(record)

        # If this is malicious traffic, bump the suspicious score.
        if record.is_malicious:
            self.suspicious_scores[host_id] += 1.0

    def prune_old_records(self, current_timestep):
        """
        Remove traffic records that are older than the rolling window.
        """
        cutoff = current_timestep - OBSERVATION_WINDOW

        for host_id in range(NUM_HOSTS):
            
            fresh_records = []
            for record in self.traffic_history[host_id]:
                if record.timestamp > cutoff:
                    fresh_records.append(record)
            self.traffic_history[host_id] = fresh_records

    def decay_suspicious_scores(self):
        """
        Apply exponential decay to all suspicious activity scores.

        Called once per timestep. Each step, every host's suspicious score
        is multiplied by the decay factor.
        """
        for host_id in range(NUM_HOSTS):
            self.suspicious_scores[host_id] *= SUSPICIOUS_DECAY_FACTOR

    def generate_normal_traffic(self, network, current_timestep, rng):
        for host_id in range(NUM_HOSTS):
            host = network.get_host(host_id)

            if host.is_quarantined:
                continue

            num_flows = rng.integers(
                NORMAL_FLOWS_PER_STEP[0],
                NORMAL_FLOWS_PER_STEP[1] + 1
            )

            for _ in range(num_flows):
                roll = rng.random()
                dest_id = None
                connection_succeeds = True

                if roll < NORMAL_SAME_SUBNET_PROB:
                    # attempting connection within same subnet
                    same_subnet_hosts = network.get_hosts_in_same_subnet(host_id)
                    if len(same_subnet_hosts) == 0:
                        continue
                    target = rng.choice(same_subnet_hosts)
                    dest_id = target.host_id

                    # Host doesn't know destination's status, it just tries.
                    # Within same subnet:
                    #   - Quarantined dest: completely offline, no response, fails
                    #   - Blocked dest: can still communicate within subnet, succeeds
                    #   - Clean/infected dest: succeeds
                    dest_host = network.get_host(dest_id)
                    if dest_host.is_quarantined:
                        connection_succeeds = False

                elif roll < NORMAL_SAME_SUBNET_PROB + NORMAL_SERVER_PROB:
                    # attempting connection with file server -> via router as file server is in a separate subnet from every other host.
                    # If this host is blocked, it can't send cross-subnet at all
                    if host.is_blocked:
                        continue
                    
                    dest_id = SERVER_HOST_ID

                    # Server might be quarantined or blocked
                    server = network.get_host(SERVER_HOST_ID)
                    if server.is_quarantined or server.is_blocked:
                        connection_succeeds = False

                else: # Cross-subnet —> attempting connection with another host in another subnet over the router.
                    
                    if host.is_blocked:
                        continue
                    
                    cross_hosts = network.get_hosts_in_different_subnets(host_id)
                    if len(cross_hosts) == 0:
                        continue
                    
                    target = rng.choice(cross_hosts)
                    dest_id = target.host_id

                    # Host doesn't know destination's status, it just tries.
                    # Cross-subnet:
                    #   - Quarantined dest: offline, fails
                    #   - Blocked dest: response gets dropped at router, fails
                    dest_host = network.get_host(dest_id)
                    if dest_host.is_quarantined or dest_host.is_blocked:
                        connection_succeeds = False
                        

                if connection_succeeds:
                    bytes_sent = rng.integers(
                        NORMAL_BYTES_RANGE[0],
                        NORMAL_BYTES_RANGE[1] + 1
                    )
                    bytes_received = rng.integers(bytes_sent, bytes_sent * 3 + 1)
                else:
                    bytes_sent = rng.integers(40, 80)  # Just a SYN packet
                    bytes_received = 0

                dest_port = rng.choice(NORMAL_PORTS)

                record = TrafficRecord(
                    source_id=host_id,
                    dest_id=dest_id,
                    dest_port=dest_port,
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    success=connection_succeeds,
                    timestamp=current_timestep,
                    is_malicious=False,
                )
                self.add_record(record)

    def compute_features(self, host_id):
        """
        Compute the 13-feature observation vector for a given host.

        This is what the RL agent sees/gets when it stands at this host.
        All features are computed from the traffic records in the rolling
        window and normalized to approximately [0, 1] range.

        The 13 features are:
        0: Total flows (connections) in the window
        1: Total connection attempts (including failed ones)
        2: Failed connection rate (fraction of attempts that failed)
        3: Unique destination IPs contacted
        4: Unique destination ports targeted
        5: Entropy of destination IPs
        6: Entropy of time intervals between successive connection requests/attempts.
        7: SYN-to-ACK ratio (proxy for -> attempt-to-success ratio)
        8: Bytes sent in window
        9: Bytes received in window
        10:Sent/received byte ratio
        11:Fano factor of packets per timestep (burstiness measure)
        12:Decayed cumulative suspicious activity score

        
        host_id: int, which host to compute features for (0-17)

    
        returns -> numpy array of 13 float values, each normalized to ~[0, 1]
        """
        records = self.traffic_history[host_id]
        features = np.zeros(NUM_TRAFFIC_FEATURES, dtype=np.float32)

        # If no traffic records exist yet, i.e. early in episode -> return zeros.
        if len(records) == 0:
            
            # include decayed suspicious score even with no current traffic
            features[12] = self.suspicious_scores[host_id] / NORM_SUSPICIOUS
            return features

        # =====================================================================
        # Feature 0: Total flows -> (so succesfull connections) -> in the rolling window
        # =====================================================================
        # Count successful connections.
        successful_flows = 0
        for record in records:
            if record.success:
                successful_flows += 1
        features[0] = successful_flows / NORM_TOTAL_FLOWS

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
        # fraction of connections that failed
        
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
        # raw number of different hosts that the machine contacted in the window
        
        unique_dests = set()
        for record in records:
            unique_dests.add(record.dest_id)
        features[3] = len(unique_dests) / NORM_UNIQUE_DEST_IPS

        # =====================================================================
        # Feature 4: Unique destination ports
        # =====================================================================
        # raw number of different services that this machine targeted
        
        unique_ports = set()
        for record in records:
            unique_ports.add(record.dest_port)
        features[4] = len(unique_ports) / NORM_UNIQUE_DEST_PORTS

        # =====================================================================
        # Feature 5: Entropy of destination IPs
        # =====================================================================
        
        # High entropy = diverse destinations (suggests scanning —> hitting many targets).
        # Low entropy = concentrated destinations (suggests beaconing —> always same C2 server).
        # Moderate entropy = normal behavior (a few regular destinations).
        #
       
        features[5] = self._compute_entropy_of_destinations(records) / NORM_ENTROPY

        # =====================================================================
        # Feature 6: Entropy of time intervals between successive connection attempts
        # =====================================================================
        
        # Beaconing: gaps are semi-regular (e.g. all around 2-3 timesteps), even with jitter, this 
        # is MORE regular than human behavior -> Low entropy.
        
        # Normal: gaps are wildly variable (0.1s, 3s, 15s, 0.5s) -> High entropy.
        
        features[6] = self._compute_entropy_of_inter_arrivals(records) / NORM_ENTROPY

        # =====================================================================
        # Feature 7: SYN-to-ACK ratio (attempt-to-success ratio)
        # =====================================================================
        # In real networking: SYN packets = connection attempts,
        # ACK packets = successful connections.
        # In our simulation -> use the ratio of total attempts to successful ones.
        
        success_count = 0
        for record in records:
            if record.success:
                success_count += 1
        if success_count > 0:
            features[7] = total_attempts / (success_count + EPSILON)
        else:
            
            features[7] = total_attempts
        
        features[7] = min(features[7] / 10.0, 1.0)

        # =====================================================================
        # Feature 8: Bytes sent in rolling window
        # =====================================================================
        # Total outbound data from this host.
       
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
        # Normal browsing: you send small requests, receive large responses -> Ratio < 1 (receive more than you send).
        # Exfiltration: you're uploading stolen data TO the C2 server -> Ratio > 1 (send more than you receive).
        
        if total_bytes_received > 0:
            ratio = total_bytes_sent / (total_bytes_received + EPSILON)
        else:
            ratio = total_bytes_sent / (1.0 + EPSILON)
        
        features[10] = min(ratio / 5.0, 1.0)

        # =====================================================================
        # Feature 11: Fano factor of connections per timestep
        # =====================================================================
        # Fano factor = variance / mean of connection counts per timestep.
        #
        # count how many connections happened in each timestep within
        # the window, then compute variance / mean of those counts.
        features[11] = self._compute_fano_factor(records) / NORM_FANO

        # =====================================================================
        # Feature 12: Decayed cumulative suspicious activity score
        # =====================================================================
        # This is the long-term memory that the rolling window can't provide.
        # It accumulates over the entire episode. Each malicious connection
        # adds 1.0 to the score. Each timestep, the score is multiplied by
        # 0.995.
        #

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