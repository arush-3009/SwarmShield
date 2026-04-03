"""
SwarmShield Traffic Recording & Feature Computation
==================================================

1. TrafficRecord: one network connection.
2. TrafficManager:
   - stores per-host outgoing / incoming traffic history
   - updates long-memory suspicion counters on Host objects
   - prunes old records
   - computes the 16 traffic observation features
   - generates legitimate traffic for all hosts each timestep
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Sequence

import numpy as np

from env.config import (
    NUM_HOSTS,
    OBSERVATION_WINDOW,
    SERVER_HOST_ID,
    EXTERNAL_C2_ID,
    HOST_TO_SUBNET,
    NORMAL_FLOWS_PER_STEP,
    NORMAL_BYTES_RANGE,
    NORMAL_SAME_SUBNET_PROB,
    NORMAL_SERVER_PROB,
    NORMAL_CROSS_SUBNET_PROB,
    NORMAL_PORTS,
    NUM_TRAFFIC_FEATURES,
    EPSILON,
    NORM_TOTAL_FLOWS,
    NORM_CONN_ATTEMPTS,
    NORM_UNIQUE_DEST_IPS,
    NORM_UNIQUE_DEST_PORTS,
    NORM_BYTES,
    NORM_FAILED_CONNS,
    NORM_UNIQUE_PEERS,
    NORM_INCOMING_SCANS,
    NORM_SERVER_CONTACTS,
    NORM_UNIQUE_SOURCE_IPS,
    NORM_UNIQUE_SOURCE_PORTS,
)

@dataclass(frozen=True)
class TrafficRecord:
    """
    One network connection record.

    - source_id: internal host 0..17
    - dest_id: internal host 0..17, or EXTERNAL_C2_ID (-1)
    - dest_port: destination service port
    - bytes_sent: payload sent by source
    - bytes_received: response bytes received by source (0 if failed)
    - success: whether the connection completed
    - timestamp: timestep index
    - is_malicious: hidden bookkeeping label, never shown directly to agents
    """

    source_id: int
    dest_id: int
    dest_port: int
    bytes_sent: int
    bytes_received: int
    success: bool
    timestamp: int
    is_malicious: bool = False



class TrafficManager:
    
    def __init__(self):
        self.outgoing_history: Dict[int, List[TrafficRecord]] = {}
        self.incoming_history: Dict[int, List[TrafficRecord]] = {}
        self.seen_peers: Dict[int, Set[int]] = {}
        self.reset()



    def reset(self) -> None:
        """Clear all histories and peer-tracking state for a new episode."""
        for host_id in range(NUM_HOSTS):
            self.outgoing_history[host_id] = []
            self.incoming_history[host_id] = []
            self.seen_peers[host_id] = set()

    
    def _rand_int_inclusive(self, rng, low: int, high: int) -> int:
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high + 1))
        if hasattr(rng, "sample"):
            return int(rng.randint(low, high))
        if hasattr(rng, "randint"):
            return int(rng.randint(low, high + 1))
        raise TypeError("rng must support integers(...), randint(...), or sample(...).")
    

    def _rand_float_01(self, rng) -> float:
        if hasattr(rng, "random"):
            return float(rng.random())
        if hasattr(rng, "random_sample"):
            return float(rng.random_sample())
        raise TypeError("rng must support random() or random_sample().")


    def _choice(self, rng, seq: Sequence[int]) -> int:
        if len(seq) == 0:
            raise ValueError("Cannot choose from an empty sequence.")
        if hasattr(rng, "choice"):
            return int(rng.choice(seq))
        if hasattr(rng, "sample"):
            return int(rng.sample(list(seq), k=1)[0])
        raise TypeError("rng must support choice(...) or sample(...).")



    def _looks_like_scan_probe(self, record: TrafficRecord) -> bool:
    
        if record.dest_id == EXTERNAL_C2_ID:
            return False

        if record.bytes_sent > 128:
            return False

        if not (1 <= record.dest_port <= 1024):
            return False

        return True

    def add_record(self, record: TrafficRecord, reached_destination: bool, network) -> None:
        
        source_id = record.source_id
        source_host = network.get_host(source_id)

        # always record from source perspective.
        self.outgoing_history[source_id].append(record)

        # source-side long-memory updates.
        if not record.success:
            source_host.note_failed_outgoing()

        if record.dest_id not in self.seen_peers[source_id]:
            self.seen_peers[source_id].add(record.dest_id)
            source_host.note_unique_peer_contact()

        if record.dest_id == SERVER_HOST_ID:
            source_host.note_server_contact()

        # destination-side recording only if packet reached real internal host.
        if reached_destination:
            if record.dest_id == EXTERNAL_C2_ID:
                raise ValueError(
                    "reached_destination=True is invalid for EXTERNAL_C2_ID. "
                    "External C2 has no incoming host history."
                )

            dest_id = record.dest_id
            dest_host = network.get_host(dest_id)

            self.incoming_history[dest_id].append(record)

            if source_id not in self.seen_peers[dest_id]:
                self.seen_peers[dest_id].add(source_id)
                dest_host.note_unique_peer_contact()

            if self._looks_like_scan_probe(record):
                dest_host.note_incoming_scan()

    def record_connection(
        self,
        network,
        source_id: int,
        dest_id: int,
        dest_port: int,
        bytes_sent: int,
        bytes_received: int,
        success: bool,
        timestamp: int,
        is_malicious: bool,
        reached_destination: bool,
    ) -> TrafficRecord:

        record = TrafficRecord(
            source_id=source_id,
            dest_id=dest_id,
            dest_port=dest_port,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            success=success,
            timestamp=timestamp,
            is_malicious=is_malicious,
        )
        self.add_record(record, reached_destination, network)
        return record

    
    def prune_old_records(self, current_timestep: int) -> None:
        
        cutoff = current_timestep - OBSERVATION_WINDOW

        for host_id in range(NUM_HOSTS):
            fresh_outgoing: List[TrafficRecord] = []
            for record in self.outgoing_history[host_id]:
                if record.timestamp > cutoff:
                    fresh_outgoing.append(record)
            self.outgoing_history[host_id] = fresh_outgoing

            fresh_incoming: List[TrafficRecord] = []
            for record in self.incoming_history[host_id]:
                if record.timestamp > cutoff:
                    fresh_incoming.append(record)
            self.incoming_history[host_id] = fresh_incoming

    
    def _sample_normal_destination(self, host_id: int, network, rng) -> int:
        """
        Sample one legitimate-traffic destination for one source host.
        """
        
        same_subnet_ids = network.get_same_subnet_host_ids(host_id, include_self=False)
        cross_subnet_ids = network.get_cross_subnet_host_ids(host_id, include_server=False)

        destination_pools: List[Sequence[int]] = []
        weights: List[float] = []

        if len(same_subnet_ids) > 0:
            destination_pools.append(same_subnet_ids)
            weights.append(NORMAL_SAME_SUBNET_PROB)

        if host_id != SERVER_HOST_ID:
            destination_pools.append([SERVER_HOST_ID])
            weights.append(NORMAL_SERVER_PROB)

        if len(cross_subnet_ids) > 0:
            destination_pools.append(cross_subnet_ids)
            weights.append(NORMAL_CROSS_SUBNET_PROB)

        if len(destination_pools) == 0:
            raise RuntimeError(f"No valid legitimate-traffic destinations for host {host_id}.")

        total_weight = 0.0
        for weight in weights:
            total_weight += weight

        roll = self._rand_float_01(rng) * total_weight
        cumulative = 0.0

        for pool, weight in zip(destination_pools, weights):
            cumulative += weight
            if roll < cumulative:
                return self._choice(rng, pool)

        #fallback
        return self._choice(rng, destination_pools[-1])

    def generate_normal_traffic(self, network, current_timestep: int, rng) -> None:
        """
        Generate legitimate traffic for all hosts for a timestep.
        """
        for host_id in range(NUM_HOSTS):
            num_flows = self._rand_int_inclusive(
                rng,
                NORMAL_FLOWS_PER_STEP[0],
                NORMAL_FLOWS_PER_STEP[1],
            )

            for _ in range(num_flows):
                dest_id = self._sample_normal_destination(host_id, network, rng)
                decision = network.decide_connection(host_id, dest_id)

                bytes_sent = self._rand_int_inclusive(
                    rng,
                    NORMAL_BYTES_RANGE[0],
                    NORMAL_BYTES_RANGE[1],
                )

                if decision.success:
                    bytes_received = self._rand_int_inclusive(
                        rng,
                        bytes_sent,
                        bytes_sent * 2,
                    )
                else:
                    bytes_received = 0

                self.record_connection(
                    network=network,
                    source_id=host_id,
                    dest_id=dest_id,
                    dest_port=self._choice(rng, NORMAL_PORTS),
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    success=decision.success,
                    timestamp=current_timestep,
                    is_malicious=False,
                    reached_destination=decision.reached_destination,
                )


    def compute_features(self, host_id: int, network) -> np.ndarray:
        """
        Compute the 16 traffic features for one host which are then used for detection of C2/suspicious activity.
        """
        features = np.zeros(NUM_TRAFFIC_FEATURES, dtype=np.float32)

        host = network.get_host(host_id)
        outgoing_records = self.outgoing_history[host_id]
        incoming_records = self.incoming_history[host_id]

        #outgoing
        
        out_total_attempts = len(outgoing_records)
        out_successful = 0
        out_failed = 0
        out_unique_dests: Set[int] = set()
        out_unique_ports: Set[int] = set()
        out_bytes_sent = 0
        out_bytes_received = 0
        out_cross_subnet = 0

        for record in outgoing_records:
            if record.success:
                out_successful += 1
            else:
                out_failed += 1

            out_unique_dests.add(record.dest_id)
            out_unique_ports.add(record.dest_port)
            out_bytes_sent += record.bytes_sent
            out_bytes_received += record.bytes_received

            if record.dest_id == EXTERNAL_C2_ID:
                out_cross_subnet += 1
            else:
                source_subnet = HOST_TO_SUBNET[record.source_id]
                dest_subnet = HOST_TO_SUBNET[record.dest_id]
                if source_subnet != dest_subnet:
                    out_cross_subnet += 1

        features[0] = out_successful / NORM_TOTAL_FLOWS
        features[1] = out_total_attempts / NORM_CONN_ATTEMPTS

        if out_total_attempts > 0:
            features[2] = out_failed / (out_total_attempts + EPSILON)
            features[7] = out_cross_subnet / (out_total_attempts + EPSILON)
        else:
            features[2] = 0.0
            features[7] = 0.0

        features[3] = len(out_unique_dests) / NORM_UNIQUE_DEST_IPS
        features[4] = len(out_unique_ports) / NORM_UNIQUE_DEST_PORTS
        features[5] = out_bytes_sent / NORM_BYTES
        features[6] = out_bytes_received / NORM_BYTES

        #incoming
        
        in_total_attempts = len(incoming_records)
        in_unique_sources: Set[int] = set()
        in_unique_ports: Set[int] = set()
        in_bytes_received = 0

        for record in incoming_records:
            in_unique_sources.add(record.source_id)
            in_unique_ports.add(record.dest_port)

            in_bytes_received += record.bytes_sent

        features[8] = in_total_attempts / NORM_CONN_ATTEMPTS
        features[9] = len(in_unique_sources) / NORM_UNIQUE_SOURCE_IPS
        features[10] = len(in_unique_ports) / NORM_UNIQUE_SOURCE_PORTS
        features[11] = in_bytes_received / NORM_BYTES


        features[12] = host.decayed_failed_outgoing / NORM_FAILED_CONNS
        features[13] = host.decayed_unique_peers / NORM_UNIQUE_PEERS
        features[14] = host.decayed_incoming_scans / NORM_INCOMING_SCANS
        features[15] = host.decayed_server_contacts / NORM_SERVER_CONTACTS

        np.clip(features, 0.0, 1.0, out=features)
        return features


assert NUM_TRAFFIC_FEATURES == 16
assert abs(
    NORMAL_SAME_SUBNET_PROB + NORMAL_SERVER_PROB + NORMAL_CROSS_SUBNET_PROB - 1.0
) < 1e-9
assert SERVER_HOST_ID == 17
assert EXTERNAL_C2_ID == -1