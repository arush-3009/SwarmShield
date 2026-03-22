"""
SwarmShield Environment Configuration
====================================

This file defines controllable constants for:
- Network topology
- Host / subnet layout
- Attacker behavior
- Legitimate traffic generation
- Action space
- Observation layout
- Reward values
- PPO hyperparameters

The environment simulates a botnet attack on an office network:
- 18 computers across 6 subnets (departments)
- 1 file server (host 17)
- 3 RL defender agents
- A scripted botnet that beacons, scans, spreads, and eventually attacks the server
"""

# =============================================================================
# NETWORK TOPOLOGY
# =============================================================================

NUM_HOSTS = 18
NUM_SUBNETS = 6
NUM_AGENTS = 3

SUBNET_NAMES = [
    "Sales",
    "Accounting",
    "Back Desks",
    "Management",
    "Conference",
    "Server Closet",
]

# Subnet sizes: 5, 3, 4, 2, 3, 1
SUBNET_HOSTS = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13],
    4: [14, 15, 16],
    5: [17],
}

HOST_NAMES = [
    "Jim",         # 0
    "Dwight",      # 1
    "Stanley",     # 2
    "Phyllis",     # 3
    "Andy",        # 4
    "Angela",      # 5
    "Oscar",       # 6
    "Kevin",       # 7
    "BackDesk1",   # 8
    "BackDesk2",   # 9
    "BackDesk3",   # 10
    "BackDesk4",   # 11
    "Michael",     # 12
    "Pam",         # 13
    "ConfLaptop1", # 14
    "ConfLaptop2", # 15
    "ConfLaptop3", # 16
    "FileServer",  # 17
]

HOST_IDS = list(range(NUM_HOSTS))
SERVER_HOST_ID = 17
REGULAR_HOST_IDS = [host_id for host_id in HOST_IDS if host_id != SERVER_HOST_ID]
SERVER_SUBNET_ID = 5

# External C2 server is not a real host in the environment.
EXTERNAL_C2_ID = -1

# Reverse mapping: host -> subnet
HOST_TO_SUBNET = {}
for subnet_id, host_list in SUBNET_HOSTS.items():
    for host_id in host_list:
        HOST_TO_SUBNET[host_id] = subnet_id


# =============================================================================
# HOST STATE MODEL
# =============================================================================

# Infection is a separate boolean in the environment.
# These constants are only for containment state.
CONTAINMENT_NONE = 0
CONTAINMENT_BLOCKED = 1
CONTAINMENT_QUARANTINED = 2


# =============================================================================
# VULNERABILITY SCORES
# =============================================================================

# Assigned fresh at each episode reset for regular hosts only.
VULN_RANGE_REGULAR = (0.03, 0.10)


# =============================================================================
# EPISODE STRUCTURE
# =============================================================================

MAX_TIMESTEPS = 200
INITIAL_INFECTIONS = 1
OBSERVATION_WINDOW = 12


# =============================================================================
# ATTACKER CONFIGURATION
# =============================================================================

# --- Beaconing ---
# Immediately on infection, then every random(2, 4) steps.
BEACON_INTERVAL_BASE = 3
BEACON_JITTER = 1
BEACON_PORT = 443
BEACON_BYTES_SENT = (100, 300)
BEACON_BYTES_RECEIVED = (50, 150)

# --- Scanning / Lateral Spread ---
# Starts 6 steps after infection. 3-5 probes per step.
# 80% same-subnet, 20% cross-subnet. File server is never a scan target.
SCAN_DELAY = 6
SCAN_PROBES_PER_STEP = (3, 5)
SCAN_SAME_SUBNET_PROB = 0.80
SCAN_PROBE_PORT_RANGE = (1, 1024)
SCAN_PROBE_BYTES = (40, 80)
SCAN_PROBE_RECV_ALREADY_INFECTED = (40, 80)
INFECTION_PAYLOAD_BYTES_SENT = (2000, 8000)
INFECTION_PAYLOAD_BYTES_RECEIVED = (100, 300)

# --- Server Attack ---
# Triggered once 5+ active uncontained infected hosts exist.
ATTACK_INFECTION_THRESHOLD = 5
ATTACK_CONNECTIONS_PER_STEP = (1, 3)
ATTACK_DEST_PORT = 443
ATTACK_BYTES_PER_CONNECTION = (4000, 20000)
SERVER_DAMAGE_PER_CONNECTION = 1.0
SERVER_DAMAGE_THRESHOLD = 300.0


# =============================================================================
# NORMAL (LEGITIMATE) TRAFFIC GENERATION
# =============================================================================

# Every host generates legitimate traffic every timestep, whether clean or infected.
NORMAL_FLOWS_PER_STEP = (1, 3)
NORMAL_BYTES_RANGE = (200, 4000)

# Destination choice probabilities.
NORMAL_SAME_SUBNET_PROB = 0.55
NORMAL_SERVER_PROB = 0.25
NORMAL_CROSS_SUBNET_PROB = 0.20

NORMAL_PORTS = [80, 443, 445, 8080]


# =============================================================================
# AGENT ACTIONS
# =============================================================================

# 22 discrete actions:
#   0       -> observe
#   1..18   -> move to host 0..17
#   19      -> block current host
#   20      -> quarantine current host
#   21      -> unblock current host
NUM_ACTIONS = 22

ACTION_OBSERVE = 0

ACTION_MOVE_BASE = 1
ACTION_MOVE_TO_HOST_0 = ACTION_MOVE_BASE
ACTION_MOVE_TO_HOST_17 = ACTION_MOVE_BASE + NUM_HOSTS - 1
ACTION_MOVE_LAST = ACTION_MOVE_TO_HOST_17

ACTION_BLOCK = ACTION_MOVE_LAST + 1
ACTION_QUARANTINE = ACTION_BLOCK + 1
ACTION_UNBLOCK = ACTION_QUARANTINE + 1

# Movement costs are blind timesteps in transit.
MOVE_WITHIN_SUBNET_COST = 1
MOVE_CROSS_SUBNET_COST = 2


# =============================================================================
# OBSERVATION SPACE
# =============================================================================

# 16 traffic features + 61 global features = 77 total.
NUM_TRAFFIC_FEATURES = 16

# --- Traffic features (0-15) ---
FEAT_OUT_SUCCESSFUL_FLOWS = 0
FEAT_OUT_TOTAL_CONN_ATTEMPTS = 1
FEAT_OUT_FAILED_CONN_RATE = 2
FEAT_OUT_UNIQUE_DEST_IPS = 3
FEAT_OUT_UNIQUE_DEST_PORTS = 4
FEAT_OUT_BYTES_SENT = 5
FEAT_OUT_BYTES_RECEIVED = 6
FEAT_OUT_CROSS_SUBNET_RATE = 7

FEAT_IN_TOTAL_CONN_ATTEMPTS = 8
FEAT_IN_UNIQUE_SOURCE_IPS = 9
FEAT_IN_UNIQUE_SOURCE_PORTS = 10
FEAT_IN_BYTES_RECEIVED = 11

FEAT_DECAY_FAILED_CONNS = 12
FEAT_DECAY_UNIQUE_PEERS = 13
FEAT_DECAY_INCOMING_SCANS = 14
FEAT_DECAY_SERVER_CONTACTS = 15

# --- Global features (16-76) ---
FEAT_SELF_POS_START = 16
FEAT_SELF_POS_END = FEAT_SELF_POS_START + NUM_HOSTS          # exclusive end = 34

FEAT_OTHER_AGENT_1_POS_START = FEAT_SELF_POS_END
FEAT_OTHER_AGENT_1_POS_END = FEAT_OTHER_AGENT_1_POS_START + NUM_HOSTS  # 52

FEAT_OTHER_AGENT_2_POS_START = FEAT_OTHER_AGENT_1_POS_END
FEAT_OTHER_AGENT_2_POS_END = FEAT_OTHER_AGENT_2_POS_START + NUM_HOSTS  # 70

FEAT_FRACTION_QUARANTINED = FEAT_OTHER_AGENT_2_POS_END       # 70
FEAT_FRACTION_BLOCKED = FEAT_FRACTION_QUARANTINED + 1        # 71
FEAT_SERVER_DAMAGE = FEAT_FRACTION_BLOCKED + 1               # 72
FEAT_EPISODE_PROGRESS = FEAT_SERVER_DAMAGE + 1               # 73
FEAT_CURRENT_HOST_BLOCKED = FEAT_EPISODE_PROGRESS + 1        # 74
FEAT_CURRENT_HOST_QUARANTINED = FEAT_CURRENT_HOST_BLOCKED + 1  # 75
FEAT_IN_TRANSIT = FEAT_CURRENT_HOST_QUARANTINED + 1          # 76

NUM_GLOBAL_FEATURES = (NUM_HOSTS * 3) + 7
OBSERVATION_SIZE = NUM_TRAFFIC_FEATURES + NUM_GLOBAL_FEATURES


# =============================================================================
# REWARD FUNCTION
# =============================================================================

# Win condition: every infected host must be QUARANTINED.
# Block does NOT count, because blocked infected hosts can still spread locally.

# --- Dense shared reward (all agents, every step) ---
REWARD_HEALTHY_HOST = 0.03
REWARD_INFECTED_UNCONTAINED = -2.50
REWARD_INFECTED_BLOCKED = -0.50
REWARD_INFECTED_QUARANTINED = 0.00
REWARD_FALSE_BLOCK_PER_STEP = -0.30
REWARD_FALSE_QUARANTINE_PER_STEP = -0.50
REWARD_AGENT_OVERLAP = -0.10
REWARD_NEW_INFECTION = -6.00
REWARD_SERVER_DAMAGE = -0.25

# --- Event rewards (acting agent only) ---
REWARD_CORRECT_BLOCK = 6.00
REWARD_CORRECT_QUARANTINE = 15.00
REWARD_QUARANTINE_UPGRADE = 6.00
REWARD_FALSE_BLOCK = -2.00
REWARD_FALSE_QUARANTINE = -4.00
REWARD_FALSE_QUARANTINE_UPGRADE = -2.00

REWARD_CORRECT_UNBLOCK = 1.50
REWARD_BAD_UNBLOCK = -4.00

REWARD_MOVE_WITHIN_SUBNET = -0.05
REWARD_MOVE_CROSS_SUBNET = -0.12

# --- Terminal rewards (all agents) ---
REWARD_SERVER_COMPROMISED = -40.00
REWARD_ALL_QUARANTINED = 40.00
REWARD_SURVIVED = 0.00


# =============================================================================
# TRAFFIC FEATURE NORMALIZATION
# =============================================================================

# All normalized values are clipped to [0, 1] in code.
SUSPICIOUS_DECAY_FACTOR = 0.97
EPSILON = 1e-8

NORM_TOTAL_FLOWS = 30.0
NORM_CONN_ATTEMPTS = 30.0
NORM_UNIQUE_DEST_IPS = 18.0
NORM_UNIQUE_DEST_PORTS = 12.0
NORM_BYTES = 60000.0
NORM_FAILED_CONNS = 20.0
NORM_UNIQUE_PEERS = 20.0
NORM_INCOMING_SCANS = 20.0
NORM_SERVER_CONTACTS = 20.0

# Helpful aliases for incoming features (same values, clearer names).
NORM_UNIQUE_SOURCE_IPS = NORM_UNIQUE_DEST_IPS
NORM_UNIQUE_SOURCE_PORTS = NORM_UNIQUE_DEST_PORTS


# =============================================================================
# PPO / IPPO HYPERPARAMETERS
# =============================================================================

PPO_LEARNING_RATE_ACTOR = 3e-4
PPO_LEARNING_RATE_CRITIC = 1e-3
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEFF = 0.05
PPO_EPOCHS = 5
PPO_BATCH_SIZE = 128
PPO_HORIZON = 1024

HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64


# =============================================================================
# CONFIG SANITY CHECKS
# =============================================================================

assert len(SUBNET_NAMES) == NUM_SUBNETS
assert len(HOST_NAMES) == NUM_HOSTS
assert sum(len(hosts) for hosts in SUBNET_HOSTS.values()) == NUM_HOSTS
assert set(HOST_TO_SUBNET.keys()) == set(HOST_IDS)

assert SERVER_HOST_ID == 17
assert SERVER_SUBNET_ID in SUBNET_HOSTS
assert SUBNET_HOSTS[SERVER_SUBNET_ID] == [SERVER_HOST_ID]

assert INITIAL_INFECTIONS >= 1
assert INITIAL_INFECTIONS <= len(REGULAR_HOST_IDS)

assert 0.0 <= SCAN_SAME_SUBNET_PROB <= 1.0
assert abs(
    NORMAL_SAME_SUBNET_PROB + NORMAL_SERVER_PROB + NORMAL_CROSS_SUBNET_PROB - 1.0
) < 1e-9

assert BEACON_INTERVAL_BASE - BEACON_JITTER >= 1

assert NUM_ACTIONS == 22
assert ACTION_UNBLOCK == NUM_ACTIONS - 1

assert NUM_TRAFFIC_FEATURES == 16
assert NUM_GLOBAL_FEATURES == 61
assert OBSERVATION_SIZE == 77
assert FEAT_IN_TRANSIT == OBSERVATION_SIZE - 1