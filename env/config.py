"""
SwarmShield Environment Configuration
======================================

The environment simulates a botnet attack on Dunder Mifflin's office network.
- 18 computers across 6 subnets/departments (subnets).
- An automated botnet infects computers and tries to steal data from the file server.
- 3 RL agents patrol the network, observing traffic and blocking infections.
- Agents are mobile, they choose which computer to monitor and when to act.

This config defines the network layout, attacker behavior, defender actions, reward structure, and all simulation parameters.
"""

import numpy as np


# =============================================================================
# NETWORK TOPOLOGY
# =============================================================================
# The network is Dunder Mifflin's office.
# Each department is a "subnet", i.e. a group of computers connected to the same local switch. Traffic within a 
# subnet flows directly. Traffic between subnets must pass through the central office router, which is the chokepoint 
# where firewall rules (iptables) can block suspicious traffic.

NUM_HOSTS = 18        # Total computers in the office
NUM_SUBNETS = 6       # Number of subnet/departments (1 subnet with just the File Server)
NUM_AGENTS = 3        # Number of defensive RL agents

# Each subnet is defined by its name and which host IDs belong to it.
# Host IDs are 0-17, assigned sequentially by department.

SUBNET_NAMES = [
    "Sales",          # Subnet 0
    "Accounting",     # Subnet 1
    "Back Desks",    # Subnet 2
    "Management",     # Subnet 3
    "Conference",     # Subnet 4
    "Server Closet",  # Subnet 5
]


# This is the fundamental network structure -> which hosts belong to which subnet.

SUBNET_HOSTS = {
    0: [0, 1, 2, 3, 4],       # Sales: Jim, Dwight, Stanley, Phyllis, Andy
    1: [5, 6, 7],              # Accounting: Angela, Oscar, Kevin
    2: [8, 9, 10, 11],         # Back Desks: 4 computers
    3: [12, 13],               # Management: Michael, Pam
    4: [14, 15, 16],           # Conference: 3 laptops
    5: [17],                   # Server Closet: File Server
}

# Human-readable names for each host
HOST_NAMES = [
    "Jim",           # 0
    "Dwight",        # 1
    "Stanley",       # 2
    "Phyllis",       # 3
    "Andy",          # 4
    "Angela",        # 5
    "Oscar",         # 6
    "Kevin",         # 7
    "BackDesk1",    # 8
    "BackDesk2",    # 9
    "BackDesk3",    # 10
    "BackDesk4",    # 11
    "Michael",       # 12
    "Pam",           # 13
    "ConfLaptop1",   # 14
    "ConfLaptop2",   # 15
    "ConfLaptop3",   # 16
    "FileServer",    # 17
]

# Reverse lookup: given a host ID, which subnet is it in?
HOST_TO_SUBNET = {}
for subnet_id, host_list in SUBNET_HOSTS.items():
    for host_id in host_list:
        HOST_TO_SUBNET[host_id] = subnet_id


# The file server is the HIGH VALUE TARGET.
# The botnet's ultimate goal is to compromise this machine.
# If attack traffic reaches the server and isn't blocked, the attacker wins.
SERVER_HOST_ID = 17


# =============================================================================
# VULNERABILITY SCORES
# =============================================================================
# When the botnet tries to infect a computer, it doesn't always succeed.
# Each computer has a "vulnerability score" -> the probability that an infection attempt works. 
# This models real-world behavious and scenarios:
#   - Some machines have outdated software (high vulnerability).
#   - Some are well-patched by IT (low vulnerability).
#   - The server is hardened, often maintained very well, but not invincible.
#
# These are randomized at the start of each episode.
# The agent never sees these numbers directly, it detects infections
# from observing traffic patterns, not from knowing who's vulnerable / more susceptible to be infected.

VULN_RANGE_REGULAR = (0.2, 0.7)   # Regular hosts: random in this range -> easier to infect than server
VULN_RANGE_SERVER = (0.05, 0.15)  # Server: random in this range -> always hard to infect


# =============================================================================
# EPISODE STRUCTURE
# =============================================================================
# An episode is one complete simulation of an attack.
# It starts with a few infected machines and ends when either:
# - The agents contain all infections (defenders win)
# - The server gets compromised (attackers win)
# - Time runs out (partial result based on network health)

MAX_TIMESTEPS = 200               # Maximum steps per episode
INITIAL_INFECTIONS = 2            # How many hosts start infected (never the server)

# Rolling observation window: traffic features are computed over the last N timesteps, not just the 
# current one. This gives enough data points to compute meaningful statistics like entropy and the fano factor -> variance / mean.
OBSERVATION_WINDOW = 10


# =============================================================================
# ATTACKER CONFIGURATION
# =============================================================================

# The botnet is automated. It follows the C2 lifecycle:

# Phase 1 -> Beaconing: infected hosts periodically "phone home" to the
#           command-and-control C2 server. Small, periodic, same destination.
# Phase 2 -> Scanning: infected hosts probe neighbors to find new targets.
#           Many connection attempts, most fail, many unique destinations.
# Phase 3 -> Infection: successful scans lead to new infections.
# Phase 4 -> Attack: once enough hosts are infected, the botnet attacks
#           the file server to steal data.

# Beaconing parameters:
# In real botnets, the beacon interval isn't perfectly regular — the malware
# adds random jitter to its timing so that security tools can't detect
# the perfect periodicity. Here, beacons happen every BASE ± JITTER timesteps.
# Also, beacons use port 443 to try to pose as regular https traffic.
BEACON_INTERVAL_BASE = 3          # Average timesteps between beacons
BEACON_JITTER = 1                 # Random variation/jitter: ±1 timestep
BEACON_BYTES = (100, 300)         # Payload size range (small, consistent)
BEACON_PORT = 443                 # Port used (disguised as HTTPS)

# Scanning parameters
# Scanning starts after a host has been infected for a few timesteps.
# The zombie probes other hosts with connection attempts. Most fail.
SCAN_DELAY = 3                    # Timesteps after infection before scanning begins
SCAN_PROBES_PER_STEP = (2, 5)    # Range of how many hosts to probe per timestep
SCAN_SAME_SUBNET_PROB = 0.7      # 70% chance each probe targets same subnet (remaining 30% targets cross-subnet i.e. over the router)

# Attack parameters
# The botnet launches its attack on the server once enough hosts are infected.
ATTACK_INFECTION_THRESHOLD = 4    # Minimum infected hosts before attack begins
ATTACK_CONNECTIONS_PER_STEP = (10, 30)   # Connections per infected host during attack
ATTACK_BYTES_PER_CONNECTION = (5000, 50000)  # Bytes per attack connection (large)

# Server damage
# The server doesn't get compromised instantly. Attack traffic accumulates
# "damage." If total damage exceeds the threshold, the server is compromised.
# This gives agents a window to react even after the attack begins.
SERVER_DAMAGE_PER_CONNECTION = 1.0
SERVER_DAMAGE_THRESHOLD = 100.0   # Total damage needed to compromise server


# =============================================================================
# NORMAL TRAFFIC GENERATION
# =============================================================================
# Every clean, unblocked host generates simulated "normal" traffic each timestep.
# This is the background noise that the botnet hides in.
# The agent learns to distinguish normal traffic and activity from malicious.

NORMAL_FLOWS_PER_STEP = (1, 4)    # How many connections each host makes per step
NORMAL_BYTES_RANGE = (200, 5000)  # Bytes per normal connection

# Where normal traffic goes:
NORMAL_SAME_SUBNET_PROB = 0.50    # 50% to coworkers in same department
NORMAL_SERVER_PROB = 0.30         # 30% to the file server (accessing files)
NORMAL_CROSS_SUBNET_PROB = 0.20   # 20% to other departments

# Ports used by normal traffic (standard web/file services)
NORMAL_PORTS = [80, 443, 445, 8080]


# =============================================================================
# AGENT ACTIONS
# =============================================================================
# Each agent picks one of these 11 actions every timestep.
# The actions range from passive (observe) to aggressive (quarantine).
# Stronger actions stop threats better but cause more damage if wrong.

NUM_ACTIONS = 11

# Action IDs
ACTION_OBSERVE = 0            # Stay and watch. Free, no effect.
ACTION_MOVE_WITHIN_SUBNET = 1 # Move to another node in same subnet. 1 step blind.
ACTION_MOVE_TO_SUBNET_0 = 2   # Move to Sales
ACTION_MOVE_TO_SUBNET_1 = 3   # Move to Accounting
ACTION_MOVE_TO_SUBNET_2 = 4   # Move to Back Desks
ACTION_MOVE_TO_SUBNET_3 = 5   # Move to Management
ACTION_MOVE_TO_SUBNET_4 = 6   # Move to Conference
ACTION_MOVE_TO_SUBNET_5 = 7   # Move to Server Closet
ACTION_BLOCK = 8              # Block host's cross-subnet traffic (iptables DROP) -> any packet the host tries to send over the 
                              # router to another subnet will be silently deleted without any alerts.
ACTION_QUARANTINE = 9         # Block ALL traffic to/from host (full isolation)
ACTION_UNBLOCK = 10           # Remove all firewall rules for this host

# Movement costs (in timesteps of being blind. Being blind implies -> agent can't observe or act)
MOVE_WITHIN_SUBNET_COST = 1   # 1 timestep blind
MOVE_CROSS_SUBNET_COST = 2    # 2 timesteps blind


# =============================================================================
# OBSERVATION SPACE
# =============================================================================
# Each agent sees a partial observation of the environment:
# - Traffic features at its current node (computed over the rolling window)
# - Global information (its position, other agents' positions, network status)
#
# The agent CANNOT see:
# - Other nodes' traffic (it must physically move there to observe)
# - Vulnerability scores (hidden — must infer from behavior)
# - Which hosts are infected (must detect from traffic patterns)

# Per-node traffic features (13 values)
# These are computed from the rolling window at the agent's current node.
NUM_TRAFFIC_FEATURES = 13

# Feature indices
FEAT_TOTAL_FLOWS = 0
FEAT_TOTAL_CONN_ATTEMPTS = 1
FEAT_FAILED_CONN_RATE = 2
FEAT_UNIQUE_DEST_IPS = 3
FEAT_UNIQUE_DEST_PORTS = 4
FEAT_ENTROPY_DEST_IPS = 5
FEAT_ENTROPY_INTER_ARRIVAL = 6
FEAT_SYN_ACK_RATIO = 7
FEAT_BYTES_SENT = 8
FEAT_BYTES_RECEIVED = 9
FEAT_SENT_RECV_RATIO = 10
FEAT_FANO = 11
FEAT_DECAY_SUSPICIOUS = 12

# Global features:
# - Current node one-hot (18 values)
# - Agent 2 position one-hot (18 values)
# - Agent 3 position one-hot (18 values)
# - Num quarantined hosts (1 value, normalized)
# - Num active block rules (1 value, normalized)
# - Episode progress (1 value, current_step / max_steps)
# - In transit flag (1 value)
NUM_GLOBAL_FEATURES = (NUM_HOSTS * 3) + 4   # = 58

# Total observation size per agent
OBSERVATION_SIZE = NUM_TRAFFIC_FEATURES + NUM_GLOBAL_FEATURES  # = 71


# =============================================================================
# REWARD FUNCTION
# =============================================================================
# Two goals in tension:
# 1. Stop the botnet (catch infections, protect the server)
# 2. Keep the office running (don't block legitimate users)
#
# An agent that blocks everything is safe but useless (office can't work).
# An agent that blocks nothing lets the botnet win.
# The reward structure forces agents to find the balance.

# Per-timestep continuous rewards (every agent, every step)
REWARD_HEALTHY_HOST = 1.0         # Per clean, unblocked, operational host
REWARD_UNCONTAINED_INFECTION = -3.0   # Per infected host NOT quarantined/blocked
REWARD_FALSE_BLOCK = -0.5        # Per clean host that IS blocked/quarantined
REWARD_AGENT_OVERLAP = -2.0      # Per pair of agents at the same node

# One-time event rewards
REWARD_CORRECT_QUARANTINE = 50.0  # Quarantine an actually infected host
REWARD_CORRECT_BLOCK = 20.0      # Block an actually infected host
REWARD_FALSE_QUARANTINE = -40.0   # Quarantine a clean host
REWARD_FALSE_BLOCK = -15.0       # Block a clean host
REWARD_CORRECT_UNBLOCK = 10.0    # Unblock a host that was falsely blocked
REWARD_BAD_UNBLOCK = -5.0        # Unblock a host that was correctly blocked

# Movement costs
REWARD_MOVE_WITHIN_SUBNET = -0.5        # Small cost for within-subnet move
REWARD_MOVE_CROSS_SUBNET = -1.0         # Bigger cost for cross-subnet move

# Episode-ending rewards
REWARD_SERVER_COMPROMISED = -200.0    # Catastrophic failure
REWARD_ALL_CONTAINED = 100.0          # Complete victory
REWARD_SURVIVED = 50.0                # Survived full episode, server safe


# =============================================================================
# TRAFFIC FEATURE COMPUTATION
# =============================================================================
# Parameters for computing observation features from raw traffic data.

# Decay factor for cumulative suspicious activity score.
# Applied as: score = score * (DECAY_FACTOR ^ timesteps_elapsed) + new_events
# At 0.995: after 10 timesteps, old score retains 0.995^10 = 0.95 (5% fade)
#           after 100 timesteps, 0.995^100 = 0.61 (39% fade)
#           after 500 timesteps, 0.995^500 = 0.08 (92% fade)
SUSPICIOUS_DECAY_FACTOR = 0.995

# Small epsilon to avoid division by zero in ratios and entropy
EPSILON = 1e-8

# Normalization constants for observation features.
# These scale raw values to approximately [0, 1] range so the neural
# network can process them easily. We pick values that represent
# "roughly the maximum expected value" for each feature.
NORM_TOTAL_FLOWS = 50.0
NORM_CONN_ATTEMPTS = 50.0
NORM_UNIQUE_DEST_IPS = 18.0       # Can't exceed total hosts
NORM_UNIQUE_DEST_PORTS = 10.0
NORM_ENTROPY = 3.0                # ln(18) ≈ 2.89, so 3.0 covers it
NORM_BYTES = 100000.0
NORM_FANO = 10.0
NORM_SUSPICIOUS = 20.0


# =============================================================================
# PPO HYPERPARAMETERS (used later in training)
# =============================================================================
# These are the same standard PPO values from the papers we studied.
# They're here so everything is in one place.

PPO_LEARNING_RATE_ACTOR = 3e-4
PPO_LEARNING_RATE_CRITIC = 1e-3
PPO_GAMMA = 0.99                  # Discount factor
PPO_GAE_LAMBDA = 0.95             # GAE parameter for advantage estimation
PPO_CLIP_EPSILON = 0.2            # Clipping parameter
PPO_ENTROPY_COEFF = 0.01          # Entropy bonus for exploration
PPO_EPOCHS = 5                    # Update epochs per batch
PPO_BATCH_SIZE = 64               # Minibatch size
PPO_HORIZON = 2048                # Steps collected before each update

# Network architecture
HIDDEN_SIZE_1 = 128               # First hidden layer
HIDDEN_SIZE_2 = 64                # Second hidden layer