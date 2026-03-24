# SwarmShield Environment Design Specification

This is the source-of-truth spec for the SwarmShield training environment.


## 1. Core Idea

1. 18 host computers across 6 subnets in a simulated office network.
2. A scripted botnet infects hosts and eventually attacks the file server.
3. 3 defender agents trained independently with PPO (IPPO).
4. Each agent has its own policy network, critic network, optimizer, and experience buffer.
5. No centralized critic, no shared parameters, no coded communication.
6. Any coordination is emergent only.


## 2. World Layout

```text
Subnet 0  Sales        : Host 0 (Jim), Host 1 (Dwight), Host 2 (Stanley), Host 3 (Phyllis), Host 4 (Andy)
Subnet 1  Accounting   : Host 5 (Angela), Host 6 (Oscar), Host 7 (Kevin)
Subnet 2  Back Desks   : Host 8 (BackDesk1), Host 9 (BackDesk2), Host 10 (BackDesk3), Host 11 (BackDesk4)
Subnet 3  Management   : Host 12 (Michael), Host 13 (Pam)
Subnet 4  Conference   : Host 14 (ConfLaptop1), Host 15 (ConfLaptop2), Host 16 (ConfLaptop3)
Subnet 5  Server Closet: Host 17 (FileServer)
```

18 hosts total. Subnet sizes: 5, 3, 4, 2, 3, 1. File server sits alone in subnet 5.


## 3. Connections

Every connection is a `TrafficRecord`:

- `source_id`: integer 0-17. Only real hosts initiate connections.
- `dest_id`: integer 0-17 for internal hosts, or -1 for external C2 server. Only beacons use -1.
- `dest_port`: integer. Normal traffic: [80, 443, 445, 8080]. Scans: random 1-1024. Beacons: 443.
- `bytes_sent`: depends on message type (Section 8).
- `bytes_received`: response bytes. Zero if connection failed.
- `success`: True if connection completed, False otherwise.
- `is_malicious`: hidden label. **Agent never sees this.** Internal bookkeeping only.
- `timestamp`: which timestep this occurred at.


## 4. Traffic Recording

Every record is stored from two perspectives when possible:

1. **Source's outgoing history** — always recorded.
2. **Destination's incoming history** — recorded only if the connection actually reached the destination.

Firewall-blocked connections (containment rules): record in source outgoing ONLY.

Non-firewall failures (e.g., exploit failed but packet arrived): record in BOTH source outgoing and destination incoming.

Records older than `OBSERVATION_WINDOW = 12` timesteps are pruned each step. All features are clipped to [0, 1] after normalization.


## 5. Host State Model

Two independent axes per host.

### 5.1 Infection State
- `infected = False` — clean.
- `infected = True` — malware present.

### 5.2 Containment State
- `CONTAINMENT_NONE` — no defender action.
- `CONTAINMENT_BLOCKED` — subnet-level firewall.
- `CONTAINMENT_QUARANTINED` — full isolation.

### 5.3 Combined Interpretation

1. **Clean + None**: Normal. Generates legit traffic. Can be scanned and infected.
2. **Infected + None**: Active botnet member. Generates legit AND malicious traffic. Dangerous.
3. **Infected + Blocked**: Same-subnet works freely (legit + malicious). Cross-subnet fails. Can still infect within subnet.
4. **Infected + Quarantined**: All traffic attempted, all fails. Fully neutralized.
5. **Clean + Blocked**: Legit same-subnet succeeds, cross-subnet fails. False positive.
6. **Clean + Quarantined**: All traffic fails. Severe false positive.

### 5.4 Infection Does Not Erase Containment

When a host gets infected (e.g., blocked clean host infected by same-subnet probe), `infected` becomes True but containment **does not change**. Only agent actions change containment.


## 6. Connection Success Logic

For any connection from sender to receiver, start with `success = True`, then check in order:

1. Sender is quarantined → `success = False`. Source outgoing only.
2. Sender is blocked AND receiver is different subnet or external → `success = False`. Source outgoing only.
3. Receiver is quarantined → `success = False`. Source outgoing only.
4. Receiver is blocked AND sender is different subnet → `success = False`. Source outgoing only.
5. Otherwise → `success = True`. Recorded bidirectionally.

This applies identically to legitimate and malicious traffic.

For scan probes: even if the connection succeeds (reaches target), infection only happens if target is clean and vulnerability roll passes.

### 6.1 Blocked (Subnet-Level Firewall)

Same-subnet traffic flows freely (subject to destination's state). Cross-subnet and external traffic fails.

Attacker on blocked infected host:
- C2 beacons: fail (external). Outgoing only.
- Same-subnet scans: proceed normally. Can infect.
- Cross-subnet scans: fail. Outgoing only.
- Server attacks: fail (cross-subnet). No damage.

### 6.2 Quarantined (Full Isolation)

All traffic fails in all directions.

Attacker on quarantined infected host:
- Everything attempted, everything fails. Outgoing only. No infections, no damage.

### 6.3 Summary Table

| Scenario | Same-subnet out | Cross-subnet out | Same-subnet in | Cross-subnet in |
|---|---|---|---|---|
| Uncontained | succeeds* | succeeds* | succeeds | succeeds |
| Blocked | succeeds* | fails | succeeds | fails |
| Quarantined | fails | fails | fails | fails |

*Subject to destination's containment state per rules 3-4 above.


## 7. Traffic Generation

Every timestep, in this order:

### 7.1 Legitimate Traffic (ALL hosts)

Every host generates `random(1, 3)` legitimate connections per step, whether clean or infected.

- Destination: 55% same-subnet, 25% file server, 20% cross-subnet.
- Port: random from [80, 443, 445, 8080].
- `bytes_sent`: random(200, 4000).
- `bytes_received`: random(bytes_sent, bytes_sent × 2) if successful, 0 if failed.
- `is_malicious = False`.
- `success`: per Section 6 logic.

### 7.2 Malicious Traffic (infected hosts)

Every infected host (regardless of containment) attempts malicious traffic. Containment determines what succeeds.

#### Beaconing

When a host is first infected, it sends a beacon immediately. The beacon timer then resets to random(2, 4) steps. Each timestep the timer decrements by 1. At 0, another beacon fires and timer resets.

- `source_id` = infected host, `dest_id = -1`, `dest_port = 443`.
- `bytes_sent`: random(100, 300). `bytes_received`: random(50, 150) if successful, 0 if failed.
- `is_malicious = True`.
- `success`: per Section 6 (C2 is external, fails if blocked or quarantined).

#### Scanning

Starts `SCAN_DELAY = 6` timesteps after infection. Each infected host sends `random(3, 5)` probes per step.

- Target: 80% same-subnet, 20% cross-subnet. File server is never a target.
- Port: random(1, 1024). `is_malicious = True`.
- `success` and recording: per Section 6.

Probe outcomes when connection reaches target:
- Target already infected: `bytes_sent` = random(40, 80), `bytes_received` = random(40, 80). No state change.
- Target clean, exploit fails: `bytes_sent` = random(40, 80), `bytes_received` = 0.
- Target clean, exploit succeeds: `bytes_sent` = random(2000, 8000), `bytes_received` = random(100, 300). Target infected, containment unchanged.

Exploit probability = target's vulnerability score.

#### Server Attack

Once `ATTACK_INFECTION_THRESHOLD = 5` active uncontained infected hosts exist, each sends attack connections to the file server.

- Connections per step: random(1, 3).
- `dest_id = 17`, `dest_port = 443`.
- `bytes_sent`: random(4000, 20000). `bytes_received`: 0.
- `is_malicious = True`.
- Each successful connection adds `1.0` damage. Server falls at `300.0`.

### 7.3 Prune Old Records

Records older than 12 timesteps removed.


## 8. Byte Size by Message Type

| Message Type | bytes_sent | bytes_received (success) | bytes_received (fail) |
|---|---|---|---|
| Legitimate | random(200, 4000) | random(bytes_sent, bytes_sent×2) | 0 |
| Scan probe (no infection) | random(40, 80) | 0 or random(40, 80) | 0 |
| Scan probe (infection) | random(2000, 8000) | random(100, 300) | N/A |
| C2 beacon | random(100, 300) | random(50, 150) | 0 |
| Server attack | random(4000, 20000) | 0 | 0 |


## 9. Vulnerability Scores

Regular hosts (0-16): `VULN_RANGE_REGULAR = (0.03, 0.10)` assigned at episode reset.

File server (host 17) is never a scan target. Vulnerability unused.


## 10. Attacker Lifecycle

1. **Initial infection**: 1 random non-server host infected at reset.
2. **Beaconing**: Immediately on infection. Interval 2-4 steps.
3. **Scanning**: 6 steps post-infection. 3-5 probes/step, 80% same-subnet.
4. **Lateral spread**: New infections start their own beacon + scan cycles.
5. **Server attack**: At 5+ active uncontained hosts, 1-3 attack connections/step each.


## 11. Defender Agents

### 11.1 Positions
Each agent occupies one host. Observation comes from that host's traffic.

### 11.2 Movement
- Within-subnet: 1 blind timestep.
- Cross-subnet: 2 blind timesteps.
- Move to current host: no-op, no cost.
- During transit: no observation, no actions.

### 11.3 Containment Actions

**Block**: Subnet-level firewall. See Section 6.1.
**Quarantine**: Full isolation. See Section 6.2.
**Unblock**: Removes containment. Infected host becomes active again.

Only agent actions change containment.

### 11.4 Server Protection Rule

Agents CAN move to host 17 and observe its traffic. Agents CANNOT contain host 17. Block, quarantine, and unblock on host 17 are no-ops: no state change, no reward, no penalty.


## 12. Observation Model

Each agent receives a 77-dimensional vector. All values clipped to [0, 1] after normalization.

### 12.1 Traffic Features (indices 0-15)

Computed from all traffic records (legit + malicious mixed) at agent's current host over last 12 timesteps.

**Outgoing (0-7):**
- 0: Successful outgoing flows (/ 30)
- 1: Total outgoing attempts (/ 30)
- 2: Failed outgoing rate (failed / total)
- 3: Unique destination IPs (/ 18)
- 4: Unique destination ports (/ 12)
- 5: Total bytes sent (/ 60000)
- 6: Total bytes received from outgoing (/ 60000)
- 7: Cross-subnet rate (cross / total)

**Incoming (8-11):**
- 8: Total incoming attempts (/ 30)
- 9: Unique source IPs (/ 18)
- 10: Unique source ports (/ 12)
- 11: Total bytes received incoming (/ 60000)

**Decayed long-memory (12-15):**
- 12: Failed outgoing connections (/ 20)
- 13: Unique peers (/ 20)
- 14: Incoming scan-like probes (/ 20)
- 15: Connections to file server (/ 20)

In transit → all 16 features zero.

### 12.2 Global Features (indices 16-76)

- 16-33: This agent's position one-hot (18).
- 34-51: Other agent 1's position one-hot (18).
- 52-69: Other agent 2's position one-hot (18).
- 70: Fraction quarantined (/ 18).
- 71: Fraction blocked (/ 18).
- 72: Normalized server damage (/ 300).
- 73: Episode progress (/ 200).
- 74: Current host blocked flag.
- 75: Current host quarantined flag.
- 76: In-transit flag.

### 12.3 Infected Host Detection Signals

- More total outgoing connections.
- More unique destination IPs and ports.
- Higher failed connection rate.
- Higher cross-subnet rate.
- Lower average bytes per connection.
- Hosts being scanned show more incoming from more unique sources.
- File server shows high incoming during attack phase.


## 13. Action Space

22 discrete actions:

- 0: **Observe**.
- 1-18: **Move to host 0-17**.
- 19: **Block** current host (no-op on server).
- 20: **Quarantine** current host (no-op on server).
- 21: **Unblock** current host (no-op on server).

In transit → action ignored.


## 14. Episode Structure

- `MAX_TIMESTEPS = 200`
- `INITIAL_INFECTIONS = 1`

### 14.1 Defender Win
Every infected host has `containment_state = QUARANTINED`. Block does NOT count — blocked infected hosts can still spread locally. At least one infection must have occurred.

### 14.2 Attacker Win
Server damage reaches 300.0.

### 14.3 Truncation
200 timesteps without either condition.


## 15. Reward Function

### 15.1 Dense Shared Reward (every step, all agents)

```text
+ 0.03 × number of clean hosts
- 1.50 × number of infected+uncontained hosts
- 0.50 × number of infected+blocked hosts
- 0.00 × number of infected+quarantined hosts
- 0.30 × number of clean+blocked hosts
- 0.50 × number of clean+quarantined hosts
- 0.10 × number of agent overlap pairs (both not in transit, same host)
- 6.00 × number of new infections this step
- 0.25 × server damage points added this step
```

### 15.2 Event Rewards (acting agent only)

| Event | Reward |
|---|---|
| Correct block (infected+uncontained → blocked) | +4.0 |
| Correct quarantine (infected+uncontained → quarantined) | +10.0 |
| Quarantine upgrade (infected+blocked → quarantined) | +4.0 |
| False block (clean+uncontained → blocked) | -3.0 |
| False quarantine (clean+uncontained → quarantined) | -6.0 |
| False quarantine upgrade (clean+blocked → quarantined) | -3.0 |
| Correct unblock (clean+contained → uncontained) | +1.5 |
| Bad unblock (infected+contained → uncontained) | -4.0 |
| Move within subnet | -0.1 |
| Move across subnet | -0.25 |

### 15.3 No-Ops (zero reward, no state change)

- Observe action
- Move to host agent is already on
- Block already-blocked host
- Block already-quarantined host (no downgrade)
- Quarantine already-quarantined host
- Unblock already-uncontained host
- Any containment action on server (host 17)

### 15.4 Terminal Rewards (all agents)

| Outcome | Reward |
|---|---|
| Server compromised | -40.0 |
| All infected hosts quarantined | +25.0 |
| Survived to time limit | 0.0 |


## 16. C2 Server Model

C2 is external. Represented by `dest_id = -1`.

- No host object exists.
- Beacons: source = infected host, dest = -1. Stored in source outgoing only.
- C2 never initiates connections. No records with `source_id = -1`.
- External to all subnets, so beacons fail from blocked/quarantined hosts.


## 17. PPO / IPPO Interface

- `reset()` → 3 observations, 3 info dicts.
- `step(actions)` takes 3 ints, returns: 3 observations, 3 rewards, 3 terminated, 3 truncated, 3 info dicts.


## 18. Timestep Order of Operations

1. Process agent actions (movement, containment).
2. Decay long-term suspicion scores.
3. Run attacker (beaconing, scanning, infection, server attacks).
4. Generate legitimate traffic for all hosts.
5. Prune old traffic records.
6. Check termination conditions.
7. Compute rewards.
8. Build observations.


## 19. Configuration Constants

```text
Network:
  NUM_HOSTS = 18, NUM_SUBNETS = 6, NUM_AGENTS = 3, SERVER_HOST_ID = 17

Episode:
  MAX_TIMESTEPS = 200, INITIAL_INFECTIONS = 1, OBSERVATION_WINDOW = 12

Vulnerability:
  VULN_RANGE_REGULAR = (0.03, 0.10)

Attacker:
  BEACON_INTERVAL_BASE = 3, BEACON_JITTER = 1
  SCAN_DELAY = 6, SCAN_PROBES_PER_STEP = (3, 5), SCAN_SAME_SUBNET_PROB = 0.80
  ATTACK_INFECTION_THRESHOLD = 5, ATTACK_CONNECTIONS_PER_STEP = (1, 3)
  SERVER_DAMAGE_PER_CONNECTION = 1.0, SERVER_DAMAGE_THRESHOLD = 300.0

Normal Traffic:
  NORMAL_FLOWS_PER_STEP = (1, 3)
  NORMAL_SAME_SUBNET_PROB = 0.55, NORMAL_SERVER_PROB = 0.25, NORMAL_CROSS_SUBNET_PROB = 0.20
  NORMAL_PORTS = [80, 443, 445, 8080]

Bytes:
  NORMAL_BYTES_RANGE = (200, 4000), SCAN_PROBE_BYTES = (40, 80)
  INFECTION_PAYLOAD_BYTES = (2000, 8000), BEACON_BYTES = (100, 300)
  ATTACK_BYTES = (4000, 20000)

Movement:
  MOVE_WITHIN_SUBNET_COST = 1, MOVE_CROSS_SUBNET_COST = 2

Observation:
  NUM_TRAFFIC_FEATURES = 16, NUM_GLOBAL_FEATURES = 61, OBSERVATION_SIZE = 77
  SUSPICIOUS_DECAY_FACTOR = 0.97

Actions:
  NUM_ACTIONS = 22

Rewards:
  REWARD_HEALTHY_HOST = 0.03
  REWARD_INFECTED_UNCONTAINED = -1.50
  REWARD_INFECTED_BLOCKED = -0.50
  REWARD_INFECTED_QUARANTINED = 0.00
  REWARD_FALSE_BLOCK_PER_STEP = -0.30
  REWARD_FALSE_QUARANTINE_PER_STEP = -0.50
  REWARD_AGENT_OVERLAP = -0.10
  REWARD_NEW_INFECTION = -6.00
  REWARD_SERVER_DAMAGE = -0.25
  REWARD_CORRECT_BLOCK = 4.00
  REWARD_CORRECT_QUARANTINE = 10.00
  REWARD_QUARANTINE_UPGRADE = 4.00
  REWARD_FALSE_BLOCK = -3.00
  REWARD_FALSE_QUARANTINE = -6.00
  REWARD_FALSE_QUARANTINE_UPGRADE = -3.00
  REWARD_CORRECT_UNBLOCK = 1.50
  REWARD_BAD_UNBLOCK = -4.00
  REWARD_MOVE_WITHIN_SUBNET = -0.10
  REWARD_MOVE_ACROSS_SUBNET = -0.25
  REWARD_SERVER_COMPROMISED = -40.00
  REWARD_ALL_QUARANTINED = 25.00
  REWARD_SURVIVED = 0.00

PPO:
  LR_ACTOR = 3e-4, LR_CRITIC = 1e-3
  GAMMA = 0.99, GAE_LAMBDA = 0.95, CLIP_EPSILON = 0.2
  ENTROPY_COEFF = 0.01, EPOCHS = 5, BATCH_SIZE = 128, HORIZON = 1024
  HIDDEN_1 = 128, HIDDEN_2 = 64
```