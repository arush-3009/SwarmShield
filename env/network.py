"""
SwarmShield Network Topology
=============================

This file defines the structure of the Dunder Mifflin network and tracks
the state of every host during a simulation episode.

WHAT THIS MODELS:
- 18 computers across 6 departments (subnets)
- Each computer has an infection status, vulnerability score, and firewall state
- The network has a central router connecting all subnets
- Same-subnet traffic flows directly; cross-subnet traffic goes through the router

WHAT A HOST IS:
In Mininet, each host is a virtual machine with a real IP address.
In our fast simulator, being designed for faster training, each host is just a dictionary of state variables.
We don't simulate real packets, instead we simulate the stats of network traffic,
because that's what the RL agents observe.
This makes training much faster than running on real Mininet.

WHAT A SUBNET IS:
A group of hosts connected to the same switch. Hosts within a subnet can
communicate directly. Hosts in different subnets must go through the router.
In network terms: same subnet = same local network = direct Layer 2 access.
Different subnets = different networks = must route through Layer 3.
"""

import numpy as np
from env.config import (
    NUM_HOSTS,
    NUM_SUBNETS,
    SUBNET_HOSTS,
    SUBNET_NAMES,
    HOST_TO_SUBNET,
    HOST_NAMES,
    SERVER_HOST_ID,
    VULN_RANGE_REGULAR,
    VULN_RANGE_SERVER,
)


# =============================================================================
# HOST STATUS VALUES
# =============================================================================
# Every host is in exactly one of these states at any time.
# The RL agents don't see these directly and they must infer infection
# from traffic patterns, but the environment tracks them internally.

#Macros for host state:

STATUS_CLEAN = 0          # Normal, not infected, fully operational
STATUS_INFECTED = 1       # Infected with malware, but still on the network
STATUS_BLOCKED = 2        # Agent blocked this host's cross-subnet traffic
STATUS_QUARANTINED = 3    # Agent fully isolated this host from the network


class Host:
    """
    Represents one computer on the network.

    Each host has:
    - An ID (0-17) and a human-readable name (e.g. Jim, Dwight, etc.)
    - A subnet it belongs to.
    - A vulnerability score.
    - An infection status (clean, infected, blocked, quarantined)
    - Tracking variables for the attacker's behavior (beacon timer, etc.)
    """

    def __init__(self, host_id):
        """
        Create a host with default values.
        The host starts clean. Vulnerability score -> randomized when the episode resets.
        """
        self.host_id = host_id
        self.name = HOST_NAMES[host_id]
        self.subnet_id = HOST_TO_SUBNET[host_id]
        self.is_server = (host_id == SERVER_HOST_ID)

        # These get set properly in reset()
        self.vulnerability = 0.0
        self.status = STATUS_CLEAN
        self.timestep_infected = -1
        self.beacon_timer = 0
        self.has_been_scanning = False

    def reset(self, rng):
        """
        Reset this host to a clean state for a new episode.

        rng: numpy random generator
        """
        # Randomize vulnerability score.
        # Regular hosts: between 0.2 and 0.7 (some easy, some hard to infect).
        # Server: between 0.05 and 0.15 (always hard, but exact value varies).
        # The agent never sees this number, it must detect infections from traffic behavior.
        if self.is_server:
            low, high = VULN_RANGE_SERVER
        else:
            low, high = VULN_RANGE_REGULAR
            
        self.vulnerability = rng.uniform(low, high)

        # Start clean and operational
        self.status = STATUS_CLEAN

        # -1 -> "never been infected"
        self.timestep_infected = -1

        # Beacon timer: countdown to next beacon.
        # Only used when host is infected. Set to 0 so first beacon
        # happens immediately after infection.
        self.beacon_timer = 0

        # Whether this host has started scanning yet.
        # Scanning starts a few timesteps after infection.
        self.has_been_scanning = False

    def infect(self, current_timestep):
        """
        Mark this host as infected.
        Called when the attacker successfully compromises this machine.
        """
        self.status = STATUS_INFECTED
        self.timestep_infected = current_timestep
        self.beacon_timer = 0  # Beacon immediately
        self.has_been_scanning = False

    def block(self):
        """
        Block this host's cross-subnet traffic.
        The host can still communicate within its subnet but cannot
        reach other departments or the server through the router.
        Basically, configuring the firewall rule -> iptables -A FORWARD -s <host_ip> -j DROP
        """
        self.status = STATUS_BLOCKED

    def quarantine(self):
        """
        Fully isolate this host from the network.
        No traffic in or out. Complete shutdown of network access.
        This is the nuclear option —> most effective if the host is
        infected, most damaging if it's clean.
        """
        self.status = STATUS_QUARANTINED

    def unblock(self):
        """
        Remove all firewall rules for this host.
        Restores full network access.
        If the host was infected, it goes back to INFECTED status
        (still compromised, just no longer blocked).
        If the host was clean and falsely blocked, it goes back to CLEAN.
        """
        # If the host was infected before being blocked/quarantined,
        # unblocking restores it to infected (the malware is still there).
        # If it was clean, it goes back to clean.
        if self.timestep_infected >= 0:
            self.status = STATUS_INFECTED
        else:
            self.status = STATUS_CLEAN

    @property
    def is_clean(self):
        return self.status == STATUS_CLEAN

    @property
    def is_infected(self):
        return self.status == STATUS_INFECTED

    @property
    def is_blocked(self):
        return self.status == STATUS_BLOCKED

    @property
    def is_quarantined(self):
        return self.status == STATUS_QUARANTINED

    @property
    def is_operational(self):
        """Can this host send and receive traffic?"""
        # Quarantined hosts are fully offline.
        # Blocked hosts can still communicate within their subnet.
        # Clean and infected hosts are fully operational.
        return self.status != STATUS_QUARANTINED

    @property
    def can_send_cross_subnet(self):
        """Can this host send traffic to other subnets?"""
        # Only clean and infected hosts can send cross-subnet.
        # Blocked hosts have their cross-subnet traffic dropped by the router.
        # Quarantined hosts can't send anything.
        return self.status in (STATUS_CLEAN, STATUS_INFECTED)


class Network:
    """
    The complete Dunder Mifflin network.

    This class manages all 18 hosts, tracks which subnet each belongs to,
    and provides helper methods for querying network state.

    The network topology is fixed (same departments, same connections every
    episode). What changes between episodes is:
    - Vulnerability scores (randomized)
    - Which hosts start infected (randomized)
    - Agent starting positions (randomized)
    """

    def __init__(self):
        """Create all hosts."""
        self.hosts = []
        for host_id in range(NUM_HOSTS):
            host = Host(host_id)
            self.hosts.append(host)

    def reset(self, rng):
        """
        Reset all hosts for a new episode.

        Args:
            rng: numpy random generator for reproducible randomness.
        """
        for host in self.hosts:
            host.reset(rng)

    def get_host(self, host_id):
        """Get a host by its ID (0-17)."""
        return self.hosts[host_id]

    def get_subnet_hosts(self, subnet_id):
        """
        Get all hosts in a given subnet.

        Returns a list of Host objects.
        Example: get_subnet_hosts(0) returns [Jim, Dwight, Stanley, Phyllis, Andy]
        """
        host_ids = SUBNET_HOSTS[subnet_id]
        result = []
        for host_id in host_ids:
            result.append(self.hosts[host_id])
        return result

    def get_hosts_in_same_subnet(self, host_id):
        """
        Get all OTHER hosts in the same subnet as the given host.

        If host_id is Jim (0), returns [Dwight(1), Stanley(2), Phyllis(3), Andy(4)].
        Does NOT include the host itself.
        """
        subnet_id = HOST_TO_SUBNET[host_id]
        result = []
        for other_id in SUBNET_HOSTS[subnet_id]:
            if other_id != host_id:
                result.append(self.hosts[other_id])
        return result

    def get_hosts_in_different_subnets(self, host_id):
        """
        Get all hosts in DIFFERENT subnets from the given host.

        These are the hosts that can only be reached through the router.
        Cross-subnet traffic is where agents positioned at the router
        can observe and block suspicious activity.
        """
        my_subnet = HOST_TO_SUBNET[host_id]
        result = []
        for other_host in self.hosts:
            if HOST_TO_SUBNET[other_host.host_id] != my_subnet:
                result.append(other_host)
        return result

    def get_infected_hosts(self):
        """
        Get all currently infected (and not blocked/quarantined) hosts.

        These are the actively dangerous hosts — they're infected and
        still have network access, so they can beacon, scan, and attack.
        """
        result = []
        for host in self.hosts:
            if host.is_infected:
                result.append(host)
        return result

    def get_all_infected_including_contained(self):
        """
        Get all hosts that have been infected, including those that are
        now blocked or quarantined.

        Used for checking win conditions: if every infected host (including
        contained ones) is quarantined, the defenders have won.
        """
        result = []
        for host in self.hosts:
            if host.timestep_infected >= 0:
                result.append(host)
        return result

    def count_by_status(self):
        """
        Count hosts in each status. Used for reward computation and
        episode termination checks.

        Returns a dictionary:
        {
            'clean': int,          # Clean AND unblocked
            'infected': int,       # Infected AND unblocked (dangerous!)
            'blocked': int,        # Blocked (could be clean or infected)
            'quarantined': int,    # Quarantined (could be clean or infected)
            'clean_blocked': int,  # Clean hosts that are blocked (false positive)
            'clean_quarantined': int,  # Clean hosts quarantined (false positive)
        }
        """
        counts = {
            'clean': 0,
            'infected': 0,
            'blocked': 0,
            'quarantined': 0,
            'clean_blocked': 0,
            'clean_quarantined': 0,
        }

        for host in self.hosts:
            if host.status == STATUS_CLEAN:
                counts['clean'] += 1

            elif host.status == STATUS_INFECTED:
                counts['infected'] += 1

            elif host.status == STATUS_BLOCKED:
                counts['blocked'] += 1
                # Was this host actually infected before being blocked?
                if host.timestep_infected < 0:
                    # Never been infected — this is a false positive
                    counts['clean_blocked'] += 1

            elif host.status == STATUS_QUARANTINED:
                counts['quarantined'] += 1
                if host.timestep_infected < 0:
                    counts['clean_quarantined'] += 1

        return counts

    def is_server_compromised(self, server_damage, threshold):
        """
        Check if the file server has been compromised.

        The server doesn't fall instantly. Attack traffic accumulates
        "damage" over time. If total damage exceeds the threshold,
        the server is compromised and the attackers win.

        Args:
            server_damage: float, accumulated damage so far
            threshold: float, damage needed for compromise
        """
        return server_damage >= threshold

    def all_infections_contained(self):
        """
        Check if every infected host is either blocked or quarantined.

        This is the DEFENDER WIN condition. Note: it's not enough to
        quarantine some infected hosts — ALL of them must be contained.
        If even one infected host remains free, it can keep spreading.
        """
        for host in self.hosts:
            # If a host has ever been infected and is currently
            # in INFECTED status (not blocked/quarantined), containment fails
            if host.is_infected:
                return False

        # Also check: are there ANY infected hosts at all?
        # If nobody was ever infected and agents just quarantined random
        # clean hosts, that's not a "win"
        infected_ever = self.get_all_infected_including_contained()
        if len(infected_ever) == 0:
            return False

        return True

    def __str__(self):
        """Pretty print the network state (for debugging)."""
        lines = []
        for subnet_id in range(NUM_SUBNETS):
            subnet_name = SUBNET_NAMES[subnet_id]
            lines.append(f"\n--- {subnet_name} (Subnet {subnet_id}) ---")
            for host in self.get_subnet_hosts(subnet_id):
                status_str = {
                    STATUS_CLEAN: "CLEAN",
                    STATUS_INFECTED: "INFECTED",
                    STATUS_BLOCKED: "BLOCKED",
                    STATUS_QUARANTINED: "QUARANTINED",
                }[host.status]
                vuln_str = f"vuln={host.vulnerability:.2f}"
                lines.append(f"  Host {host.host_id:2d} ({host.name:12s}): "
                             f"{status_str:12s} {vuln_str}")
        return "\n".join(lines)