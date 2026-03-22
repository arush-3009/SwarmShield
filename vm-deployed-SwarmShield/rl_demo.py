#!/usr/bin/env python3
"""
SwarmShield RL-Driven Mininet Demo
====================================

This is the real deal:
1. Builds the Mininet network (18 hosts, 6 subnets)
2. Loads trained IPPO agents from checkpoints
3. Runs the SwarmShield environment
4. Agents make REAL decisions using their trained policy
5. Every block/quarantine/unblock gets applied as iptables rules
   on the actual Mininet hosts in real time

The simulation drives the logic. Mininet mirrors the physical effects.
"""

import sys
import os
import time

# Add the RL project to path
sys.path.insert(0, os.path.expanduser("~/swarmshield-rl"))

import torch
import numpy as np

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel
from mininet.node import OVSBridge

from env.swarmshield_env import SwarmShieldEnv
from agents.ippo import IPPO
from env.config import (
    NUM_AGENTS, NUM_HOSTS, NUM_ACTIONS,
    ACTION_OBSERVE, ACTION_MOVE_BASE, ACTION_MOVE_LAST,
    ACTION_BLOCK, ACTION_QUARANTINE, ACTION_UNBLOCK,
    HOST_NAMES, SUBNET_HOSTS, SERVER_DAMAGE_THRESHOLD,
)

# =========================================================================
# MININET TOPOLOGY
# =========================================================================

SUBNET_NAMES = [
    "Sales", "Accounting", "BackDesks",
    "Management", "Conference", "ServerCloset",
]


class SwarmShieldTopo(Topo):
    def build(self):
        core = self.addSwitch("s6")
        for subnet_id, host_ids in SUBNET_HOSTS.items():
            sw = self.addSwitch(f"s{subnet_id}")
            self.addLink(sw, core)
            for host_id in host_ids:
                ip = f"10.0.{subnet_id}.{host_id + 1}/16"
                h = self.addHost(f"h{host_id}", ip=ip)
                self.addLink(h, sw)


# =========================================================================
# HELPERS
# =========================================================================

def action_name(action):
    if action == ACTION_OBSERVE:
        return "OBSERVE"
    if ACTION_MOVE_BASE <= action <= ACTION_MOVE_LAST:
        target = action - ACTION_MOVE_BASE
        return f"MOVE->h{target}({HOST_NAMES[target]})"
    if action == ACTION_BLOCK:
        return "BLOCK"
    if action == ACTION_QUARANTINE:
        return "QUARANTINE"
    if action == ACTION_UNBLOCK:
        return "UNBLOCK"
    return f"ACTION_{action}"


def apply_quarantine(mn_host):
    """Full isolation: drop all traffic."""
    mn_host.cmd("iptables -F")
    mn_host.cmd("iptables -A INPUT -j DROP")
    mn_host.cmd("iptables -A OUTPUT -j DROP")


def apply_block(mn_host, subnet_id):
    """Subnet-level block: allow same-subnet, drop cross-subnet."""
    mn_host.cmd("iptables -F")
    mn_host.cmd(f"iptables -A INPUT -s 10.0.{subnet_id}.0/24 -j ACCEPT")
    mn_host.cmd("iptables -A INPUT -j DROP")
    mn_host.cmd(f"iptables -A OUTPUT -d 10.0.{subnet_id}.0/24 -j ACCEPT")
    mn_host.cmd("iptables -A OUTPUT -j DROP")


def apply_unblock(mn_host):
    """Remove all iptables rules."""
    mn_host.cmd("iptables -F")


def clear_all_rules(mn_hosts):
    for h in mn_hosts.values():
        h.cmd("iptables -F 2>/dev/null")


# =========================================================================
# DISPLAY
# =========================================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_state(timestep, infos, actions, env):
    info = infos[0]
    os.system("clear")

    print(f"{BOLD}{'='*70}")
    print(f"  SWARMSHIELD RL DEMO — TIMESTEP {timestep}")
    print(f"{'='*70}{RESET}\n")

    # Infection status
    infected_ids = env.network.get_all_infected_host_ids()
    blocked_ids = []
    quarantined_ids = []
    for hid in range(NUM_HOSTS):
        host_obj = env.network.get_host(hid)
        if host_obj.is_blocked:
            blocked_ids.append(hid)
        if host_obj.is_quarantined:
            quarantined_ids.append(hid)

    # Network map
    print(f"  {BOLD}NETWORK STATUS:{RESET}")
    for subnet_id, host_ids in SUBNET_HOSTS.items():
        subnet_str = f"  {SUBNET_NAMES[subnet_id]:12s} |"
        for hid in host_ids:
            name = HOST_NAMES[hid][:8]
            if hid in quarantined_ids and hid in infected_ids:
                tag = f"{BLUE}[Q]{name}{RESET}"
            elif hid in blocked_ids and hid in infected_ids:
                tag = f"{YELLOW}[B]{name}{RESET}"
            elif hid in infected_ids:
                tag = f"{RED}[!]{name}{RESET}"
            elif hid in quarantined_ids:
                tag = f"{CYAN}[Q]{name}{RESET}"
            elif hid in blocked_ids:
                tag = f"{CYAN}[B]{name}{RESET}"
            else:
                tag = f"{GREEN}[ ]{name}{RESET}"
            subnet_str += f" {tag}"
        print(subnet_str)

    print()
    print(f"  {BOLD}LEGEND:{RESET} {GREEN}[ ]=Clean{RESET}  "
          f"{RED}[!]=Infected{RESET}  "
          f"{YELLOW}[B]=Inf+Blocked{RESET}  "
          f"{BLUE}[Q]=Inf+Quarantined{RESET}  "
          f"{CYAN}[B]/[Q]=FalsePositive{RESET}")

    # Stats
    print(f"\n  {BOLD}STATS:{RESET}")
    print(f"    Infected total:       {info['infected_total']}")
    print(f"    Infected uncontained: {RED}{info['infected_uncontained']}{RESET}")
    print(f"    Infected blocked:     {YELLOW}{info['infected_blocked']}{RESET}")
    print(f"    Infected quarantined: {BLUE}{info['infected_quarantined']}{RESET}")
    print(f"    Server damage:        {info['server_damage']:.0f} / {SERVER_DAMAGE_THRESHOLD:.0f}")
    print(f"    New infections:       {info['num_new_infections_this_step']}")

    # Agent actions
    print(f"\n  {BOLD}AGENT ACTIONS:{RESET}")
    for i in range(NUM_AGENTS):
        agent_info = infos[i]
        host = agent_info['current_host']
        transit = agent_info['in_transit']
        act = action_name(actions[i])
        loc = f"h{host}({HOST_NAMES[host]})"
        if transit:
            loc += " [IN TRANSIT]"
        print(f"    Agent {i}: at {MAGENTA}{loc}{RESET} -> {act}")

    # Terminal check
    if info.get('all_infections_quarantined', False):
        print(f"\n  {BOLD}{GREEN}>>> ALL INFECTIONS QUARANTINED — DEFENDERS WIN! <<<{RESET}")
    elif info.get('server_compromised', False):
        print(f"\n  {BOLD}{RED}>>> SERVER COMPROMISED — ATTACKERS WIN! <<<{RESET}")

    print(f"\n{'='*70}")


# =========================================================================
# MAIN DEMO LOOP
# =========================================================================

def run_demo(seed=None, speed=0.5):
    """
    speed: seconds between timesteps. 0.5 = fast, 2.0 = slow for narration.
    """
    setLogLevel("warning")

    # --- Build Mininet ---
    print("Building Mininet network...")
    topo = SwarmShieldTopo()
    net = Mininet(topo=topo, switch=OVSBridge, controller=None)
    net.start()

    mn_hosts = {}
    for i in range(NUM_HOSTS):
        mn_hosts[i] = net.get(f"h{i}")

    clear_all_rules(mn_hosts)

    # --- Load RL agents ---
    print("Loading trained RL agents...")
    device = torch.device("cpu")
    ippo = IPPO(device)

    checkpoint_dir = os.path.expanduser("~/swarmshield-rl/checkpoints/best")
    if not os.path.exists(os.path.join(checkpoint_dir, "agent_0.pt")):
        checkpoint_dir = os.path.expanduser("~/swarmshield-rl/checkpoints/latest")
    if not os.path.exists(os.path.join(checkpoint_dir, "agent_0.pt")):
        checkpoint_dir = os.path.expanduser("~/swarmshield-rl/checkpoints")

    ippo.load_all(checkpoint_dir)
    print(f"Loaded from {checkpoint_dir}")

    # --- Create env ---
    env = SwarmShieldEnv(seed=seed)
    observations, infos = env.reset()

    # Track containment state to detect changes
    prev_blocked = set()
    prev_quarantined = set()

    timestep = 0
    done = False

    print("Starting RL demo...\n")
    time.sleep(1)

    while not done:
        # Agent decisions (deterministic = greedy, best action)
        actions, _, _ = ippo.select_actions(observations)

        # Step environment
        observations, rewards, dones, truncateds, infos = env.step(actions)

        timestep += 1
        done = dones[0] or truncateds[0]

        # --- Apply containment to Mininet ---
        current_blocked = set()
        current_quarantined = set()

        for hid in range(NUM_HOSTS):
            host_obj = env.network.get_host(hid)
            if host_obj.is_quarantined:
                current_quarantined.add(hid)
            elif host_obj.is_blocked:
                current_blocked.add(hid)

        # Apply new quarantines
        for hid in current_quarantined - prev_quarantined:
            apply_quarantine(mn_hosts[hid])

        # Apply new blocks
        for hid in current_blocked - prev_blocked:
            subnet_id = None
            for sid, hids in SUBNET_HOSTS.items():
                if hid in hids:
                    subnet_id = sid
                    break
            if subnet_id is not None:
                apply_block(mn_hosts[hid], subnet_id)

        # Apply unblocks (was contained, now isn't)
        for hid in (prev_blocked | prev_quarantined) - (current_blocked | current_quarantined):
            apply_unblock(mn_hosts[hid])

        prev_blocked = current_blocked
        prev_quarantined = current_quarantined

        # --- Display ---
        print_state(timestep, infos, actions, env)
        time.sleep(speed)

    # --- End ---
    time.sleep(2)
    clear_all_rules(mn_hosts)
    net.stop()

    info = infos[0]
    if info.get('all_infections_quarantined', False):
        return "win"
    elif info.get('server_compromised', False):
        return "loss"
    return "survive"


if __name__ == "__main__":
    # Try different seeds to find a winning episode
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    result = run_demo(seed=seed, speed=speed)
    print(f"\nEpisode result: {result}")
