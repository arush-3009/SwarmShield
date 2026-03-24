"""
SwarmShield Evaluation Engine
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from env.swarmshield_env import SwarmShieldEnv
from env.config import (
    HOST_NAMES, SUBNET_NAMES, HOST_TO_SUBNET,
    NUM_HOSTS, NUM_AGENTS, MAX_TIMESTEPS,
    SERVER_HOST_ID, SERVER_DAMAGE_THRESHOLD,
    ACTION_OBSERVE, ACTION_MOVE_BASE, ACTION_MOVE_LAST,
    ACTION_BLOCK, ACTION_QUARANTINE, ACTION_UNBLOCK,
    CONTAINMENT_NONE, CONTAINMENT_BLOCKED, CONTAINMENT_QUARANTINED,
)
from agents.ippo import IPPO

ACTION_NAMES = {
    ACTION_OBSERVE: "OBSERVE",
    ACTION_BLOCK: "BLOCK",
    ACTION_QUARANTINE: "QUARANTINE",
    ACTION_UNBLOCK: "UNBLOCK",
}
AGENT_NAMES = ["Dwight", "Jim", "Michael"]

def action_to_string(action):
    if action in ACTION_NAMES:
        return ACTION_NAMES[action]
    if ACTION_MOVE_BASE <= action <= ACTION_MOVE_LAST:
        target = action - ACTION_MOVE_BASE
        return f"MOVE->{HOST_NAMES[target]}"
    return f"ACTION_{action}"

class EvalEngine:
    def __init__(self, checkpoint_dir=None, device='cpu'):
        self.device = device
        self.env = SwarmShieldEnv()
        self.ippo = IPPO(device=device)
        if checkpoint_dir is None:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            for subdir in ['checkpoints/best', 'checkpoints/latest', 'checkpoints']:
                path = os.path.join(base, subdir)
                if os.path.exists(os.path.join(path, 'agent_0.pt')):
                    checkpoint_dir = path
                    break
            if checkpoint_dir is None:
                raise FileNotFoundError("No checkpoint directory found")
        print(f"[engine] Loading checkpoints from: {checkpoint_dir}")
        self.ippo.load_all(checkpoint_dir)
        self.deterministic = True
        self.current_obs = None
        self.episode_num = 0
        self.wins = 0
        self.losses = 0
        self.survived = 0
        self.done = True
        self.action_log = []
        self.prev_infected = set()

    def reset(self, seed=None, extra_infections=0):
        obs, infos = self.env.reset(seed=seed)
        if extra_infections > 0:
            from env.config import REGULAR_HOST_IDS
            available = [h.host_id for h in self.env.network.hosts
                         if not h.infected and not h.is_server]
            n = min(extra_infections, len(available))
            chosen = self.env.rng.choice(available, size=n, replace=False)
            for hid in chosen:
                self.env.network.hosts[int(hid)].infect(0)
            obs = self.env._build_all_observations()
        self.current_obs = obs
        self.done = False
        self.episode_num += 1
        self.action_log = []
        self.prev_infected = set()
        for h in self.env.network.hosts:
            if h.infected:
                self.prev_infected.add(h.host_id)
        return self._build_state(actions=None, rewards=None, infos=infos)

    def step(self):
        if self.done:
            return None
        if self.deterministic:
            actions = self.ippo.select_actions_deterministic(self.current_obs)
        else:
            actions, _, _ = self.ippo.select_actions(self.current_obs)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self.current_obs = obs
        term = terminated[0]
        trunc = truncated[0]
        if term or trunc:
            self.done = True
            if self.env.network.all_infections_quarantined():
                self.wins += 1
            elif self.env.network.is_server_compromised(
                self.env.attacker.server_damage, SERVER_DAMAGE_THRESHOLD
            ):
                self.losses += 1
            else:
                self.survived += 1
        state = self._build_state(actions=actions, rewards=rewards, infos=infos)
        for h in self.env.network.hosts:
            if h.infected:
                self.prev_infected.add(h.host_id)
        return state

    def _build_state(self, actions, rewards, infos):
        hosts = []
        for h in self.env.network.hosts:
            hosts.append({
                'id': h.host_id, 'name': h.name, 'subnet': h.subnet_id,
                'subnet_name': SUBNET_NAMES[h.subnet_id],
                'infected': h.infected, 'containment': h.containment_state,
                'is_server': h.is_server,
            })
        agents = []
        for i, ag in enumerate(self.env.agent_states):
            action_str = None
            action_raw = None
            if actions is not None:
                action_raw = int(actions[i])
                action_str = action_to_string(action_raw)
            agents.append({
                'id': i, 'name': AGENT_NAMES[i],
                'host': ag.current_host, 'host_name': HOST_NAMES[ag.current_host],
                'in_transit': ag.in_transit,
                'target': ag.transit_target,
                'target_name': HOST_NAMES[ag.transit_target] if ag.transit_target is not None else None,
                'transit_remaining': ag.transit_remaining,
                'action': action_str, 'action_raw': action_raw,
            })
        traffic = []
        ts = self.env.current_timestep - 1 if self.env.current_timestep > 0 else 0
        for host_id in range(NUM_HOSTS):
            for rec in self.env.traffic_manager.outgoing_history[host_id]:
                if rec.timestamp == ts and rec.is_malicious:
                    if rec.dest_id == -1:
                        ttype = 'c2_beacon'
                    elif rec.dest_id == SERVER_HOST_ID:
                        ttype = 'server_attack'
                    elif rec.bytes_sent < 200:
                        ttype = 'scan'
                    else:
                        ttype = 'lateral'
                    traffic.append({
                        'src': rec.source_id, 'dst': rec.dest_id,
                        'type': ttype, 'success': rec.success, 'bytes': rec.bytes_sent,
                    })
        events = []
        newly_infected = infos[0].get('newly_infected_this_step', []) if infos else []
        for hid in newly_infected:
            events.append({'type': 'infection', 'host': hid, 'host_name': HOST_NAMES[hid]})
        if actions is not None:
            for i in range(NUM_AGENTS):
                ag = self.env.agent_states[i]
                a = int(actions[i])
                if a == ACTION_BLOCK:
                    events.append({'type': 'block', 'agent': i, 'host': ag.current_host})
                elif a == ACTION_QUARANTINE:
                    events.append({'type': 'quarantine', 'agent': i, 'host': ag.current_host})
                elif a == ACTION_UNBLOCK:
                    events.append({'type': 'unblock', 'agent': i, 'host': ag.current_host})
        if actions is not None:
            for i in range(NUM_AGENTS):
                ag = self.env.agent_states[i]
                self.action_log.append({
                    'timestep': self.env.current_timestep, 'agent': i,
                    'agent_name': AGENT_NAMES[i],
                    'action': action_to_string(int(actions[i])),
                    'host': ag.current_host, 'host_name': HOST_NAMES[ag.current_host],
                })
        counts = self.env.network.count_by_status()
        outcome = None
        if self.done:
            if self.env.network.all_infections_quarantined():
                outcome = 'win'
            elif self.env.network.is_server_compromised(
                self.env.attacker.server_damage, SERVER_DAMAGE_THRESHOLD
            ):
                outcome = 'loss'
            else:
                outcome = 'survived'
        return {
            'timestep': self.env.current_timestep,
            'max_timesteps': MAX_TIMESTEPS,
            'episode': self.episode_num,
            'hosts': hosts, 'agents': agents,
            'server_damage': float(self.env.attacker.server_damage),
            'server_damage_max': float(SERVER_DAMAGE_THRESHOLD),
            'traffic': traffic, 'events': events,
            'counts': {
                'clean': counts['clean_total'],
                'infected_uncontained': counts['infected_uncontained'],
                'infected_blocked': counts['infected_blocked'],
                'infected_quarantined': counts['infected_quarantined'],
                'infected_total': counts['infected_total'],
                'blocked_total': counts['blocked_total'],
                'quarantined_total': counts['quarantined_total'],
            },
            'terminated': self.done, 'outcome': outcome,
            'wins': self.wins, 'losses': self.losses, 'survived': self.survived,
            'rewards': [float(r) for r in rewards] if rewards else [0.0, 0.0, 0.0],
            'action_log': self.action_log[-30:],
        }
