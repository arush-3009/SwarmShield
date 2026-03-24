"""SwarmShield Visual Demo Server — RL + Mininet + Browser, all in sync"""
import argparse, json, os, sys, threading, time, datetime, re
from queue import Queue, Empty
from flask import Flask, render_template, Response, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engine import EvalEngine

from env.config import (
    SUBNET_HOSTS, NUM_HOSTS, NUM_AGENTS, HOST_NAMES,
    SERVER_DAMAGE_THRESHOLD,
)

mn_net = None
mn_hosts = {}
mn_enabled = False
MN_CMD_LOG = os.path.expanduser("~/swarmshield-rl/mininet_commands.log")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

SUBNET_NAMES = ["Sales", "Accounting", "BackDesks", "Management", "Conference", "ServerCloset"]
AGENT_NAMES = ["Dwight", "Jim", "Michael"]

def mn_log(msg):
    print("  " + DIM + "[mininet]" + RESET + " " + msg)
    clean = re.sub(r'\033\[[0-9;]*m', '', msg)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with open(MN_CMD_LOG, "a") as f:
        f.write("[" + ts + "] " + clean + "\n")
        f.flush()

def setup_mininet():
    global mn_net, mn_hosts, mn_enabled
    from mininet.topo import Topo
    from mininet.net import Mininet
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge

    class SwarmShieldTopo(Topo):
        def build(self):
            core = self.addSwitch("s6")
            for subnet_id, host_ids in SUBNET_HOSTS.items():
                sw = self.addSwitch("s" + str(subnet_id))
                self.addLink(sw, core)
                for host_id in host_ids:
                    ip = "10.0." + str(subnet_id) + "." + str(host_id + 1) + "/16"
                    h = self.addHost("h" + str(host_id), ip=ip)
                    self.addLink(h, sw)

    setLogLevel("warning")
    topo = SwarmShieldTopo()
    mn_net = Mininet(topo=topo, switch=OVSBridge, controller=None)
    mn_net.start()
    for i in range(NUM_HOSTS):
        mn_hosts[i] = mn_net.get("h" + str(i))
        mn_hosts[i].cmd("iptables -F 2>/dev/null")
    mn_enabled = True
    open(MN_CMD_LOG, "w").close()
    mn_log("Network started: 18 hosts, 7 switches, 6 subnets")

def apply_quarantine(hid):
    if not mn_enabled: return
    mn_hosts[hid].cmd("iptables -F")
    mn_hosts[hid].cmd("iptables -A INPUT -j DROP")
    mn_hosts[hid].cmd("iptables -A OUTPUT -j DROP")
    mn_log(BLUE + "QUARANTINE" + RESET + " h" + str(hid) + "(" + HOST_NAMES[hid] + "): iptables -A INPUT -j DROP; iptables -A OUTPUT -j DROP")

def apply_block(hid):
    if not mn_enabled: return
    subnet_id = None
    for sid, hids in SUBNET_HOSTS.items():
        if hid in hids:
            subnet_id = sid
            break
    if subnet_id is None: return
    mn_hosts[hid].cmd("iptables -F")
    mn_hosts[hid].cmd("iptables -A INPUT -s 10.0." + str(subnet_id) + ".0/24 -j ACCEPT")
    mn_hosts[hid].cmd("iptables -A INPUT -j DROP")
    mn_hosts[hid].cmd("iptables -A OUTPUT -d 10.0." + str(subnet_id) + ".0/24 -j ACCEPT")
    mn_hosts[hid].cmd("iptables -A OUTPUT -j DROP")
    mn_log(YELLOW + "BLOCK" + RESET + " h" + str(hid) + "(" + HOST_NAMES[hid] + "): allow 10.0." + str(subnet_id) + ".0/24, drop rest")

def apply_unblock(hid):
    if not mn_enabled: return
    mn_hosts[hid].cmd("iptables -F")
    mn_log(GREEN + "UNBLOCK" + RESET + " h" + str(hid) + "(" + HOST_NAMES[hid] + "): iptables -F")

def clear_all_mn():
    if not mn_enabled: return
    for h in mn_hosts.values():
        h.cmd("iptables -F 2>/dev/null")
    mn_log("All iptables rules cleared")

prev_containment = {}

def sync_mininet_to_state(state_dict):
    global prev_containment
    if not mn_enabled: return
    if not state_dict or 'hosts' not in state_dict: return
    for h in state_dict['hosts']:
        hid = h['id']
        cont = h['containment']
        old_cont = prev_containment.get(hid, 0)
        if cont != old_cont:
            if cont == 2: apply_quarantine(hid)
            elif cont == 1: apply_block(hid)
            else:
                if old_cont > 0: apply_unblock(hid)
        prev_containment[hid] = cont

def stop_mininet():
    global mn_net, mn_enabled
    if mn_enabled and mn_net:
        clear_all_mn()
        mn_net.stop()
        mn_enabled = False

def print_terminal_state(state_dict):
    if not state_dict or 'hosts' not in state_dict: return
    ts = state_dict['timestep']
    print("\n" + BOLD + "=" * 70)
    print("  SWARMSHIELD — TIMESTEP " + str(ts))
    print("=" * 70 + RESET)
    for subnet_id, host_ids in SUBNET_HOSTS.items():
        row = "  " + SUBNET_NAMES[subnet_id].ljust(12) + " |"
        for hid in host_ids:
            h = state_dict['hosts'][hid]
            name = HOST_NAMES[hid][:8]
            inf = h['infected']
            cont = h['containment']
            if inf and cont == 2: row += " " + BLUE + "[Q]" + name + RESET
            elif inf and cont == 1: row += " " + YELLOW + "[B]" + name + RESET
            elif inf: row += " " + RED + "[!]" + name + RESET
            elif cont > 0: row += " " + CYAN + "[C]" + name + RESET
            else: row += " " + GREEN + "[ ]" + name + RESET
        print(row)
    c = state_dict['counts']
    dmg = state_dict['server_damage']
    mx = state_dict['server_damage_max']
    print("\n  Infected: " + RED + str(c['infected_total']) + RESET + " (uncontained:" + str(c['infected_uncontained']) + " blocked:" + str(c['infected_blocked']) + " quarantined:" + str(c['infected_quarantined']) + ")")
    print("  Server damage: " + str(int(dmg)) + "/" + str(int(mx)))
    for ag in state_dict['agents']:
        loc = "h" + str(ag['host']) + "(" + ag['host_name'] + ")"
        act = ag['action'] or 'INIT'
        transit = " [TRANSIT]" if ag['in_transit'] else ""
        print("  Agent " + ag['name'].ljust(8) + ": " + MAGENTA + loc + transit + RESET + " -> " + act)
    if state_dict.get('outcome') == 'win':
        print("\n  " + BOLD + GREEN + ">>> ALL INFECTIONS QUARANTINED — DEFENDERS WIN! <<<" + RESET)
    elif state_dict.get('outcome') == 'loss':
        print("\n  " + BOLD + RED + ">>> SERVER COMPROMISED — ATTACKERS WIN! <<<" + RESET)
    elif state_dict.get('outcome') == 'survived':
        print("\n  " + BOLD + YELLOW + ">>> TIME EXPIRED <<<" + RESET)

app = Flask(__name__)
engine = None
state_queues = []
state_queues_lock = threading.Lock()
control = {
    'paused': True, 'speed': 0.5, 'seed': None,
    'deterministic': True, 'mode': 'live',
    'recording': None, 'replay_index': 0,
}
latest_state = None

def broadcast_state(state):
    global latest_state
    latest_state = state
    sync_mininet_to_state(state)
    print_terminal_state(state)
    with state_queues_lock:
        dead = []
        for i, q in enumerate(state_queues):
            try:
                while q.qsize() > 20:
                    try: q.get_nowait()
                    except Empty: break
                q.put_nowait(state)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            state_queues.pop(i)

@app.route('/')
def index(): return render_template('floor.html')
@app.route('/floor')
def floor(): return render_template('floor.html')
@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/stream')
def stream():
    q = Queue(maxsize=100)
    with state_queues_lock:
        state_queues.append(q)
    if latest_state: q.put(latest_state)
    def event_stream():
        try:
            while True:
                try:
                    state = q.get(timeout=2.0)
                    yield "data: " + json.dumps(state) + "\n\n"
                except Empty:
                    yield "data: " + json.dumps({'heartbeat': True}) + "\n\n"
        except GeneratorExit:
            with state_queues_lock:
                if q in state_queues: state_queues.remove(q)
    return Response(event_stream(), mimetype='text/event-stream',
        headers={'Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'})

@app.route('/control', methods=['POST'])
def handle_control():
    global prev_containment
    data = request.json
    action = data.get('action')
    if action == 'pause': control['paused'] = True
    elif action == 'resume': control['paused'] = False
    elif action == 'toggle_pause': control['paused'] = not control['paused']
    elif action == 'restart':
        control['paused'] = True
        seed = data.get('seed', control['seed'])
        control['seed'] = seed
        clear_all_mn()
        prev_containment = {}
        ei = control.get('extra_infections', 0)
        state = engine.reset(seed=int(seed) if seed is not None else None, extra_infections=ei)
        broadcast_state(state)
    elif action == 'speed_up': control['speed'] = max(0.05, control['speed'] * 0.7)
    elif action == 'speed_down': control['speed'] = min(3.0, control['speed'] * 1.4)
    elif action == 'set_speed': control['speed'] = max(0.05, min(3.0, float(data.get('value', 0.5))))
    elif action == 'deterministic':
        engine.deterministic = data.get('value', True)
        control['deterministic'] = engine.deterministic
    elif action == 'step':
        if engine.done:
            seed = control.get('seed')
            if seed is not None: seed = int(seed) + engine.episode_num
            clear_all_mn()
            prev_containment = {}
            ei = control.get('extra_infections', 0)
            state = engine.reset(seed=int(seed) if seed is not None else None, extra_infections=ei)
        else:
            state = engine.step()
        if state: broadcast_state(state)
    elif action == 'get_state':
        if latest_state: return jsonify(latest_state)
    return jsonify({'status':'ok','paused':control['paused'],'speed':round(control['speed'],3)})

def simulation_loop():
    global prev_containment
    while engine is None: time.sleep(0.1)
    seed = control.get('seed')
    ei = control.get('extra_infections', 0)
    clear_all_mn()
    prev_containment = {}
    state = engine.reset(seed=int(seed) if seed is not None else None, extra_infections=ei)
    broadcast_state(state)
    while True:
        if control['paused']:
            time.sleep(0.05)
            continue
        if control['mode'] == 'replay':
            recording = control.get('recording', [])
            idx = control.get('replay_index', 0)
            if idx < len(recording):
                state = recording[idx]
                control['replay_index'] = idx + 1
                broadcast_state(state)
            else:
                control['paused'] = True
            time.sleep(control['speed'])
            continue
        if engine.done:
            time.sleep(1.5)
            seed = control.get('seed')
            if seed is not None: next_seed = int(seed) + engine.episode_num
            else: next_seed = None
            clear_all_mn()
            prev_containment = {}
            state = engine.reset(seed=next_seed, extra_infections=ei)
            broadcast_state(state)
            time.sleep(control['speed'])
            continue
        state = engine.step()
        if state: broadcast_state(state)
        time.sleep(control['speed'])

def main():
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['live','replay'], default='live')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--recording', type=str, default=None)
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--stochastic', action='store_true', default=False)
    parser.add_argument('--autoplay', action='store_true', default=False)
    parser.add_argument('--extra-infections', type=int, default=0)
    parser.add_argument('--mininet', action='store_true', default=False)
    args = parser.parse_args()
    control['seed'] = args.seed
    control['speed'] = args.speed
    control['mode'] = args.mode
    control['paused'] = not args.autoplay
    control['extra_infections'] = args.extra_infections
    if args.mode == 'replay' and args.recording:
        with open(args.recording, 'r') as f:
            control['recording'] = json.load(f)
        control['replay_index'] = 0
    if args.mininet:
        setup_mininet()
    engine = EvalEngine(checkpoint_dir=args.checkpoint, device='cpu')
    engine.deterministic = not args.stochastic
    control['deterministic'] = engine.deterministic
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    mn_status = "ACTIVE" if mn_enabled else "OFF"
    print("\n" + "=" * 60)
    print("  SWARMSHIELD — RL + MININET + VISUAL DEMO")
    print("=" * 60)
    print("  Floor Plan:  http://0.0.0.0:" + str(args.port) + "/floor")
    print("  Dashboard:   http://0.0.0.0:" + str(args.port) + "/dashboard")
    print("  Mininet:     " + mn_status)
    print("  RL Policy:   " + ("stochastic" if args.stochastic else "deterministic"))
    print("  Speed:       " + str(args.speed) + "s per step")
    print("=" * 60 + "\n")
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    finally:
        stop_mininet()

if __name__ == '__main__':
    main()
