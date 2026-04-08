# SwarmShield

SwarmShield is a multi-agent reinforcement learning approach for cyber defense in a simulated office network.

The project combines:

- a custom Python environment with 18 hosts across 6 subnets
- a scripted attacker that beacons, scans, spreads laterally, and attacks a file server
- 3 independent PPO defender agents trained with IPPO-style interaction
- a browser-based visual demo
- an optional Mininet-backed Linux demo that mirrors containment actions as real `iptables` rules

## What This Repo Contains

- `env/`
  The environment, attacker logic, network model, traffic generation, and test script.
- `agents/`
  PPO agent, IPPO wrapper, and neural network definitions.
- `train/`
  Training entrypoint.
- `visual-demo/`
  Flask server, visual state builder, browser UI, replay recorder, and seed finder.
- `rl_demo.py`
  Terminal + Mininet demo that runs the trained policy directly.
- `checkpoints/`
  Saved model weights and a training plot.
- `Environment-Design-Spec.md`
  Design document for the environment.

## Recommended Platform

Training can run on macOS or Linux.

The Mininet-backed demos should be run on Linux.


## Python Requirements

This project uses these Python packages directly:

- `numpy`
- `torch`
- `matplotlib`
- `Flask`

The standard library covers the rest.

Mininet is not installed through `pip`; install it through Linux package manager.

## Quick Setup

Run all commands from the repo root.

### 1. Clone the repo

```bash
git clone <https://github.com/arush-3009/SwarmShield>
cd SwarmShield
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

### 4. Install Mininet on Linux for the Mininet demos

On Ubuntu:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch
```

## Files Needed To Reproduce Results

The repo already includes trained checkpoints in:

- `checkpoints/`
- `checkpoints/best/`
- `checkpoints/latest/`

That means you can run the demos immediately without retraining.

## Testing The Environment

From the repo root:

```bash
python3 -m env.test_env
```

## Run The Browser Visual Demo

From the repo root:

```bash
python3 visual-demo/server.py --port 5050 --autoplay
```

Then open:

- `http://127.0.0.1:5050/floor`
- `http://127.0.0.1:5050/dashboard`

If running from inside a Linux VM, open the VM IP from host machine instead.

### Useful visual demo flags

Deterministic policy, autoplay:

```bash
python3 visual-demo/server.py --port 5050 --autoplay
```

Stochastic policy:

```bash
python3 visual-demo/server.py --port 5050 --autoplay --stochastic
```

Start from a chosen seed:

```bash
python3 visual-demo/server.py --port 5050 --autoplay --seed 42
```

Start with extra infected hosts:

```bash
python3 visual-demo/server.py --port 5050 --autoplay --extra-infections 2
```

Replay a saved recording:

```bash
python3 visual-demo/server.py --mode replay --recording visual-demo/recordings/seed_42.json --port 5050 --autoplay
```

## Run The Browser Visual Demo With Mininet

Run this on Linux:

```bash
sudo -E python3 visual-demo/server.py --port 5050 --autoplay --mininet
```

This starts the visual demo and also mirrors containment actions into Mininet using `iptables`.

To watch the Mininet command log in another terminal:

```bash
./mn_tail.sh
```

The Mininet command log is written to:

- `mininet_commands.log`

## Run The Terminal Mininet Demo

Run this on Linux:

```bash
sudo -E python3 rl_demo.py
```

Optional seed and speed:

```bash
sudo -E python3 rl_demo.py 42 0.5
```

Arguments:

- first positional argument: seed
- second positional argument: seconds per step

This demo:

- builds the Mininet topology
- loads trained checkpoints
- runs the environment
- applies block/quarantine/unblock actions as real `iptables` rules

## Record A Demo Episode

From the repo root:

```bash
python3 visual-demo/recorder.py --seed 42
```

Optional output path:

```bash
python3 visual-demo/recorder.py --seed 42 --output visual-demo/recordings/seed_42.json
```

## Search For Good Demo Seeds

From the repo root:

```bash
python3 visual-demo/seed_finder.py --start 0 --count 200 --top 10
```

This scores deterministic rollouts and prints the best seeds for demos.

## Train The Agents

From the repo root:

```bash
python3 train/training.py
```

- training writes checkpoints to `checkpoints/`
- training resumes automatically from `checkpoints/latest/`

### Device selection

Training device is controlled manually in:

- `train/training.py`

using

- `DEVICE_TO_TRAIN_ON = 2`

Meaning:

- `0` = CUDA GPU
- `1` = Apple MPS
- `2` = CPU


## Reproduce The Current Demo Setup

### Option A: Linux machine only

1. Clone the repo.
2. Create and activate a Python virtual environment.
3. Install `requirements.txt`.
4. Install Mininet through `apt` if you want Mininet-backed demos.
5. Run `python3 -m env.test_env` once.
6. Run `python3 visual-demo/server.py --port 5050 --autoplay` for the browser demo.
7. Run `sudo -E python3 visual-demo/server.py --port 5050 --autoplay --mininet` if you want Mininet integration.
8. Run `sudo -E python3 rl_demo.py` for the terminal Mininet demo.

### Option B: macOS for editing, Linux VM for demos

1. Edit and commit on Mac.
2. Push to GitHub.
3. Pull the changes inside the Linux VM.
4. In the VM, activate the Python virtual environment.
5. In the VM, run:

```bash
python3 visual-demo/server.py --port 5050 --autoplay
```

6. From Mac browser, open the VM IP:

```text
http://<VM_IP>:5050/floor
http://<VM_IP>:5050/dashboard
```

7. For Mininet-backed behavior, run instead:

```bash
sudo -E python3 visual-demo/server.py --port 5050 --autoplay --mininet
```

8. For the terminal Mininet demo:

```bash
sudo -E python3 rl_demo.py
```

## Controls In The Floor/Dashboard Demo

Keyboard shortcuts:

- `Space` = play/pause
- `R` = restart
- `+` or `=` = speed up
- `-` or `_` = slow down
- `N` = single step

## Note

- The repo contains trained checkpoints already, so demos can run without fresh training.
