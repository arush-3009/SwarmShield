import numpy as np
from env.swarmshield_env import SwarmShieldEnv
from env.config import NUM_AGENTS, NUM_ACTIONS, MAX_TIMESTEPS

env = SwarmShieldEnv()
obs, infos = env.reset()

print("Environment created successfully.")
print(f"Observation shape per agent: {obs[0].shape}")
print(f"Number of agents: {NUM_AGENTS}")
print(f"Number of actions: {NUM_ACTIONS}")
print(f"Initial network state:")
print(env)
print()

total_reward = 0.0
rng = np.random.default_rng(42)

for step in range(MAX_TIMESTEPS + 5):
    # Random actions for all agents
    actions = []
    for i in range(NUM_AGENTS):
        actions.append(rng.integers(0, NUM_ACTIONS))

    obs, rewards, dones, truncateds, infos = env.step(actions)
    total_reward += rewards[0]

    if step % 50 == 0:
        info = infos[0]
        print(
            f"Step {step:3d} | "
            f"Reward {rewards[0]:7.1f} | "
            f"Infected {info.get('infected_count', '?')} | "
            f"Blocked {info.get('blocked_count', '?')} | "
            f"Quarantined {info.get('quarantined_count', '?')} | "
            f"ServerDmg {info.get('server_damage', '?'):.1f}"
        )

    if dones[0] or truncateds[0]:
        print(f"\nEpisode ended at step {step}")
        if dones[0]:
            print("Terminated (server compromised or all contained)")
        else:
            print("Truncated (max timesteps reached)")
        print(f"Total reward (agent 0): {total_reward:.1f}")
        print(f"Final info: {infos[0]}")
        break

print("\nTest PASSED. Environment runs without errors.")