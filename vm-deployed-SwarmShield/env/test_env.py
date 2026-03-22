"""
Simple test file for SwarmShieldEnv.

Purpose:
- Print high-level environment information first.
- Reset the environment and print a clear summary.
- Run a few simple test steps.
- Then run one short random episode and print everything in a readable way.

This is meant for quick repeated sanity-checking, not for training.
"""

import random

from env.config import (
    NUM_AGENTS,
    NUM_ACTIONS,
    NUM_HOSTS,
    NUM_SUBNETS,
    OBSERVATION_SIZE,
    MAX_TIMESTEPS,
    SERVER_DAMAGE_THRESHOLD,
    ACTION_OBSERVE,
    ACTION_BLOCK,
    ACTION_QUARANTINE,
    ACTION_UNBLOCK,
    ACTION_MOVE_BASE,
    ACTION_MOVE_LAST,
)
from env.swarmshield_env import SwarmShieldEnv


# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------

def line(char="=", width=80):
    print(char * width)


def section(title):
    line("=")
    print(title)
    line("=")


def subsection(title):
    print()
    line("-")
    print(title)
    line("-")


def action_to_text(action):
    if action == ACTION_OBSERVE:
        return "OBSERVE"

    if ACTION_MOVE_BASE <= action <= ACTION_MOVE_LAST:
        target_host = action - ACTION_MOVE_BASE
        return f"MOVE_TO_HOST_{target_host}"

    if action == ACTION_BLOCK:
        return "BLOCK_CURRENT_HOST"

    if action == ACTION_QUARANTINE:
        return "QUARANTINE_CURRENT_HOST"

    if action == ACTION_UNBLOCK:
        return "UNBLOCK_CURRENT_HOST"

    return f"UNKNOWN_ACTION_{action}"


def print_env_overview():
    section("SWARMSHIELD ENVIRONMENT OVERVIEW")

    print("High-level configuration:")
    print(f"- Number of defender agents: {NUM_AGENTS}")
    print(f"- Number of hosts: {NUM_HOSTS}")
    print(f"- Number of subnets: {NUM_SUBNETS}")
    print(f"- Number of discrete actions per agent: {NUM_ACTIONS}")
    print(f"- Observation vector length per agent: {OBSERVATION_SIZE}")
    print(f"- Maximum episode length: {MAX_TIMESTEPS}")
    print(f"- Server damage threshold for attacker win: {SERVER_DAMAGE_THRESHOLD}")

    print()
    print("Action space meaning:")
    print(f"- 0 = OBSERVE")
    print(f"- {ACTION_MOVE_BASE} to {ACTION_MOVE_LAST} = MOVE TO HOST 0 to HOST {NUM_HOSTS - 1}")
    print(f"- {ACTION_BLOCK} = BLOCK CURRENT HOST")
    print(f"- {ACTION_QUARANTINE} = QUARANTINE CURRENT HOST")
    print(f"- {ACTION_UNBLOCK} = UNBLOCK CURRENT HOST")

    print()
    print("What this test file does:")
    print("- Resets the environment")
    print("- Prints initial observations and initial world state")
    print("- Runs a few small hand-picked test actions")
    print("- Runs one short random episode")
    print("- Prints rewards, infection counts, containment counts, transit state, and termination flags")


def print_agent_observation_summary(obs_list):
    subsection("OBSERVATION SUMMARY")

    print(f"Number of observation vectors returned: {len(obs_list)}")
    for i, obs in enumerate(obs_list):
        print(f"Agent {i}: observation length = {len(obs)}, first 10 values = {obs[:10]}")


def print_infos_summary(infos):
    subsection("AGENT INFO SUMMARY")

    for i, info in enumerate(infos):
        print(f"Agent {i}:")
        print(f"  current_host                = {info['current_host']}")
        print(f"  in_transit                  = {info['in_transit']}")
        print(f"  transit_target              = {info['transit_target']}")
        print(f"  transit_remaining           = {info['transit_remaining']}")
        print(f"  infected_total              = {info['infected_total']}")
        print(f"  infected_uncontained        = {info['infected_uncontained']}")
        print(f"  infected_blocked            = {info['infected_blocked']}")
        print(f"  infected_quarantined        = {info['infected_quarantined']}")
        print(f"  clean_total                 = {info['clean_total']}")
        print(f"  clean_blocked               = {info['clean_blocked']}")
        print(f"  clean_quarantined           = {info['clean_quarantined']}")
        print(f"  blocked_total               = {info['blocked_total']}")
        print(f"  quarantined_total           = {info['quarantined_total']}")
        print(f"  server_damage               = {info['server_damage']}")
        print(f"  overlap_pairs               = {info['overlap_pairs']}")
        print(f"  all_infections_quarantined  = {info['all_infections_quarantined']}")
        print(f"  server_compromised          = {info['server_compromised']}")
        print()


def print_step_result(step_idx, actions, rewards, terminated, truncated, infos):
    section(f"STEP {step_idx} RESULT")

    print("Actions taken this step:")
    for i, action in enumerate(actions):
        print(f"- Agent {i}: {action_to_text(action)} ({action})")

    print()
    print("Rewards returned this step:")
    for i, reward in enumerate(rewards):
        print(f"- Agent {i}: reward = {reward:.4f}")

    print()
    print("Termination flags:")
    for i in range(len(terminated)):
        print(f"- Agent {i}: terminated = {terminated[i]}, truncated = {truncated[i]}")

    print()
    print("Environment state after this step:")
    # All agents should see same global counts, so reading from info[0] is enough
    shared = infos[0]
    print(f"- timestep                     = {shared['timestep']}")
    print(f"- newly infected this step     = {shared['newly_infected_this_step']}")
    print(f"- num new infections this step = {shared['num_new_infections_this_step']}")
    print(f"- infected_total               = {shared['infected_total']}")
    print(f"- infected_uncontained         = {shared['infected_uncontained']}")
    print(f"- infected_blocked             = {shared['infected_blocked']}")
    print(f"- infected_quarantined         = {shared['infected_quarantined']}")
    print(f"- clean_total                  = {shared['clean_total']}")
    print(f"- clean_blocked                = {shared['clean_blocked']}")
    print(f"- clean_quarantined            = {shared['clean_quarantined']}")
    print(f"- blocked_total                = {shared['blocked_total']}")
    print(f"- quarantined_total            = {shared['quarantined_total']}")
    print(f"- server_damage                = {shared['server_damage']:.2f}")
    print(f"- server_damage_delta          = {shared['server_damage_delta']:.2f}")
    print(f"- overlap_pairs                = {shared['overlap_pairs']}")
    print(f"- all_infections_quarantined   = {shared['all_infections_quarantined']}")
    print(f"- server_compromised           = {shared['server_compromised']}")

    print()
    print("Per-agent runtime state:")
    for i, info in enumerate(infos):
        print(f"- Agent {i}: host={info['current_host']}, in_transit={info['in_transit']}, "
              f"target={info['transit_target']}, remaining={info['transit_remaining']}, "
              f"local_event_reward={info['local_event_reward']:.4f}, "
              f"shared_reward={info['shared_reward']:.4f}, "
              f"terminal_reward={info['terminal_reward']:.4f}")


# -----------------------------------------------------------------------------
# Test scenarios
# -----------------------------------------------------------------------------

def run_basic_reset_test(env):
    section("TEST 1: RESET TEST")

    observations, infos = env.reset()

    print("Environment reset completed successfully.")
    print_agent_observation_summary(observations)
    print_infos_summary(infos)


def run_simple_manual_steps(env):
    section("TEST 2: SIMPLE MANUAL STEPS")

    observations, infos = env.reset()

    print("Starting from a fresh reset for manual steps.")
    print("We will do a few easy-to-read actions first.")

    # Step 1: everyone observes
    actions = [ACTION_OBSERVE, ACTION_OBSERVE, ACTION_OBSERVE]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    print_step_result(1, actions, rewards, terminated, truncated, infos)

    if any(terminated) or any(truncated):
        return

    # Step 2: move agents to different places
    actions = [
        ACTION_MOVE_BASE + 1,   # agent 0 move to host 1
        ACTION_MOVE_BASE + 6,   # agent 1 move to host 6
        ACTION_MOVE_BASE + 13,  # agent 2 move to host 13
    ]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    print_step_result(2, actions, rewards, terminated, truncated, infos)

    if any(terminated) or any(truncated):
        return

    # Step 3: while some may still be in transit, ask all to observe
    actions = [ACTION_OBSERVE, ACTION_OBSERVE, ACTION_OBSERVE]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    print_step_result(3, actions, rewards, terminated, truncated, infos)

    if any(terminated) or any(truncated):
        return

    # Step 4: try containment actions on current hosts
    actions = [ACTION_BLOCK, ACTION_QUARANTINE, ACTION_UNBLOCK]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    print_step_result(4, actions, rewards, terminated, truncated, infos)


def run_random_episode(env, max_steps=12):
    section("TEST 3: SHORT RANDOM EPISODE")

    observations, infos = env.reset()
    print("Fresh episode started.")
    print("Now running a short random episode so you can repeatedly sanity-check behavior.")

    step_idx = 0
    done = False

    while not done and step_idx < max_steps:
        step_idx += 1

        actions = []
        for _ in range(NUM_AGENTS):
            action = random.randint(0, NUM_ACTIONS - 1)
            actions.append(action)

        observations, rewards, terminated, truncated, infos = env.step(actions)
        print_step_result(step_idx, actions, rewards, terminated, truncated, infos)

        done = any(terminated) or any(truncated)

    subsection("RANDOM EPISODE FINISHED")
    if done:
        print("Episode ended because at least one termination/truncation flag became True.")
    else:
        print("Random episode stopped because the short test step limit was reached.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print_env_overview()

    subsection("CREATING ENVIRONMENT")
    env = SwarmShieldEnv(seed=42)
    print("Environment object created successfully.")

    run_basic_reset_test(env)
    run_simple_manual_steps(env)
    run_random_episode(env, max_steps=10)

    section("ALL TESTS FINISHED")
    print("If you reached this point without crashes, the environment is at least structurally working.")
    print("Next, you can keep rerunning this file whenever you change env logic.")


if __name__ == "__main__":
    main()