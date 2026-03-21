"""
SwarmShield IPPO (Independent PPO)
====================================

IPPO is just N independent PPO agents. That's it.

Each agent has its own actor network, its own critic network,
its own optimizer, and its own experience buffer. They don't
share weights, gradients, or anything else.

The multi-agent part is just:
- Each agent gets its own observation from the environment
- Each agent picks its own action independently
- Each agent gets its own reward
- Each agent updates independently

This class manages the N agents and provides convenient methods
for the training loop to interact with all agents at once.
"""

from agents.ppo import PPOAgent
from env.config import NUM_AGENTS


class IPPO:
    """
    Independent PPO: N separate PPO agents.

    This is the simplest multi-agent RL setup -> No communication,
    no shared parameters, no centralized critic. Each agent
    independently learns to be a good network defender.

    The emergent coordination, if any, comes from the shared
    reward signal and the fact that each agent observes the
    other agents' positions in its observation vector.
    """

    def __init__(self, device):
        """
        Create N independent PPO agents.

        device: torch device ('cpu', 'mps', or 'cuda')
        """
        self.device = device
        self.num_agents = NUM_AGENTS

        
        self.agents = []
        for i in range(self.num_agents):
            agent = PPOAgent(device)
            self.agents.append(agent)

    def select_actions(self, observations):
        """
        Each agent selects an action from its own observation.

        observations: list of N numpy arrays (one per agent)

        Returns:
            actions: list of N ints
            log_probs: list of N floats
            values: list of N floats
        """
        actions = []
        log_probs = []
        values = []

        for i in range(self.num_agents):
            action, log_prob, value = self.agents[i].select_action(observations[i])
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return actions, log_probs, values

    def store_transitions(self, observations, actions, log_probs, rewards, dones, values):
        """
        Store one timestep of experience for all agents.

        Each argument is a list of N items (one per agent).
        """
        for i in range(self.num_agents):
            self.agents[i].store_transition(
                observations[i],
                actions[i],
                log_probs[i],
                rewards[i],
                dones[i],
                values[i],
            )

    def update_all(self, last_observations):
        """
        Run PPO update for all agents.

        last_observations: list of N numpy arrays —> the observations
            after the last collected timestep, needed for GAE bootstrapping.

        Returns:
            list of N stat dicts (actor_loss, critic_loss, entropy per agent)
        """
        all_stats = []

        for i in range(self.num_agents):
            
            # Get the critic's value estimate for the last observation
            last_value = self.agents[i].get_value(last_observations[i])

            # Run PPO update for this agent
            stats = self.agents[i].update(last_value)
            all_stats.append(stats)

        return all_stats

    def save_all(self, directory):
        """Save all agent weights to a directory."""
        for i in range(self.num_agents):
            filepath = f"{directory}/agent_{i}.pt"
            self.agents[i].save(filepath)

    def load_all(self, directory):
        """Load all agent weights from a directory."""
        for i in range(self.num_agents):
            filepath = f"{directory}/agent_{i}.pt"
            self.agents[i].load(filepath)