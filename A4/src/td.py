import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
from dm_control.rl.control import Environment

# Initialize the Q-table with dimensions corresponding to each discretized state variable.
def initialize_q_table(state_bins: dict, action_bins: list) -> np.ndarray:
    """
    Initialize the Q-table with dimensions corresponding to each discretized state variable.

    Args:
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        np.ndarray: A Q-table initialized to zeros with dimensions matching the state and action space.
    """
    # States (keys) = state_bins[position][0][2] * state_bins[position][1][2] * state_bins[position][2][2] *
    #                   state_bins[velocity][0][2] * state_bins[velocity][1][2]
    # Actions = len(action_bins)
    pos_1 = len(state_bins['position'][0])
    pos_2 = len(state_bins['position'][1])
    pos_3 = len(state_bins['position'][2])
    vel_1 = len(state_bins['velocity'][0])
    vel_2 = len(state_bins['velocity'][1])
    q_table = np.zeros((pos_1, pos_2, pos_3, vel_1, vel_2, len(action_bins)))
    return q_table
    


# TD Learning algorithm
def td_learning(env: Environment, num_episodes: int, alpha: float, gamma: float, epsilon: float, state_bins: dict, action_bins: list, q_table:np.ndarray=None) -> tuple:
    """
    TD Learning algorithm for the given environment.

    Args:
        env (Environment): The environment to train on.
        num_episodes (int): The number of episodes to train.
        alpha (float): The learning rate. 
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.
        q_table (np.ndarray): The Q-table to start with. If None, initialize a new Q-table.

    Returns:
        tuple: The trained Q-table and the list of total rewards per episode.
    """
    if q_table is None:
        q_table = initialize_q_table(state_bins, action_bins)
        
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # reset env
        timestep = env.reset()
        while not timestep.last():
            # Select Action
            if np.random.rand() < epsilon:
                action_id = np.random.randint(len(action_bins))
            else:
                action_id = greedy_policy(q_table)(quantize_state(timestep.observation, state_bins))
            action = action_bins[action_id]
            
            # Save state before action
            state = quantize_state(timestep.observation, state_bins)

            # Take action
            timestep = env.step(action)

            # Update Q-table
            reward = timestep.reward
            
            next_state = quantize_state(timestep.observation, state_bins)
            max_next_reward_action = greedy_policy(q_table)(next_state)
            q_table[state][action_id] += alpha * (reward + gamma * q_table[next_state][max_next_reward_action] - q_table[state][action_id])
            
            # Track rewards
            rewards.append(reward)

    return q_table, rewards


def greedy_policy(q_table: np.ndarray) -> callable:
    """
    Define a greedy policy based on the Q-table.    

    Args:
        q_table (np.ndarray): The Q-table from which to derive the policy.

    Returns:
        callable: A function that takes a state and returns the best action. 
    """
    def policy(state):
        return np.argmax(q_table[state])
    return policy