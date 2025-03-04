import numpy as np


# Quantize the state space and action space
def quantize_state(state: dict, state_bins: dict) -> tuple:
    """
    
    Given the state and the bins for each state variable, quantize the state space.

    Args:
        state (dict): The state to be quantized.
        state_bins (dict): The bins used for quantizing each dimension of the state.

    Returns:
        tuple: The quantized representation of the state.
    """
    # position: [x1, x2, x3], velocity: [v1, v2]
    quantized = []
    for key in state.keys():
        for i in range(len(state[key])):
            quantized.append(int(np.digitize(state[key][i], state_bins[key][i])) - 1)
    return tuple(quantized)


def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    bin_num = np.digitize(action, bins)
    if bin_num == 0:
        return 0.0
    return float(bin_num) - 1


# Define custom bin configurations for each observation dimension.
# For each observation variable, specify a list of (min, max, bin_nums)
# In this example, we assume:
#   - 'position' has 3 dimensions and we use BIGGER_SIZE bins for each.
#   - 'velocity' has 2 dimensions and we use MEDIUM_SIZE bins for each.
# Define custom bin configurations for each observation dimension individually.
# You can adjust the range and bin count for each dimension as needed.
pos_bins_config = [
    (-1.0, 1.0, 2),  # Config for the first position component
    (-1.0, 1.0, 2),  # Config for the second position component
    (-2.0, 2.0, 5),  # Config for the third position component
]

vel_bins_config = [
    (-1.0, 1.0, 2),  # Config for the first velocity component
    (-10.0, 10.0, 5),  # Config for the second velocity component
]

obs_bins_config = {"position": pos_bins_config, "velocity": vel_bins_config}

# Create a list of bins arrays corresponding to each observation dimension.
state_bins = {
    key: [
        np.linspace(min_val, max_val, bin_num) for min_val, max_val, bin_num in config
    ]
    for key, config in obs_bins_config.items()
}

action_bins_config = (-1, 1, 4)

action_bins = np.linspace(*action_bins_config)

from dm_control import suite
random_state = np.random.RandomState(42)
env = suite.load("cartpole", "balance", task_kwargs={"random": random_state})
time_step = env.reset()
obs = time_step.observation

quantized_state = quantize_state(obs, state_bins)
print("Quantized State:", quantized_state)