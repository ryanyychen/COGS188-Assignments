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
    # TODO
    ...


def quantize_action(action: float, bins: list) -> int:
    """
    Quantize the action based on the provided bins. 
    """
    # TODO
    ...


