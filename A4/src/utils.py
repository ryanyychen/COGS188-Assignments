import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

def run_episode(env, policy_fn, state_bins, action_bins):
    """
    
    Run a single episode of the environment using the specified policy.

    Args:
        env (Environment): The environment to interact with.
        policy_fn (callable): A function that takes a state and returns an action.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        tuple: Total reward obtained during the episode and list of frames and observations, and rewards
    """
    state = quantize_state(env.reset().observation, state_bins)
    total_reward = 0
    observations = []
    rewards = []
    frames = []
    ticks = []
    while True:
        action = policy_fn(state)
        time_step = env.step(action_bins[action])
        next_state = quantize_state(time_step.observation, state_bins)
        reward = time_step.reward
        total_reward += reward
        observations.append(time_step.observation)
        rewards.append(reward)
        # Render the environment
        frame = env.physics.render(camera_id=0, height=240, width=320)
        frames.append(frame)
        ticks.append(env.physics.data.time)
        if time_step.last():
            break

        state = next_state

    return total_reward, frames, (observations, rewards, ticks)

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())