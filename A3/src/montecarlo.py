import numpy as np
import random
from collections import defaultdict
from src.racetrack import RaceTrack

class MonteCarloControl:
    """
    Monte Carlo Control with Weighted Importance Sampling for off-policy learning.
    
    This class implements the off-policy every-visit Monte Carlo Control algorithm
    using weighted importance sampling to estimate the optimal policy for a given
    environment.
    """
    def __init__(self, env: RaceTrack, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
        """
        Initialize the Monte Carlo Control object. 

        Q, C, and policies are defaultdicts that have keys representing environment states.  
        Defaultdicts (search up the docs!) allow you to set a sensible default value 
        for the case of Q[new state never visited before] (and likewise with C/policies).  
        

        Hints: 
        - Q/C/*_policy should be defaultdicts where the key is the state
        - each value in the dict is a numpy vector where position is indexed by action
        - That is, these variables are setup like Q[state][action]
        - state key will be the numpy state vector cast to string (dicts require hashable keys)
        - Q should default to Q0, C should default to 0
        - *_policy should default to equiprobable (random uniform) actions
        - store everything as a class attribute:
            - self.env, self.gamma, self.Q, etc...

        Args:
            env (racetrack): The environment in which the agent operates.
            gamma (float): The discount factor.
            Q0 (float): the initial Q values for all states (e.g. optimistic initialization)
            max_episode_size (int): cutoff to prevent running forever during MC
        
        Returns: none, stores data as class attributes
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_size = max_episode_size

        # Initialize states
        self.all_states = self._generate_all_states()

        # Initialize Q with Q0 and C with 0
        self.Q = defaultdict(lambda: np.full(env.n_actions, Q0))
        self.C = defaultdict(lambda: np.zeros(env.n_actions))

        # Policy initializations
        self.target_policy = defaultdict(lambda: np.full(env.n_actions, 1/env.n_actions))
        self.behavior_policy = defaultdict(lambda: np.full(env.n_actions, 1/env.n_actions))
        
        # Generate the policies
        self.create_target_greedy_policy()
        self.create_behavior_egreedy_policy()
    
    def _generate_all_states(self):
        """
        Generate all possible states in the environment.
        
        Returns:
            list: A list of tuples representing all valid (x, y, vx, vy) states.
        """
        states = []
        for x in range(self.env.course.shape[0]):
            for y in range(self.env.course.shape[1]):
                for vx in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                    for vy in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                        if self.env.course[x, y] != -1:  # Not a wall
                            states.append((x, y, vx, vy))
        return states


    def create_target_greedy_policy(self):
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        greedy_policy = defaultdict(lambda: np.zeros(self.env.n_actions))
        for state in self.Q.keys():
            max_reward = 0
            best_action = 0
            for action in range(self.env.n_actions):
                # Test all actions and find optimal
                reward, next_state = self._simulate_action(state, action)
                # Store best action
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            # Create probability vector for greedy action
            prob = np.zeros(self.env.n_actions)
            prob[best_action] = 1
            greedy_policy[state] = prob
            assert np.sum(prob) == 1, "Probabilities do not sum to 1"

        self.target_policy = greedy_policy


    def create_behavior_egreedy_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        for state in self.target_policy.keys():
            prob = self.target_policy[state]

            # Create epsilon-greedy probability vector
            new_prob = np.zeros(self.env.n_actions)
            new_prob.fill(self.epsilon/(self.env.n_actions - 1))
            new_prob[np.argmax(prob)] = 1 - self.epsilon

            # Store new probability vector
            self.behavior_policy[state] = new_prob
            assert np.sum(new_prob) == 1, "Probabilities do not sum to 1"
        
        # Update target policy to behavior policy
        self.target_policy = self.behavior_policy
        
    def egreedy_selection(self, state):
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        prob = self.behavior_policy[state]
        assert np.sum(prob) == 1, "Probabilities do not sum to 1"
        action = random.choices(range(self.env.n_actions), weights=prob)
        return int(action[0])

    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        episode = []
        self.env.reset()
        state = None

        for _ in range(self.max_episode_size):
            state = tuple(self.env.get_state())

            # Select and take action
            action = self.egreedy_selection(state)
            reward = self.env.take_action(action)

            # Track action in episode
            episode.append((state, action, reward))

            # Exit early if terminal state
            if self.env.is_terminal_state():
                break
        return episode
        
    
    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        episode = []
        self.env.reset()
        state = None

        for _ in range(self.max_episode_size):
            state = tuple(self.env.get_state())

            # Select and take action
            action = int(np.argmax(self.target_policy[state]))
            reward = self.env.take_action(action)

            # Track action in episode
            episode.append((state, action, reward))

            # Exit early if terminal state
            if self.env.is_terminal_state():
                break

        return episode
    
    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        func_val = 0
        weight = 1

        for state, action, reward in reversed(episode):
            func_val = reward + self.gamma * func_val

            # Update cumulative count
            self.C[state][action] += weight

            # Update Q value
            self.Q[state][action] += (weight / self.C[state][action]) * (func_val - self.Q[state][action])

            # Not a valid action (probability = 0)
            if self.target_policy[state][action] == 0:
                break

            # Update weight
            weight *= self.target_policy[state][action] / self.behavior_policy[state][action]
    
    def update_onpolicy(self, episode):
        """
        Update the Q-values using first visit epsilon-greedy. 
        See Figure 5.6, p. 127 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        func_val = 0
        visited = set()

        for state, action, reward in reversed(episode):
            func_val = reward + self.gamma * func_val

            if (state, action) not in visited:
                visited.add((state, action))
                self.C[state][action] += 1
                self.Q[state][action] += 1/self.C[state][action] * (func_val - self.Q[state][action])


    def train_offpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for _ in range(num_episodes):
            episode = self.generate_egreedy_episode()
            self.update_offpolicy(episode)


    def get_greedy_policy(self):
        """
        Retrieve the learned target policy in the form of an action index per state
        
        Returns:
            dict: The learned target policy.
        """
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = np.argmax(actions)
        return policy
    
    def _simulate_action(self, state, action):
        """
        Simulate taking an action from a given state.
        
        Args:
            state (tuple): The current state (x, y, vx, vy).
            action (int): The action to take.
        
        Returns:
            tuple: (reward, new_state) where new_state is the next state tuple.
        """
        x, y, vx, vy = state
        self.env.position = np.array([x, y])
        self.env.velocity = np.array([vx, vy])
        reward = self.env.take_action(int(action))
        new_state = tuple(self.env.get_state())
        return reward, new_state