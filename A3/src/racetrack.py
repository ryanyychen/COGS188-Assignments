# racetrack.py

import numpy as np
import random

class RaceTrack(object):
    """
    RaceTrack object maintains and updates the race track
    state. Interaction with the class is through
    the take_action() method. The take_action() method returns
    a successor state and reward (i.e. s' and r)

    The class constructor is given a race course as a list of
    strings. The constructor loads the course and initializes
    the environment state.
    """

    def __init__(self, course):
        """
        Load race course, set any min or max limits in the
        environment (e.g. max speed), and set initial state.
        Initial state is random position on start line with
        velocity = (0, 0).

        Args:
            course: List of text strings used to construct
                race-track.
                    '+': start line
                    '-': finish line
                    'o': track
                    'X': wall

        Returns:
            self
        """
        self.NOISE = 0.0
        self.MAX_VELOCITY = 4
        self.start_positions = []
        self.course = None
        self._load_course(course)
        self._random_start_position()
        self.velocity = np.array([0, 0], dtype=np.int16)
        self.n_actions = 9 #valid action is acceleration of -1, 0, +1 in each axis x,y.

    def take_action(self, action):
        """
        Take action, update to next state and return reward

        Args:
            action: actions can be representated in two different and equivalent ways:
                1. Tuple of requested (x,y) acceleration; valid values [-1, 0, +1] in each axis.
                2. Integer 0 - 9 representing the 9 possible (x,y) accerlation combos

        Returns:
            reward: integer
        """
        assert type(action) in [tuple, int], "type of action should be tuple or int, is {}".format(type(action))

        if type(action) == int:
            assert action in range(self.n_actions)
            # turn it into a tuple representation for ._update_velocity()
            action = self.action_to_tuple(action)

        if type(action) == tuple:
            assert len(action) == 2
            assert action[0] in [-1, 0, 1]
            assert action[1] in [-1, 0, 1]

        if self.is_terminal_state():
            return 0
        self._update_velocity(action)
        return self._update_position() #this gives us reward

        

    def get_state(self):
        """Return array (position_x, position_y, velocity_x, velocity_y). """
        return np.concatenate( [self.position.copy(), self.velocity.copy()])

    def _update_velocity(self, action):
        """
        Update x- and y-velocity. Clip at 0 and self.MAX_VELOCITY

        Args:
            action: 2-tuple of requested change in velocity in x- and
                y-direction. valid action is -1, 0, +1 in each axis.
        """
        if np.random.rand() > self.NOISE:
            self.velocity += np.array(action, dtype=np.int16)
            self.velocity = np.minimum(self.velocity, self.MAX_VELOCITY)
            self.velocity = np.maximum(self.velocity, 0)

    def reset(self):
        self._random_start_position()
        self.velocity = np.array([0, 0], dtype=np.int16)

    def _update_position(self):
        """
        Update position based on present velocity. Check at fine time
        scale for wall or finish. If wall is hit, set position to random
        position at start line. If finish is reached, set position to
        first crossed point on finish line.

        Returns:
            reward: integer
        """
        for tstep in range(0, self.MAX_VELOCITY + 1):
            t = tstep / self.MAX_VELOCITY
            pos = self.position + np.round(self.velocity * t).astype(np.int16)
            if self._is_wall(pos):
                self._random_start_position()
                self.velocity = np.array([0, 0], dtype=np.int16)
                return -5 # this is the penalty for hitting a wall
            if self._is_finish(pos):
                self.position = pos
                self.velocity = np.array([0, 0], dtype=np.int16)
                return 0 
        self.position = pos
         # a small negative reward each (non-collision/non-finish) time step 
         # encourages finding the shortest path
        return -1

    def _random_start_position(self):
        """Set car to random position on start line"""
        self.position = np.array(random.choice(self.start_positions), dtype=np.int16)

    def _load_course(self, course):
        """Load course. Internally represented as numpy array"""
        y_size, x_size = len(course), len(course[0])
        self.course = np.zeros((x_size, y_size), dtype=np.int16)
        for y in range(y_size):
            for x in range(x_size):
                point = course[y][x]
                if point == "o":
                    self.course[x, y] = 1
                elif point == "-":
                    self.course[x, y] = 0
                elif point == "+":
                    self.course[x, y] = 2
                elif point == "W":
                    self.course[x, y] = -1
        # flip left/right so (0,0) is in bottom-left corner
        self.course = np.fliplr(self.course)
        for y in range(y_size):
            for x in range(x_size):
                if self.course[x, y] == 0:
                    self.start_positions.append((x, y))

    def _is_wall(self, pos):
        """Return True is position is wall"""
        return self.course[pos[0], pos[1]] == -1

    def _is_finish(self, pos):
        """Return True if position is finish line"""
        return self.course[pos[0], pos[1]] == 2

    def is_terminal_state(self):
        """Return True at episode terminal state"""
        return self.course[self.position[0], self.position[1]] == 2

    def action_to_tuple(self, a):
        """Convert integer action to 2-tuple: (ax, ay)"""
        ax = a // 3 - 1
        ay = a % 3 - 1

        return ax, ay

    def tuple_to_action(self, a):
        """Convert 2-tuple to integer action: {0-8}"""
        return int((a[0] + 1) * 3 + a[1] + 1)
    


# You could define some example courses here or in main.py:
big_course = [
    "WWWWWWWWWWWWWWWWWW",
    "WWWWooooooooooooo+",
    "WWWoooooooooooooo+",
    "WWWoooooooooooooo+",
    "WWooooooooooooooo+",
    "Woooooooooooooooo+",
    "Woooooooooooooooo+",
    "WooooooooooWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WoooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWooooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWoooooooWWWWWWWW",
    "WWWWooooooWWWWWWWW",
    "WWWWooooooWWWWWWWW",
    "WWWW------WWWWWWWW",
]

tiny_course = [
    "WWWWWW",
    "Woooo+",
    "Woooo+",
    "WooWWW",
    "WooWWW",
    "WooWWW",
    "WooWWW",
    "W--WWW",
]