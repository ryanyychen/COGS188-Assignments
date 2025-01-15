import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        if (node in self.neighbors):
            return
        
        self.neighbors.append(node)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    nodes_dict = {}

    for i in range(rows):
        for j in range(cols):
            state = maze[i][j]
            if (state == 0):
                # Create new node if not in nodes_dict, otherwise retrieve existing node
                node = None
                if (nodes_dict.get((i,j)) != None):
                    node = nodes_dict.get((i,j))
                else:
                    node = Node((i,j))
                    nodes_dict[(i,j)] = node

                # Top neighbor
                if (i > 0 and maze[i-1][j] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i-1, j))
                    if (neighbor == None):
                        neighbor = Node((i-1, j))
                        nodes_dict[(i-1, j)] = neighbor
                        
                    node.add_neighbor(neighbor)

                # Bottom neighbor
                if (i < rows - 1 and maze[i+1][j] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i+1, j))
                    if (neighbor == None):
                        neighbor = Node((i+1, j))
                        nodes_dict[(i+1, j)] = neighbor

                    node.add_neighbor(Node((i+1, j)))

                # Left neighbor
                if (j > 0 and maze[i][j-1] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i, j-1))
                    if (neighbor == None):
                        neighbor = Node((i, j-1))
                        nodes_dict[(i, j-1)] = neighbor
                        
                    node.add_neighbor(Node((j, j-1)))

                # Right neighbor
                if (j < cols - 1 and maze[i][j+1] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i, j+1))
                    if (neighbor == None):
                        neighbor = Node((i, j+1))
                        nodes_dict[(i, j+1)] = neighbor

                    node.add_neighbor(Node((i, j+1)))

    # TODO: Assign start_node and goal_node if they exist in nodes_dict
    start_node = nodes_dict.get((0, 0))
    goal_node = nodes_dict.get((rows-1, cols-1))

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS
    queue = deque()
    visited = []
    parents = {}
    queue.append((start_node, None))

    # Run BFS until all nodes explored
    while (len(queue) != 0):
        # Use currNode and parentNode to track BFS state
        next = queue.popleft()
        currNode, parentNode = next[0], next[1]

        visited.append(currNode)
        parents[currNode] = parentNode

        if (currNode == goal_node):
            break

        for neighbor in currNode.neighbors:
            if (neighbor not in visited):
                queue.append((neighbor, currNode))
    
    if currNode != goal_node:
        return None
    
    # Reconstruct path
    path = [goal_node.value]
    currNode = goal_node
    while (parents[currNode] != None):
        prevNode = parents[currNode]
        path.append(prevNode.value)
        currNode = prevNode
    return path


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    return 0


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    return None


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    return None


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
