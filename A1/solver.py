import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np
import time

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
            if (maze[i][j] == 0):
                # Create new node if not in nodes_dict, otherwise retrieve existing node
                node = nodes_dict.get((i,j))
                if (node == None):
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
                if (i < rows-1 and maze[i+1][j] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i+1, j))
                    if (neighbor == None):
                        neighbor = Node((i+1, j))
                        nodes_dict[(i+1, j)] = neighbor

                    node.add_neighbor(neighbor)

                # Left neighbor
                if (j > 0 and maze[i][j-1] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i, j-1))
                    if (neighbor == None):
                        neighbor = Node((i, j-1))
                        nodes_dict[(i, j-1)] = neighbor
                        
                    node.add_neighbor(neighbor)

                # Right neighbor
                if (j < cols-1 and maze[i][j+1] == 0):
                    # Get neighbor node if existing, else create new node for it
                    neighbor = nodes_dict.get((i, j+1))
                    if (neighbor == None):
                        neighbor = Node((i, j+1))
                        nodes_dict[(i, j+1)] = neighbor

                    node.add_neighbor(neighbor)
            else:
                continue

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
    if (start_node == None or goal_node == None):
        return None
    
    queue = []
    visited = []
    parents = {}
    queue.append((start_node, None))

    # Run BFS until all nodes explored
    while (len(queue) != 0):
        # Use currNode and parentNode to track BFS state
        next = queue.pop(0)
        currNode, parentNode = next[0], next[1]

        visited.append(currNode)
        parents[currNode] = parentNode

        if (currNode == goal_node):
            break

        for neighbor in currNode.neighbors:
            if (neighbor not in visited and (neighbor, currNode) not in queue):
                queue.append((neighbor, currNode))
    
    if currNode != goal_node:
        return None
    
    # Reconstruct path
    return reconstruct_path(goal_node, parents)


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
    if (start_node == None or goal_node == None):
        return None
    
    stack = list()
    visited = []
    parents = {}
    stack.append((start_node, None))

    # Run BFS until all nodes explored
    while (len(stack) != 0):
        # Use currNode and parentNode to track BFS state
        next = stack.pop()
        currNode, parentNode = next[0], next[1]

        visited.append(currNode)
        parents[currNode] = parentNode

        if (currNode == goal_node):
            break

        for neighbor in currNode.neighbors:
            if (neighbor not in visited and (neighbor, currNode) not in stack):
                stack.append((neighbor, currNode))
    
    if currNode != goal_node:
        return None
    
    # Reconstruct path
    return reconstruct_path(goal_node, parents)


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
    if (start_node == None or goal_node == None):
        return None
    
    pq = []
    gscore = {(0, 0): manhattan_distance(start_node, start_node)}
    fscore = {(0, 0): manhattan_distance(start_node, goal_node)}
    heapq.heappush(pq, (fscore[(0, 0)], start_node, None))
    visited = []
    parents = {}
    
    # Run A* until all nodes explored or goal found
    while (len(pq) != 0):
        # Use currNode and parentNode to track BFS state
        next = heapq.heappop(pq)
        cost, currNode, parentNode = next[0], next[1], next[2]

        visited.append(currNode)
        parents[currNode] = parentNode

        if (currNode == goal_node):
            break

        # g_score = parent's g_score + 1 step
        g_score = gscore[currNode.value] + 1

        for neighbor in currNode.neighbors:
            f_score = g_score + manhattan_distance(neighbor, goal_node)
            # Update gscore and fscore disctionaries accordingly as needed
            if (neighbor.value not in fscore.keys() or f_score < fscore[neighbor.value]):
                gscore[neighbor.value] = g_score
                fscore[neighbor.value] = f_score

            if (neighbor not in visited):
                tuple = (fscore[neighbor.value], neighbor, currNode)
                if (tuple not in pq):
                    heapq.heappush(pq, tuple)
        #time.sleep(0.1)
    
    if currNode != goal_node:
        return None

    # Reconstruct path
    return reconstruct_path(goal_node, parents)

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    return abs(node_a.value[0] - node_b.value[0]) \
         + abs(node_a.value[1] - node_b.value[1])


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
    if (start_node == None or goal_node == None):
        return None
    
    qStart = []
    qGoal = []
    vStart = []
    vGoal = []
    parentStart = {}
    parentGoal = {}
    qStart.append((start_node, None))
    qGoal.append((goal_node, None))
    connecting = None

    # Run Double BFS until all nodes explored or path found
    while (len(qStart) != 0 and len(qGoal) != 0):
        # Pop from both queues
        nextStart = qStart.pop(0)
        nextGoal = qGoal.pop(0)

        # Grab current nodes and parents from both BFS
        currStart, pStart = nextStart[0], nextStart[1]
        currGoal, pGoal = nextGoal[0], nextGoal[1]

        # Append current nodes to respective visited lists
        vStart.append(currStart)
        vGoal.append(currGoal)

        # Track parents
        parentStart[currStart] = pStart
        parentGoal[currGoal] = pGoal

        if (currStart in vGoal):
            connecting = currStart
            break
        elif (currGoal in vStart):
            connecting = currGoal
            break

        for neighborStart in currStart.neighbors:
            if (neighborStart not in vStart and (neighborStart, currStart) not in qStart):
                qStart.append((neighborStart, currStart))
        
        for neighborGoal in currGoal.neighbors:
            if (neighborGoal not in vGoal and (neighborGoal, currGoal) not in qGoal):
                qGoal.append((neighborGoal, currGoal))
    
    if (connecting == None):
        return None
    
    # Reconstruct path
    fromStart = reconstruct_path(connecting, parentStart)
    fromGoal = reconstruct_path(connecting, parentGoal)
    fromGoal.reverse()
    path = fromStart + fromGoal
    return path


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

    if (start_node == None or goal_node == None):
        return None

    parents = {(start_node.value): None}
    max_iter = 1000
    curr = start_node
    cost = manhattan_distance(curr, goal_node)
    path = [start_node.value]

    for i in range(max_iter):
        # Break if no neighbors
        if (len(curr.neighbors) == 0):
            break

        # Select random neighbor of current node
        neighbor_idx = random.randint(0, len(curr.neighbors)-1)
        neighbor = curr.neighbors[neighbor_idx]

        # Compute neighbor's cost
        next_cost = manhattan_distance(neighbor, goal_node)

        # Move if cost is lower or with probability e^(-cost_diff / temperature)
        if (next_cost < cost or random.random() <= math.exp(-1*(cost - next_cost)/temperature)):
            # Track prevNode for neighbor
            parents[neighbor] = curr
            path.append(neighbor.value)
            curr = neighbor
            cost = next_cost
        
        if (curr == goal_node):
            break
        
        # Reduce temperature
        if (temperature >= min_temperature):
            temperature -= cooling_rate
    
    if (curr != goal_node):
        return None

    return path


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
    path = []
    path.append(end_node.value)
    currNode = end_node
    while (parent_map[currNode] != None):
        prevNode = parent_map[currNode]
        path.append(prevNode.value)
        currNode = prevNode
    path.reverse()
    return path


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

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
