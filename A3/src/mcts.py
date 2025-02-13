import copy
import random
import src.logic as logic
import src.constants as c

class MCTSNode:
    """
    Monte Carlo Tree Search Node representing a game state.

    Attributes:
        matrix (list): The current game board (a 2D list).
        score (int): The current score of the game.
        parent (MCTSNode): The parent node in the MCTS tree.
        children (dict): Dictionary mapping moves (keys) to child nodes (MCTSNode).
        visits (int): The number of times this node has been visited.
        value (float): The total accumulated rollout value from this node.
    """
    def __init__(self, matrix: list, score: int = 0, parent: 'MCTSNode' = None):
        self.matrix = copy.deepcopy(matrix)
        self.score = score
        self.parent = parent
        self.children = {}  # Keys: move (e.g., c.KEY_UP), Values: MCTSNode instance
        self.visits = 0
        self.value = 0

    def is_terminal(self) -> bool:
        """
        Check if the game state in this node is terminal (i.e., either a win or a loss).

        Returns:
            bool: True if the state is terminal; False otherwise.
        """
        return logic.game_state(self.matrix) in ('win', 'lose')

    def expand(self) -> None:
        """
        Expand the current node by generating all valid moves.

        For each move (up, down, left, right), if applying that move results in a change 
        in the board (i.e., a valid move), then create a new child node representing the 
        new game state after the move. The child node is stored in self.children with the 
        move as the key.

        Returns:
            None
        """
        moves = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        for move in moves:
            new_matrix, moved, reward = self.apply_move(self.matrix, move)
            if moved:
                child_node = MCTSNode(new_matrix, self.score + reward, parent=self)
                self.children[move] = child_node

    def best_child(self, c_param: float = 1.4) -> tuple:
        """
        Select the best child node based on the Upper Confidence Bound for Trees (UCT) metric.

        The UCT value for a child node is computed as:
        
            UCT = (child.value / (child.visits + 1e-5)) + 
                  c_param * sqrt( (2 * log(self.visits + 1e-5)) / (child.visits + 1e-5) )
        
        Args:
            c_param (float): Exploration parameter (default is 1.4).
        
        Returns:
            tuple: A tuple (move, child_node) corresponding to the child with the highest UCT value.
        
        TODO:
          - Implement the UCT calculation for each child.
          - Return the move and the corresponding child node with the highest UCT.
        """
        # TODO: Implement best_child function.
        pass

    def apply_move(self, matrix: list, move: str) -> tuple:
        """
        Apply the specified move to the game board.

        Args:
            matrix (list): The current game board.
            move (str): The move to apply (one of c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, or c.KEY_RIGHT).
        
        Returns:
            tuple: (new_matrix, moved, reward) where:
                - new_matrix (list): The game board after applying the move.
                - moved (bool): True if the move changes the board; otherwise False.
                - reward (int): The reward obtained from the move.
        """
        if move == c.KEY_UP:
            return logic.up(matrix)
        elif move == c.KEY_DOWN:
            return logic.down(matrix)
        elif move == c.KEY_LEFT:
            return logic.left(matrix)
        elif move == c.KEY_RIGHT:
            return logic.right(matrix)
        else:
            # Fallback (should not happen)
            return logic.left(matrix)

def rollout(matrix: list) -> int:
    """
    Perform a random rollout (simulation) starting from the given board state.

    At each step, randomly choose one of the possible moves (up, down, left, right) and apply it.
    Continue the simulation until a terminal state (win or lose) is reached. Return the result of 
    the rollout: 1 if the final state is a win, -1 if it is a loss, and 0 otherwise.

    Args:
        matrix (list): The starting game board.
    
    Returns:
        int: The outcome of the rollout (1 for win, -1 for lose, 0 otherwise).
    
    TODO:
      - Implement the random simulation loop using the move functions from logic.
      - Use logic.add_two to add a new '2' tile after a valid move.
      - Break the loop if a move does not change the board.
    """
    # TODO: Implement rollout simulation.
    pass

def backpropagate(node: MCTSNode, result: int) -> None:
    """
    Propagate the result of a rollout up the MCTS tree.

    Starting from the provided node, update the visit count and accumulated value of that node 
    and all its ancestor nodes by adding the rollout result.

    Args:
        node (MCTSNode): The node from which to start backpropagation.
        result (int): The result of the rollout (1, -1, or 0).
    
    Returns:
        None
    
    TODO:
      - Update the node's visits and value.
      - Continue updating until reaching the root node.
    """
    # TODO: Implement backpropagation.
    pass

def search(root: MCTSNode, iterations: int = 100) -> str:
    """
    Perform the Monte Carlo Tree Search starting from the root node.

    For a specified number of iterations, perform the following steps:
      1. **Selection:** Starting at the root, traverse the tree by repeatedly selecting 
         the best child (using best_child) until a node is reached that is terminal or not fully expanded.
      2. **Expansion:** If the node is not terminal, expand it (using expand) to generate its children.
      3. **Rollout:** From one of the new children, perform a random rollout (using rollout) to obtain a result.
      4. **Backpropagation:** Propagate the rollout result up the tree (using backpropagate).
    
    After all iterations are completed, return the move from the root that is deemed best (using best_child with c_param=0).

    Args:
        root (MCTSNode): The starting node (root of the search tree).
        iterations (int): The number of iterations to run the search.
    
    Returns:
        str: The best move (one of c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT) determined by the search.
    
    TODO:
      - Implement the full MCTS loop following the steps above.
    """
    # TODO: Implement the search function.
    pass

if __name__ == "__main__":
    # Example usage:
    # Initialize the game board using logic.new_game.
    initial_matrix = logic.new_game(c.GRID_LEN)
    root_node = MCTSNode(initial_matrix)
    
    # Run the MCTS search for a fixed number of iterations.
    best_move = search(root_node, iterations=50)
    print("Recommended move:", best_move)