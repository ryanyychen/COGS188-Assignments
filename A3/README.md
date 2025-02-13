# COGS 188 Assignment 3

This assignment has two parts:

1. Implementing Monte Carlo Tree Search on 2048
2. Implementing Dynamic Programming on the Door and Key Problem

## Part 1: Trying Monte Carlo Tree Search on 2048 (5 points)

In this assignment, you will implement the Monte Carlo Tree Search (MCTS) algorithm for the 2048 game. Your focus is to show your understanding of MCTS by completing the missing parts in the provided **mcts.py** file. The other modules (**constants.py**, **logic.py**, and **puzzle.py**) are fully implemented and will be used by your MCTS code for simulation and visualization.

Before getting started, we recommend reading through the other files, especially **puzzle.py** and **logic.py**, to understand the game mechanics and the provided functions. The **constants.py** file contains the game's constants, such as the board size and the probability of generating a 2 or 4 tile.

If you are unfamiliar with the 2048 game, you can play it by running the following command:

```bash
python3 puzzle.py
```

This will open up a GUI window where you can play the game using the arrow keys. The game is played on a 4x4 grid, where the player can move the tiles in four directions: up, down, left, and right (you can use the arrow keys for these moves).

The player can only move the tiles in one direction per turn, and all the tiles move in that direction until they hit the wall or another tile. When two tiles with the same number collide, they merge into a single tile with a value equal to the sum of the two tiles. The goal of the game is to reach the 2048 tile by merging the tiles. It's not very easy to reach the 2048 tile!

### Task

Let's try to use the Monte Carlo Tree Search (MCTS) algorithm to play the 2048 game. MCTS is a heuristic search algorithm that uses random simulations to find the best move in a game.  Your assignment is to fill in the missing parts (marked with **TODO**) in **mcts.py**. Specifically, you need to implement the following functions:

1. **MCTSNode.best_child(c_param: float = 1.4) -> tuple**  
   - **Input:** The current node's children and an exploration parameter.
   - **Output:** A tuple `(move, child_node)` representing the child with the highest UCT value.
   - **Task:** Calculate the UCT value for each child and return the best one.

2. **rollout(matrix: list) -> int**  
   - **Input:** A starting game board (2D list).
   - **Output:** An integer result (1 if win, -1 if lose, or 0 otherwise) from a random simulation.
   - **Task:** Perform a random simulation until a terminal state is reached, using the functions from **logic.py**.

3. **backpropagate(node: MCTSNode, result: int) -> None**  
   - **Input:** A node and the result of a rollout.
   - **Output:** None (updates the node and its ancestors).
   - **Task:** Update the visits and value for the node and all its parent nodes with the rollout result.

4. **search(root: MCTSNode, iterations: int = 100) -> str**  
   - **Input:** The root node of the search tree and the number of iterations to perform.
   - **Output:** A string representing the best move (one of `c.KEY_UP`, `c.KEY_DOWN`, `c.KEY_LEFT`, or `c.KEY_RIGHT`).
   - **Task:** Implement the full MCTS loop:
     - **Selection:** Traverse the tree from the root to a leaf.
     - **Expansion:** Expand the leaf if it is non-terminal.
     - **Rollout:** Perform a random simulation from the new node.
     - **Backpropagation:** Propagate the result up the tree.
     - Finally, select and return the best move from the root.

### How to Run

To test your implementation, first run the provided **puzzle.py** file.
  
```bash
python3 puzzle.py
```

Then, instead of manually playing the game, press the `m` key on your keyboard to let the MCTS algorithm play the game. The algorithm will play the game using the MCTS algorithm. You can also adjust the number of iterations in the **mcts.py** file to see how the algorithm performs with more or fewer iterations.

You can also try out playing the game with a few other options:

* `a`: Play the game with random moves (this would not work well, but is a good baseline).
* `m`: Play the game with MCTS.
* `s`: Play the game manually (using the arrow keys).
* `n`: Start a new game (creates a new board).

### Evaluation

It is pretty hard to get the 2048 tile using MCTS (why do you think that is?). We recommend opening `puzzle.py` and modifying the number of iterations used for MCTS to play around with the performance. Note that using a larger number of iterations will increase the time taken to make a move but may improve the performance of the MCTS algorithm.

## Part 2: MuJoCo simulation of Cartpole (5 pts)

Please view the instruction of how to complete this task in `cartpole_td.ipynb` and submit files to Gradescope following those instructions.