import random
import src.constants as c

def new_game(n: int) -> list:
    """
    Create a new game board with size n x n and add two '2' tiles.
    
    Args:
        n (int): Size of the board (number of rows/columns).
    
    Returns:
        list: A 2D list representing the game board.
    """
    # Create an n x n matrix initialized with zeros.
    board = [[0 for _ in range(n)] for _ in range(n)]
    # Add two '2' tiles on the board.
    board = add_two(board)
    board = add_two(board)
    return board

def add_two(mat: list) -> list:
    """
    Add a tile with value 2 in a random empty cell on the board.
    
    Args:
        mat (list): 2D list representing the game board.
    
    Returns:
        list: Updated board after adding a new '2' tile.
    """
    n = len(mat)
    # Continue choosing a random position until an empty cell (0) is found.
    while True:
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if mat[a][b] == 0:
            mat[a][b] = 2
            break
    return mat

def game_state(mat: list) -> str:
    """
    Check the current state of the game.
    
    Args:
        mat (list): 2D list representing the game board.
    
    Returns:
        str: 'win' if any tile is 2048, 'lose' if no moves are possible,
             or 'not over' if the game can continue.
    """
    rows = len(mat)
    cols = len(mat[0])
    
    # Check for win (tile 2048 exists)
    for i in range(rows):
        for j in range(cols):
            if mat[i][j] == 2048:
                return 'win'
    
    # Check for any empty cell
    for i in range(rows):
        for j in range(cols):
            if mat[i][j] == 0:
                return 'not over'
    
    # Check for adjacent cells that are equal (horizontal adjacent)
    for i in range(rows):
        for j in range(cols - 1):
            if mat[i][j] == mat[i][j+1]:
                return 'not over'
    
    # Check for adjacent cells that are equal (vertical adjacent)
    for j in range(cols):
        for i in range(rows - 1):
            if mat[i][j] == mat[i+1][j]:
                return 'not over'
    
    # If no empty cells or mergeable cells, the game is lost.
    return 'lose'

def merge(mat: list, done: bool) -> tuple:
    """
    Merge adjacent identical tiles in each row and calculate the reward.
    
    Args:
        mat (list): 2D list representing the game board (after compressing).
        done (bool): Flag indicating if a merge has already occurred.
    
    Returns:
        tuple: A tuple (new_mat, done, reward) where:
            - new_mat is the updated board,
            - done is True if any merge happened,
            - reward is the value of the merged tile.
    """
    reward = 0
    rows = len(mat)
    cols = len(mat[0])
    for i in range(rows):
        for j in range(cols - 1):
            # If two adjacent cells are equal and not zero, merge them.
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2          # Double the left cell's value.
                mat[i][j+1] = 0         # Set the right cell to 0.
                reward += mat[i][j]     # Add merged value to the reward.
                done = True             # Mark that a merge has occurred.
    return mat, done, reward

def up(game: list) -> tuple:
    """
    Shift the board upward.
    
    Uses helper functions (transpose, cover_up, merge) to perform the move.
    
    Args:
        game (list): 2D list representing the game board.
    
    Returns:
        tuple: A tuple (new_game, done, reward) after shifting up.
    """
    # 1. Transpose the board.
    transposed = transpose(game)
    # 2. Use cover_up() to shift numbers to the left.
    new_game, done = cover_up(transposed)
    # 3. Merge the tiles.
    new_game, done, reward = merge(new_game, done)
    # 4. Use cover_up() again to shift the board.
    new_game, _ = cover_up(new_game)
    # 5. Transpose the board back to its original orientation.
    final_game = transpose(new_game)
    return final_game, done, reward

# ====================================
# HELPER FUNCTIONS (provided for you)
# ====================================

def reverse(mat: list) -> list:
    """
    Reverse each row of the board.
    
    Args:
        mat (list): 2D list representing the game board.
    
    Returns:
        list: A new 2D list with each row reversed.
    """
    new = []
    for i in range(len(mat)):
        new.append(mat[i][::-1])
    return new

def transpose(mat: list) -> list:
    """
    Transpose the board (swap rows and columns).
    
    Args:
        mat (list): 2D list representing the game board.
    
    Returns:
        list: The transposed board.
    """
    return [list(row) for row in zip(*mat)]

def cover_up(mat: list) -> tuple:
    """
    Shift all non-zero elements of each row to the left.
    
    Args:
        mat (list): 2D list representing the game board.
    
    Returns:
        tuple: A tuple (new_mat, done) where new_mat is the updated board after shifting,
               and done is True if any shift occurred.
    """
    new = [[0] * c.GRID_LEN for _ in range(c.GRID_LEN)]
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done

def down(game: list) -> tuple:
    """
    Shift the board downward.
    
    Args:
        game (list): 2D list representing the game board.
    
    Returns:
        tuple: A tuple (new_game, done, reward) after shifting down.
    """
    game = reverse(transpose(game))
    new_game, done = cover_up(game)
    new_game, done, reward = merge(new_game, done)
    new_game = cover_up(new_game)[0]
    new_game = transpose(reverse(new_game))
    return new_game, done, reward

def left(game: list) -> tuple:
    """
    Shift the board to the left.
    
    Args:
        game (list): 2D list representing the game board.
    
    Returns:
        tuple: A tuple (new_game, done, reward) after shifting left.
    """
    new_game, done = cover_up(game)
    new_game, done, reward = merge(new_game, done)
    new_game = cover_up(new_game)[0]
    return new_game, done, reward

def right(game: list) -> tuple:
    """
    Shift the board to the right.
    
    Args:
        game (list): 2D list representing the game board.
    
    Returns:
        tuple: A tuple (new_game, done, reward) after shifting right.
    """
    game = reverse(game)
    new_game, done = cover_up(game)
    new_game, done, reward = merge(new_game, done)
    new_game = cover_up(new_game)[0]
    new_game = reverse(new_game)
    return new_game, done, reward
