import math
import random
import numpy as np

# Board dimensions for Connect Four
ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    """
    Creates an empty Connect Four board (numpy 2D array).

    Returns:
    np.ndarray:
        A 2D numpy array of shape (ROW_COUNT, COLUMN_COUNT) filled with zeros (float).
    """
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, row, col, piece):
    """
    Places a piece (1 or 2) at the specified (row, col) position on the board.

    Parameters:
    board (np.ndarray): The current board, shape (ROW_COUNT, COLUMN_COUNT).
    row (int): The row index where the piece should be placed.
    col (int): The column index where the piece should be placed.
    piece (int): The piece identifier (1 or 2).
    
    Returns:
    None. The 'board' is modified in-place. Do NOT return a new board!
    """
    board[row][col] = piece


def is_valid_location(board, col):
    """
    Checks if dropping a piece in 'col' is valid (column not full).

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to check.

    Returns:
    bool: True if it's valid to drop a piece in this column, False otherwise.
    """
    for row in range(len(board)):
        if board[row][col] == 0:
            return True
    return False


def get_next_open_row(board, col):
    """
    Gets the next open row in the given column.

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to search.

    Returns:
    int: The row index of the lowest empty cell in this column.
    """
    row = -1
    for r in range(len(board) - 1, -1, -1):
        if board[r][col] == 0:
            row = r
    return row


def winning_move(board, piece):
    """
    Checks if the board state is a winning state for the given piece.

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    bool: True if 'piece' has a winning 4 in a row, False otherwise.
    This requires checking horizontally, vertically, and diagonally.
    """
    board_rows = len(board)
    board_cols = len(board[1])
    # Check horizontal locations
    for r in range(board_rows):
        for c in range(board_cols - 3):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations
    for r in range(board_rows - 3):
        for c in range(board_cols):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check diagonals
    for r in range(board_rows - 3):
        for c in range(board_cols - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    for r in range(board_rows - 3):
        for c in range(3, board_cols):
            if board[r][c] == piece and board[r + 1][c - 1] == piece and board[r + 2][c - 2] == piece and board[r + 3][c - 3] == piece:
                return True

    return False


def get_valid_locations(board):
    """
    Returns a list of columns that are valid to drop a piece.

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    list of int: The list of column indices that are not full.
    """
    valid_cols = []
    for col in range(len(board[0])):
        if is_valid_location(board, col):
            valid_cols.append(col)
    return valid_cols


def is_terminal_node(board):
    """
    Checks if the board is in a terminal state:
      - Either player has won
      - Or no valid moves remain

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    if (winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0):
        return True
    return False


def score_position(board, piece):
    """
    Evaluates the board for the given piece.
    (Already implemented to highlight center-column preference.)

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    int: Score favoring the center column. 
         (This can be extended with more advanced heuristics.)
    """
    # This is already done for you; no need to modify
    # The heuristic here scores the center column higher, which means
    # it prefers to play in the center column.
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Performs minimax with alpha-beta pruning to choose the best column.

    Parameters:
    board (np.ndarray): The current board.
    depth (int): Depth to which the minimax tree should be explored.
    alpha (float): Alpha for alpha-beta pruning.
    beta (float): Beta for alpha-beta pruning.
    maximizingPlayer (bool): Whether it's the maximizing player's turn.

    Returns:
    tuple:
        - (column (int or None), score (float)):
          column: The chosen column index (None if no moves).
          score: The heuristic score of the board state.
    """
    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, 2):
                return (None, math.inf)
            elif winning_move(board, 1):
                return (None, -math.inf)
            else:
                return (None, 0)  # No valid moves left, game over
        else:
            return (None, score_position(board, 2))  # Score board at max depth

    # AI's turn
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)
            # Calculate the score after dropping the piece
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            # If score if better than current value, update value and track column (maximizer)
            if new_score > value:
                value = new_score
                column = col
            # New alpha equal to max of current alpha and value
            alpha = max(alpha, value)
            # Pruning, no need to explore here
            # Minimizer will not allow this to happen
            if alpha >= beta:
                break
        return column, value

    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)
            # Calculate the score after dropping the piece
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            # If score if worse than current value, update value and track column (minimizer)
            if new_score < value:
                value = new_score
                column = col
            # New beta equal to min of current beta and
            beta = min(beta, value)
            # Pruning, no need to explore here
            # Maximizer will not choose this
            if beta <= alpha:
                break
        return column, value


if __name__ == "__main__":
    # Simple debug scenario
    # Example usage: create a board, drop some pieces, then call minimax
    example_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    print("Debug: Created an empty Connect Four board.\n")
    print(example_board)

    # TODO: Students can test their own logic here once implemented, e.g.:
    # drop_piece(example_board, some_row, some_col, 1)
    # col, score = minimax(example_board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    # print("Chosen column:", col, "Score:", score)
