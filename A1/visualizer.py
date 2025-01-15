import pygame
import numpy as np
import argparse
import random

# Import the graph-based solver methods
# Make sure your solver.py has these functions:
#   parse_maze_to_graph(maze),
#   bfs, dfs, astar, bidirectional_search, and simulated_annealing
from solver import (
    parse_maze_to_graph,
    bfs,
    dfs,
    astar,
    bidirectional_search,
    simulated_annealing
)

# Constants
SCREEN_SIZE = 600
GRID_SIZE = 25
BLOCK_SIZE = SCREEN_SIZE // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Maze Solver")

# Maze generation (0 = open, 1 = wall)
maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2])
maze[0][0] = 0  # Start must be open
maze[GRID_SIZE - 1][GRID_SIZE - 1] = 0  # Goal must be open

def draw_maze(maze):
    """
    Draws the maze onto the screen.
    White = open cell, Black = wall, Green = start, Red = goal.
    """
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            color = WHITE if maze[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)

    # Start cell in green
    start_rect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, GREEN, start_rect)
    # Goal cell in red
    goal_rect = pygame.Rect((GRID_SIZE - 1) * BLOCK_SIZE, (GRID_SIZE - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, RED, goal_rect)

def draw_path(path):
    """
    Draws the solution path in blue, and re-draws start (green) and goal (red).
    'path' is expected to be a list of (row, col) tuples.
    """
    for (row, col) in path:
        rect = pygame.Rect(col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, BLUE, rect)

    # Re-draw start and goal
    start_rect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, GREEN, start_rect)
    goal_rect = pygame.Rect((GRID_SIZE - 1) * BLOCK_SIZE, (GRID_SIZE - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, RED, goal_rect)

def solve_maze(maze, algorithm):
    """
    Converts the 2D maze into a graph, then runs the chosen algorithm.
    Returns a list of (row, col) coordinates if a path is found, else None.
    """
    # Parse the NumPy array 'maze' into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze)

    # If start_node or goal_node is None, there's no valid path
    if start_node is None or goal_node is None:
        return None

    # Select and run the algorithm
    if algorithm == "simulated_annealing":
        path = simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01)
    elif algorithm == "astar":
        path = astar(start_node, goal_node)
    elif algorithm == "bfs":
        path = bfs(start_node, goal_node)
    elif algorithm == "dfs":
        path = dfs(start_node, goal_node)
    elif algorithm == "bidirectional":
        path = bidirectional_search(start_node, goal_node)
    else:
        path = None

    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "algorithm",
        help="Algorithm to solve the maze",
        choices=["simulated_annealing", "astar", "bfs", "dfs", "bidirectional"]
    )
    args = parser.parse_args()

    print(f"Using {args.algorithm.capitalize()}")

    running = True
    maze_solved = False
    path = None

    # Main Pygame loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Check if user clicked "New Map" or "Try Again"
            if event.type == pygame.MOUSEBUTTONDOWN:
                # "New Map" button region
                new_map_rect = pygame.Rect(SCREEN_SIZE - 150, 0, 150, 50)
                if new_map_rect.collidepoint(event.pos):
                    # Generate a new random maze
                    maze_solved = False
                    path = None
                    maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2])
                    maze[0][0] = 0
                    maze[GRID_SIZE - 1][GRID_SIZE - 1] = 0

                # "Try Again" button region (only for Simulated Annealing)
                if args.algorithm == "simulated_annealing":
                    try_again_rect = pygame.Rect(SCREEN_SIZE - 150, 60, 150, 50)
                    if try_again_rect.collidepoint(event.pos):
                        maze_solved = False
                        path = None

        # Drawing / Updating
        screen.fill(BLACK)
        draw_maze(maze)

        # If not solved yet, run the solver
        if not maze_solved:
            path = solve_maze(maze, args.algorithm)
            if path:
                maze_solved = True

        # If we have a solved path, draw it
        if maze_solved and path:
            draw_path(path)

        # Draw "New Map" button
        new_map_rect = pygame.Rect(SCREEN_SIZE - 150, 0, 150, 50)
        pygame.draw.rect(screen, RED, new_map_rect)
        font = pygame.font.Font(None, 36)
        text = font.render("New Map", True, WHITE)
        text_rect = text.get_rect(center=new_map_rect.center)
        screen.blit(text, text_rect)

        # Draw "Try Again" button if we're using simulated_annealing
        if args.algorithm == "simulated_annealing":
            try_again_rect = pygame.Rect(SCREEN_SIZE - 150, 60, 150, 50)
            pygame.draw.rect(screen, BLUE, try_again_rect)
            text = font.render("Try Again", True, WHITE)
            text_rect = text.get_rect(center=try_again_rect.center)
            screen.blit(text, text_rect)

        pygame.display.flip()

    pygame.quit()
