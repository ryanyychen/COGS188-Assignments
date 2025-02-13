from tkinter import Frame, Label, CENTER
import random
import src.logic as logic
import src.constants as c
from src.mcts import MCTSNode, search

def gen() -> int:
    """
    Generate a random integer between 0 and GRID_LEN - 1.
    
    Returns:
        int: A random integer.
    """
    return random.randint(0, c.GRID_LEN - 1)

class GameGrid(Frame):
    """
    Class for managing the 2048 game interface using Tkinter.
    """
    def __init__(self):
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.score = 0
        self.score_label = Label(self, text=f"Score: {self.score}", bg=c.BACKGROUND_COLOR_GAME, font=c.FONT)
        self.score_label.grid(row=0, column=0, columnspan=c.GRID_LEN, sticky="e")

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.auto_play_enabled = False
        self.auto_mcts_enabled = False
        self.update_grid_cells()

        self.mainloop()

    def init_grid(self) -> None:
        """
        Initialize the grid cells for the game display.
        """
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()
        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(row=i, column=j, padx=c.GRID_PADDING, pady=c.GRID_PADDING)
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2
                )
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self) -> None:
        """
        Update the grid cells to reflect the current game board.
        """
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT.get(new_number, c.BACKGROUND_COLOR_CELL_EMPTY),
                        fg=c.CELL_COLOR_DICT.get(new_number, "#776e65")
                    )
        self.update_idletasks()
    
    def check_win_lose(self) -> None:
        """
        Check if the game has been won or lost and update the grid cells with a message.
        """
        state = logic.game_state(self.matrix)
        if state == 'win':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][0].configure(text="Press", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][1].configure(text="\'n\'", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][2].configure(text="to", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][3].configure(text="Restart", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        elif state == 'lose':
            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][0].configure(text="Press", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][1].configure(text="\'n\'", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][2].configure(text="to", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][3].configure(text="Restart", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def key_down(self, event) -> None:
        """
        Handle key press events for game control.
        
        Args:
            event: The key press event.
        """
        key = event.keysym
        if key == c.KEY_QUIT: 
            exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
        elif key.lower() == 'a':
            if not self.auto_play_enabled:
                self.auto_play_enabled = True
                self.start_random()
        elif key.lower() == 's':
            self.auto_play_enabled = False
            self.auto_mcts_enabled = False
        elif key.lower() == 'm':
            if not self.auto_mcts_enabled:
                self.auto_mcts_enabled = True
                self.start_mcts()
        elif key in self.commands:
            self.matrix, done, reward = self.commands[key](self.matrix)
            if done:
                self.score += reward
                self.score_label.configure(text=f"Score: {self.score}")
                self.matrix = logic.add_two(self.matrix)
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                self.check_win_lose()
        elif key == 'n':
            self.matrix = logic.new_game(c.GRID_LEN)
            self.auto_play_enabled = False
            self.auto_mcts_enabled = False
            self.score = 0
            self.score_label.configure(text=f"Score: {self.score}")
            self.history_matrixs = []
            self.update_grid_cells()

    def auto_play(self) -> None:
        """
        Automatically play the game using random moves.
        """
        if not self.auto_play_enabled:
            return
        move = random.choice(list(self.commands.keys()))
        new_board, moved, reward = self.commands[move](self.matrix)
        if moved:
            self.score += reward
            self.matrix = logic.add_two(new_board)
            self.history_matrixs.append(self.matrix)
            self.score_label.configure(text=f"Score: {self.score}")
            self.update_grid_cells()
            self.check_win_lose()
        if logic.game_state(self.matrix) in ("win", "lose"):
            return
        self.after(10, self.auto_play)

    def start_random(self) -> None:
        """
        Start automatic play with random moves.
        """
        self.auto_play()
    
    def mcts(self) -> None:
        """
        Automatically play the game using Monte Carlo Tree Search.
        """
        if not self.auto_mcts_enabled:
            return
        root = MCTSNode(self.matrix, self.score)
        best_move = search(root, iterations=1000)
        new_board, moved, reward = self.commands[best_move](self.matrix)
        if moved:
            self.score += reward
            self.matrix = logic.add_two(new_board)
            self.history_matrixs.append(self.matrix)
            self.score_label.configure(text=f"Score: {self.score}")
            self.update_grid_cells()
            self.check_win_lose()
        if logic.game_state(self.matrix) in ("win", "lose"):
            return
        self.after(10, self.mcts)
    
    def start_mcts(self) -> None:
        """
        Start automatic play using Monte Carlo Tree Search.
        """
        self.mcts()

    def generate_next(self) -> None:
        """
        Generate the next '2' tile in a random empty cell.
        """
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

if __name__ == "__main__":
    game_grid = GameGrid()
