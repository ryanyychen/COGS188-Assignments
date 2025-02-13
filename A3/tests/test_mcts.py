import unittest
import copy
import random
import src.constants as c
import src.logic as logic
from src.mcts import MCTSNode, rollout, backpropagate, search

class TestMCTS(unittest.TestCase):
    def setUp(self):
        # Create an initial game board using logic.new_game
        self.initial_board = logic.new_game(c.GRID_LEN)
        self.root = MCTSNode(self.initial_board)

    def test_apply_move(self):
        """
        Test that applying a move returns a valid board, a boolean for changed state,
        and an integer reward.
        """
        moves = [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]
        for move in moves:
            new_board, moved, reward = self.root.apply_move(self.initial_board, move)
            # new_board should be a 2D list with the same dimensions as initial_board.
            self.assertIsInstance(new_board, list)
            self.assertTrue(len(new_board) == c.GRID_LEN)
            self.assertTrue(all(isinstance(row, list) for row in new_board))
            self.assertIsInstance(moved, bool)
            self.assertIsInstance(reward, int)

    def test_rollout(self):
        """
        Test that a rollout from an initial board returns a valid result (1, 0, or -1).
        """
        result = rollout(copy.deepcopy(self.initial_board))
        self.assertIn(result, [1, 0, -1])

    def test_backpropagation(self):
        """
        Test that backpropagation properly updates the visits and value for a node
        and its parent.
        """
        # Create a child node manually and add it to root.
        child = MCTSNode(copy.deepcopy(self.initial_board), score=0, parent=self.root)
        self.root.children[c.KEY_UP] = child

        original_root_visits = self.root.visits
        original_child_visits = child.visits

        # Backpropagate a rollout result of 1 from the child.
        backpropagate(child, 1)

        # Check that the child's visits and value are updated.
        self.assertEqual(child.visits, original_child_visits + 1)
        self.assertEqual(child.value, 1)

        # Check that the parent's visits and value are updated.
        self.assertEqual(self.root.visits, original_root_visits + 1)
        self.assertEqual(self.root.value, 1)

    def test_best_child(self):
        """
        Test that best_child returns a tuple (move, child) where move is a valid key and
        the returned child matches the one stored in the node's children.
        """
        # Manually create two children with different (artificial) visit and value counts.
        child1 = MCTSNode(copy.deepcopy(self.initial_board), score=0, parent=self.root)
        child1.visits = 10
        child1.value = 20

        child2 = MCTSNode(copy.deepcopy(self.initial_board), score=0, parent=self.root)
        child2.visits = 5
        child2.value = 15

        self.root.children[c.KEY_UP] = child1
        self.root.children[c.KEY_DOWN] = child2
        self.root.visits = 15  # Set parent's visits for UCT calculation

        best_move, best_child = self.root.best_child(c_param=1.4)
        self.assertIn(best_move, [c.KEY_UP, c.KEY_DOWN])
        self.assertEqual(best_child, self.root.children[best_move])

    def test_search(self):
        """
        Test that search returns a valid move (one of the allowed keys).
        """
        best_move = search(self.root, iterations=10)
        self.assertIn(best_move, [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT])

if __name__ == '__main__':
    unittest.main()
