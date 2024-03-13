# # Make sure all necessary imports are included
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import matplotlib.pyplot as plt
# import random

# from tqdm.notebook import tqdm
# import ipywidgets as widgets
# from IPython.display import display, clear_output

# from games import TicTacToe
# from SimplePolicyNetwork import SimplePolicyNetwork, makeValidMove

# from games import TicTacToe
# from SimplePolicyNetwork import SimplePolicyNetwork, makeValidMove

# import unittest


# board_size = (3, 3)  # For Tic Tac Toe
# num_moves = board_size[0] * board_size[1]

# class TestGameOutcome(unittest.TestCase):
#     def test_game_play(self):
#         game = TicTacToe()
#         model = SimplePolicyNetwork(board_size, num_moves)

#         while not game.is_game_over():
#             makeValidMove(game, model)

#         if game.winner:
#             self.assertIsNotNone(game.winner, "Expected a winner but got None.")
#         else:
#             self.assertTrue(game.is_draw(), "Expected a draw but game did not end as such.")







# if __name__ == '__main__':
#     unittest.main()

# import unittest
# import numpy as np
# from games import TicTacToe  

# class TestTicTacToe(unittest.TestCase):
#     def setUp(self):
#         self.game = TicTacToe()

#     def test_initialization(self):
#         expected_board = np.zeros((3, 3), dtype=int)
#         self.assertTrue(np.array_equal(self.game.board, expected_board))
#         self.assertEqual(self.game.turn, 1)
#         self.assertEqual(self.game.winner, 0)

#     def test_is_full(self):
#         self.assertFalse(self.game.is_full())
#         self.game.board = np.ones((3, 3), dtype=int)
#         self.assertTrue(self.game.is_full())

#     def test_is_winner(self):
#         # Test rows and columns
#         for player in [1, 2]:
#             for i in range(3):
#                 self.game.board = np.zeros((3, 3), dtype=int)
#                 self.game.board[i] = player
#                 self.assertTrue(self.game.is_winner(player))
#                 self.game.board = np.zeros((3, 3), dtype=int)
#                 self.game.board[:, i] = player
#                 self.assertTrue(self.game.is_winner(player))
#         # Test diagonals
#         self.game.board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         self.assertTrue(self.game.is_winner(1))
#         self.game.board = np.array([[0, 0, 2], [0, 2, 0], [2, 0, 0]])
#         self.assertTrue(self.game.is_winner(2))

#     def test_is_game_over(self):
#         self.game.board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]])
#         self.assertTrue(self.game.is_game_over())
#         self.game.reset()
#         self.game.board = np.array([[1, 2, 1], [2, 1, 2], [0, 1, 2]])
#         self.assertFalse(self.game.is_game_over())

#     def test_get_valid_moves(self):
#         self.game.board[0, 0] = 1
#         self.game.board[1, 1] = 2
#         valid_moves = self.game.get_valid_moves()
#         expected_moves = np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]])
#         self.assertTrue(np.array_equal(valid_moves, expected_moves))

#     def test_make_and_undo_move(self):
#         move = (0, 0)
#         self.game.make_move(move)
#         self.assertEqual(self.game.board[0, 0], 1)
#         self.game.undo_move(move)
#         self.assertEqual(self.game.board[0, 0], 0)

#     def test_reset(self):
#         self.game.board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]])
#         self.game.reset()
#         expected_board = np.zeros((3, 3), dtype=int)
#         self.assertTrue(np.array_equal(self.game.board, expected_board))

# if __name__ == '__main__':
#     unittest.main()


import unittest
import torch
from games import TicTacToe

class TestTicTacToe(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()

    def test_initialization(self):
        expected_board = torch.zeros(3, 3, dtype=torch.int)
        self.assertTrue(torch.equal(self.game.board, expected_board))
        self.assertEqual(self.game.turn, 1)
        self.assertEqual(self.game.winner, 0)

    def test_is_full(self):
        self.assertFalse(self.game.is_full())
        self.game.board = torch.ones(3, 3, dtype=torch.int)
        self.assertTrue(self.game.is_full())

    def test_is_winner(self):
        # Test rows and columns
        for player in [1, 2]:
            for i in range(3):
                self.game.board = torch.zeros(3, 3, dtype=torch.int)
                self.game.board[i] = player
                self.assertTrue(self.game.is_winner(player))
                self.game.board = torch.zeros(3, 3, dtype=torch.int)
                self.game.board[:, i] = player
                self.assertTrue(self.game.is_winner(player))
        # Test diagonals
        self.game.board = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int)
        self.assertTrue(self.game.is_winner(1))
        self.game.board = torch.tensor([[0, 0, 2], [0, 2, 0], [2, 0, 0]], dtype=torch.int)
        self.assertTrue(self.game.is_winner(2))

    def test_is_game_over(self):
        self.game.board = torch.tensor([[1, 2, 1], [2, 1, 2], [2, 1, 2]], dtype=torch.int)
        self.assertTrue(self.game.is_game_over())
        self.game.reset()
        self.game.board = torch.tensor([[1, 2, 1], [2, 1, 2], [0, 1, 2]], dtype=torch.int)
        self.assertFalse(self.game.is_game_over())

    def test_get_valid_moves(self):
        self.game.board[0, 0] = 1
        self.game.board[1, 1] = 2
        valid_moves = self.game.get_valid_moves()
        expected_moves = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]], dtype=torch.int)
        # For comparing lists of coordinates, might need to convert to sets of tuples due to ordering and uniqueness
        self.assertTrue(set(map(tuple, valid_moves.numpy())) == set(map(tuple, expected_moves.numpy())))

    def test_make_and_undo_move(self):
        move = (0, 0)
        self.game.make_move(move)
        self.assertEqual(self.game.board[0, 0].item(), 1)
        self.game.undo_move(move)
        self.assertEqual(self.game.board[0, 0].item(), 0)

    def test_reset(self):
        self.game.board = torch.tensor([[1, 2, 1], [2, 1, 2], [2, 1, 2]], dtype=torch.int)
        self.game.reset()
        expected_board = torch.zeros(3, 3, dtype=torch.int)
        self.assertTrue(torch.equal(self.game.board, expected_board))

if __name__ == '__main__':
    unittest.main()

