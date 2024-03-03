import unittest
import numpy as np
from games import UltimateTicTacToe  


class TestUltimateTicTacToe(unittest.TestCase):
    def setUp(self):
        self.game = UltimateTicTacToe()

    def test_initial_state(self):
        # Test that the game initializes correctly
        self.assertEqual(self.game.turn, 1, "Initial turn should be 1")
        self.assertEqual(self.game.winner, 0, "Initial winner should be 0")
        self.assertIsNone(self.game.last_move, "Initial last move should be None")
        for row in self.game.boards:
            for board in row:
                self.assertTrue(np.array_equal(board.board, np.zeros((3, 3))), "Each TicTacToe board should be initialized with zeros")

    def test_make_move(self):
        # Test making a move on an empty board
        self.game.make_move(((0, 0), (1, 1)))
        self.assertEqual(self.game.boards[0, 0].board[1, 1], 1, "The move should be made by player 1")
        self.assertEqual(self.game.turn, 2, "The turn should switch to player 2")

    def test_invalid_move(self):
        # Test making an invalid move (on a non-empty cell)
        self.game.make_move(((0, 0), (1, 1)))  # First move by player 1
        self.game.make_move(((0, 0), (1, 1)))  # Try to make the same move
        self.assertEqual(self.game.boards[0, 0].board[1, 1], 1, "The cell should still belong to player 1")
        self.assertEqual(self.game.turn, 2, "The turn should still be player 2's")

    def test_game_over_by_win(self):
        # Test game over by winning a board and then the main board
        moves = [((0, 0), (0, 0)), ((1, 1), (0, 0)), ((0, 0), (1, 1)), ((1, 1), (1, 1)), ((0, 0), (2, 2))]
        for move in moves:
            self.game.make_move(move)
        self.assertTrue(self.game.boards[0, 0].is_game_over(), "The small board should be won")
        self.assertEqual(self.game.main_board[0, 0], 1, "The main board should reflect player 1's win")
        self.assertEqual(self.game.winner, 0, "The overall game should not have a winner yet")

    def test_reset(self):
        # Test resetting the game
        self.game.make_move(((0, 0), (1, 1)))  # Make a move
        self.game.reset()
        self.assertEqual(self.game.turn, 1, "After reset, turn should be 1")
        self.assertEqual(self.game.winner, 0, "After reset, winner should be 0")
        self.assertIsNone(self.game.last_move, "After reset, last move should be None")
        for row in self.game.boards:
            for board in row:
                self.assertTrue(np.array_equal(board.board, np.zeros((3, 3))), "After reset, each TicTacToe board should be initialized with zeros")

    def test_overall_game_win(self):
        # Player 1 wins top-left small board (0,0)
        moves = [
            ((0, 0), (0, 0)), ((1, 0), (0, 0)),  # Player 1, Player 2
            ((0, 0), (1, 1)), ((1, 0), (1, 1)),  # Player 1, Player 2
            ((0, 0), (2, 2)), ((1, 0), (1, 0)),  # Player 1, Player 2
        ]
        
        # Player 1 wins top-center small board (0,1)
        moves += [
            ((0, 1), (0, 0)), ((1, 1), (0, 0)),  # Player 1, Player 2
            ((0, 1), (1, 1)), ((1, 1), (1, 1)),  # Player 1, Player 2
            ((0, 1), (2, 2)), ((1, 0), (0, 1)),  # Player 1, Player 2
        ]

        # Player 1 wins top-right small board (0,2)
        moves += [
            ((0, 2), (0, 0)), ((1, 2), (0, 0)),  # Player 1, Player 2
            ((0, 2), (1, 1)), ((1, 2), (1, 1)),  # Player 1, Player 2
            ((0, 2), (2, 2)), ((1, 0), (1, 1)),  # Player 1, Player 2
        ]

        # Execute the moves
        for move in moves:
            self.game.make_move(move)

        # Check if the game recognizes Player 1 as the overall winner
        self.assertTrue(self.game.is_game_over(), "The game should be over.")
        self.assertEqual(self.game.winner, 1, "Player 1 should be the winner of the overall game.")

    def test_initial_valid_moves(self):
        valid_moves = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 81, "Initially, all 81 moves should be valid")
        for move in valid_moves:
            self.assertIsInstance(move, tuple, "Each move should be a tuple")
            self.assertEqual(len(move), 2, "Each move should have two elements")
            self.assertIsInstance(move[0], tuple, "The first element of each move should be a tuple")
            self.assertIsInstance(move[1], tuple, "The second element of each move should be a tuple")
            self.assertEqual(len(move[0]), 2, "The board position should be a tuple of two integers")
            self.assertEqual(len(move[1]), 2, "The move position should be a tuple of two integers")

    def test_valid_moves_after_specific_move(self):
        # Make a specific move and check the valid moves are in the correct small board
        self.game.make_move(((0, 0), (1, 1)))  # Player 1 makes a move in the center of the top-left small board
        valid_moves = self.game.get_valid_moves()
        expected_moves = [((1, 1), (i, j)) for i in range(3) for j in range(3)]
        self.assertTrue(all(move in valid_moves for move in expected_moves), "Valid moves should be directed to the (1, 1) small board")

    def test_valid_moves_when_small_board_won(self):
        # Simulate a small board being won and check valid moves are in other boards
        self.game.boards[0, 0].winner = 1  # Top-left small board is won by Player 1
        self.game.last_move = ((0, 0), (2, 2))  # Last move directs to the won board
        valid_moves = self.game.get_valid_moves()
        # Exclude the won board from expected moves
        expected_moves = [((i, j), (k, l)) for i in range(3) for j in range(3) if (i, j) != (0, 0) for k in range(3) for l in range(3)]
        self.assertTrue(all(move in valid_moves for move in expected_moves), "Valid moves should exclude the won (0, 0) small board")

    def test_valid_moves_when_small_board_full_draw(self):
        # Fill a small board without winning to simulate a draw and check valid moves
        for i in range(3):
            for j in range(3):
                self.game.boards[0, 0].board[i, j] = 1 if (i + j) % 2 == 0 else 2  # Alternate between Player 1 and 2
        self.game.last_move = ((0, 0), (2, 2))  # Last move directs to the full board
        valid_moves = self.game.get_valid_moves()
        # Exclude the full board from expected moves
        expected_moves = [((i, j), (k, l)) for i in range(3) for j in range(3) if (i, j) != (0, 0) for k in range(3) for l in range(3)]
        self.assertTrue(all(move in valid_moves for move in expected_moves), "Valid moves should exclude the full (0, 0) small board")

    def test_no_valid_moves_at_endgame(self):
        # Simulate an endgame scenario where the game is over
        for i in range(3):
            for j in range(3):
                self.game.boards[i, j].winner = 1 if (i + j) % 2 == 0 else 2  # Alternate wins between Player 1 and 2
                self.game.update_main_board((i, j))
        self.assertTrue(self.game.is_game_over(), "The game should be over")
        valid_moves = self.game.get_valid_moves()
        print(valid_moves)
        self.assertEqual(len(valid_moves), 0, "There should be no valid moves at the end of the game")


# class TestUltimateTicTacToe(unittest.TestCase):
#     def setUp(self):
#         self.game = UltimateTicTacToe()

#     def test_initialization(self):
#         expected_board = np.zeros((3, 3), dtype=int)
#         for i in range(3):
#             for j in range(3):
#                 self.assertTrue(np.array_equal(self.game.boards[i, j].board, expected_board))
#         self.assertEqual(self.game.main_board, np.zeros((3, 3), dtype=int))
#         self.assertEqual(self.game.turn, 1)
#         self.assertEqual(self.game.winner, 0)

#     def test_is_full(self):
#         self.assertFalse(self.game.is_full())
#         for i in range(3):
#             for j in range(3):
#                 self.game.boards[i, j].board = np.ones((3, 3), dtype=int)

#         self.assertTrue(self.game.is_full())
        


if __name__ == '__main__':
    unittest.main()
    