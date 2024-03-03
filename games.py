import numpy as np

# Implementing Tic Tac Toe

# The game is represented as a 3x3 matrix
# 0 represents an empty cell
# 1 represents a cell with a cross
# 2 represents a cell with a circle

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.turn = 1
        self.winner = 0

    def is_full(self):
        return np.all(self.board)

    def is_winner(self, player):
        for i in range(3):
            if np.all(self.board[i] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(self.board.diagonal() == player) or np.all(np.fliplr(self.board).diagonal() == player):
            return True
        return False

    def is_game_over(self):
        # Check if any player has won
        for player in [1, 2]:
            if self.is_winner(player):
                self.winner = player
                return True
        # If no winner and the board is full, it's a draw
        if self.is_full():
            self.winner = 0  # Indicate a draw
            return True
        # Game is not over
        return False


    def get_valid_moves(self):
        return np.argwhere(self.board == 0)
    
    def get_valid_moves_indices(self):
        return np.flatnonzero(self.board == 0)
    

    def make_move(self, move):
        self.board[tuple(move)] = self.turn
        self.turn = 3 - self.turn

    def make_move_from_index(self, index):
        move = np.unravel_index(index, (3, 3))
        self.make_move(move)

    def undo_move(self, move):
        self.board[tuple(move)] = 0
        self.turn = 3 - self.turn

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)

    def __str__(self):
        return str(self.board)
