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
        self.winner = 0 #

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
            if self.winner == player:
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

        # check if the move is valid
        if self.board[tuple(move)] != 0:
            raise ValueError('Invalid move')
        
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
        self.turn = 1
        self.winner = 0 #

    def __str__(self):
        return str(self.board)
    


class UltimateTicTacToe:
    def __init__(self):
        # Initialize 3x3 grid of TicTacToe boards
        self.boards = np.array([[TicTacToe() for _ in range(3)] for _ in range(3)])
        # Main board to track winners of small boards
        self.main_board = np.zeros((3, 3), dtype=int)
        self.turn = 1
        self.winner = 0
        # Last move format: ((board_row, board_col), (move_row, move_col))
        self.last_move = None

    def update_main_board(self, board_pos):
        board = self.boards[board_pos]
        if board.winner:
            self.main_board[board_pos] = board.winner

    def is_full(self):
        return np.all(self.main_board)

    def is_winner(self, player):
        for i in range(3):
            if np.all(self.main_board[i] == player) or np.all(self.main_board[:, i] == player):
                return True
        if np.all(np.diag(self.main_board) == player) or np.all(np.diag(np.fliplr(self.main_board)) == player):
            return True
        return False

    def is_game_over(self):
        for player in [1, 2]:
            if self.is_winner(player):
                self.winner = player
                return True
        if self.is_full():
            self.winner = 0  # Indicate a draw
            return True
        return False

    # def get_valid_moves(self):
    #     if self.last_move is None:
    #         return [(i, j) for i in range(3) for j in range(3) if self.main_board[i, j] == 0]
    #     else:
    #         board_pos, _ = self.last_move
    #         if self.boards[board_pos].winner == 0:
    #             return [(board_pos, move) for move in self.boards[board_pos].get_valid_moves_indices()]
    #         else:
    #             # If the target board is already won or full, the player can choose any other board
    #             return [(i, j) for i in range(3) for j in range(3) if self.main_board[i, j] == 0]

    def get_valid_moves(self):
        valid_moves = []

        for i in range(3):
            for j in range(3):
                if not self.boards[i, j].is_game_over():  # Check if the small board is not full/won
                    print(self.boards[i, j].winner)
                    for k in range(3):
                        for l in range(3):
                            if self.boards[i, j].board[k, l] == 0:  # Check if the cell is empty
                                valid_moves.append(((i, j), (k, l)))
        return valid_moves

    def make_move(self, move):
        board_pos, move_pos = move
        board = self.boards[board_pos]
        if not board.winner and board.board[move_pos] == 0:
            board.turn = self.turn
            board.make_move(move_pos)
            board.is_game_over()
            self.update_main_board(board_pos)
            self.last_move = (board_pos, move_pos)
            self.turn = 3 - self.turn
        

    def undo_move(self, move):
        board_pos, move_pos = move
        board = self.boards[board_pos]
        board.undo_move(move_pos)
        self.update_main_board(board_pos)  # In case undoing the move changes the winner
        self.last_move = None  # Reset last move as undoing might change the game state significantly
        self.turn = 3 - self.turn

    def reset(self):
        for row in self.boards:
            for board in row:
                board.reset()
        self.main_board = np.zeros((3, 3), dtype=int)
        self.turn = 1
        self.winner = 0
        self.last_move = None

    def __str__(self):
        return '\n'.join([' '.join([str(board) for board in row]) for row in self.boards])

    def __repr__(self):
        return self.__str__()
