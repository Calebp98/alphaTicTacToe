import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from board_helper import convert_board_to_input


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Policy network for tic tac toe that outputs a probability distribution over all moves
class SimplePolicyNetwork(nn.Module):
    def __init__(self, board_size, num_moves):
        """
        Initializes the Policy Network.
        :param board_size: Tuple of the board dimensions, e.g., (19, 19) for Go.
        :param num_moves: Total number of possible moves in the game.
        """
        super(SimplePolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * board_size[0] * board_size[1], num_moves)

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: Input tensor, the game state.
        :return: Probability distribution over all possible moves.
        """
        # Apply two convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Output layer with a softmax to get probabilities
        x = self.fc(x)
        return F.softmax(x, dim=1)


# def select_move(probabilities, valid_moves_indices, num_moves):
#     """
#     Select the move with the highest probability that is also a valid move.
#     """
#     # Zero out the probabilities of moves that are not valid
#     prob_masked = probabilities.clone().detach()
#     prob_masked[0, np.setdiff1d(np.arange(num_moves), valid_moves_indices)] = 0
#     # Select the move with the highest probability
#     move_index = torch.argmax(prob_masked).item()
#     return move_index
    
def select_move(probabilities, valid_moves_indices, num_moves):
    """
    Select the move with the highest probability that is also a valid move.
    """
    # Initialize a mask with zeros for all possible moves
    mask = torch.zeros(num_moves, device=probabilities.device, dtype=torch.bool)

    # Set the indices of valid moves to True
    mask[valid_moves_indices] = True

    # Adjust the mask shape to match probabilities tensor ([1, 9])
    mask = mask.unsqueeze(0)  # Now mask shape is [1, 9]

    # Zero out the probabilities of moves that are not valid by applying the mask
    prob_masked = probabilities.clone().detach()
    prob_masked[~mask] = 0  # Invert the mask to zero out invalid moves
    
    # Select the move with the highest probability among valid moves
    move_index = torch.argmax(prob_masked).item()
    return move_index



def makeValidMove(game, model):
    num_moves = game.board.numel() 
    # Ensure current_state_tensor is correctly shaped and on the correct device
    current_state_tensor = convert_board_to_input(game.board).to(device)
    probabilities = model(current_state_tensor).to(device)
    
    # Directly use the tensor returned by get_valid_moves_indices
    valid_moves_indices = game.get_valid_moves_indices().to(device)  # Ensure it's on the correct device
    
    # Now select_move should work without modifications, assuming num_moves is correctly defined
    selected_move_index = select_move(probabilities, valid_moves_indices, num_moves)
    game.make_move_from_index(selected_move_index)

    return torch.log(probabilities.squeeze(0)[selected_move_index])
