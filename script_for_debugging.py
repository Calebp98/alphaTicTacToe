# Make sure all necessary imports are included
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import random

from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output

from games import TicTacToe

# Implementing a simple policy network for tic tac toe that outputs a probability distribution over all legal moves

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

# Example usage
board_size = (3, 3)  # For Tic Tac Toe
num_moves = board_size[0] * board_size[1]  # Assuming each cell is a possible move
model = SimplePolicyNetwork(board_size, num_moves)

def convert_board_to_input(board):
    """
    Convert the game board to a tensor suitable for the policy network.
    The input is a 4D tensor: [batch_size, channels, height, width].
    """
    # Convert the board to a tensor with shape (1, 1, 3, 3)
    # 1 channel, the board's state is represented in a 3x3 grid
    board_tensor = torch.tensor(board, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    return board_tensor

def select_move(probabilities, valid_moves_indices):
    """
    Select the move with the highest probability that is also a valid move.
    """
    # Zero out the probabilities of moves that are not valid
    prob_masked = probabilities.clone().detach()
    prob_masked[0, np.setdiff1d(np.arange(num_moves), valid_moves_indices)] = 0
    # Select the move with the highest probability
    move_index = torch.argmax(prob_masked).item()
    return move_index


class RandomAgent:
    def __init__(self):
        pass

    def select_move(self, valid_moves_indices):
        """
        Selects a move randomly from the list of valid move indices.
        :param valid_moves_indices: A list of indices representing valid moves.
        :return: An index representing the selected move.
        """
        return random.choice(valid_moves_indices)

def play_game_with_random_old(policy_model, game, random_agent):
    game.reset()
    # Randomly choose which agent starts
    # game.turn = random.choice([1, 2])
    while not game.is_game_over():
        if game.turn == 1:  # Policy network plays and is player 1
            current_state_tensor = convert_board_to_input(game.board)
            probabilities = policy_model(current_state_tensor)
            valid_moves_indices = game.get_valid_moves_indices()
            selected_move_index = select_move(probabilities, valid_moves_indices)
            game.make_move_from_index(selected_move_index)
        else:  # Random agent's turn
            valid_moves_indices = game.get_valid_moves_indices()
            selected_move_index = random_agent.select_move(valid_moves_indices)
            game.make_move_from_index(selected_move_index)
    return game.winner # return 1 if first player to play won


# Defining a function to simulate a game between two random agents
def play_game_random_vs_random(game, agent1, agent2):
    game.reset()
    # Randomly choose which agent starts
    game.turn = random.choice([1, 2])
    while not game.is_game_over():
        if game.turn == 1:  # Random agent 1's turn
            valid_moves_indices = game.get_valid_moves_indices()
            selected_move_index = agent1.select_move(valid_moves_indices)
            game.make_move_from_index(selected_move_index)
        else:  # Random agent 2's turn
            valid_moves_indices = game.get_valid_moves_indices()
            selected_move_index = agent2.select_move(valid_moves_indices)
            game.make_move_from_index(selected_move_index)
    return game.winner


def play_game_with_random(policy_model, game, random_start=True):
    if random_start:
        game.turn = random.choice([1, 2])
    
    while not game.is_game_over():
        if game.turn == 1:  # Policy network's turn
            state = convert_board_to_input(game.board)
            probs = policy_model(state)
            
            valid_moves = game.get_valid_moves_indices()
            action = select_move(probs, valid_moves)

            game.make_move_from_index(action)
            # print(1)
            # print(action)
            # print(np.unravel_index(action, (3, 3)))
        else:  # Random agent's turn
            valid_moves = game.get_valid_moves_indices()
            action = np.random.choice(valid_moves)
            game.make_move_from_index(action)
            # print(2)
            # print(action)
            # print(np.unravel_index(action, (3, 3)))
        # print(game)
        # print("------")
    return game.winner



def random_agent_benchmark(model, total_games=500, random_start=True):
    random_agent = RandomAgent()
    testGame = TicTacToe()
    win_count = 0
    loss_count = 0
    draw_count = 0
    for _ in range(total_games):
        testGame.reset()

        outcome = play_game_with_random(model, testGame, random_start=random_start)
        if outcome == 1:
            win_count += 1
        elif outcome == 2:
            loss_count += 1
        elif outcome == 0:
            draw_count += 1

    print(f"Win Rate: {win_count/total_games*100}%")
    print(f"Loss Rate: {loss_count/total_games*100}%")
    print(f"Draw Rate: {draw_count/total_games*100}%")


def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
    and a discount factor gamma.
    """
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Initialize environment, policy network, and optimizer
game = TicTacToe()
policy_network = SimplePolicyNetwork(board_size=(3, 3), num_moves=9)
optimizer = optim.Adam(policy_network.parameters(), lr=0.00001)

num_episodes = 1000


def get_reward(game):
    """
    Defines the reward for the policy network based on the game's outcome.
    Assume the policy network always plays as player 1.
    """
    if game.is_winner(1):  # Policy network wins
        return 1
    elif game.is_winner(2):  # Policy network loses
        return -1
    return 0  # No reward for intermediate moves


win_record = []
policy_losses = []  # Store policy losses for plotting
episode_rewards = []  # Average reward per episode

# Training loop
for episode in tqdm(range(num_episodes), desc="Processing Episodes"):
    saved_log_probs = []
    rewards = []
    game.reset()
    # game.turn = random.choice([1, 2])
    # print("Game: ", game)
    # print("Episode: ", episode)
    while not game.is_game_over():
        if game.turn == 1:  # Policy network's turn
            state = convert_board_to_input(game.board)
            probs = policy_network(state)
            # action = torch.multinomial(probs, 1).item()  # Sample action

            valid_moves = game.get_valid_moves_indices()
            action = select_move(probs, valid_moves)
            saved_log_probs.append(torch.log(probs.squeeze(0)[action]))
            game.make_move_from_index(action)
        else:  # Random agent's turn
            valid_moves = game.get_valid_moves_indices()
            action = np.random.choice(valid_moves)
            game.make_move_from_index(action)
        reward = get_reward(game)  # Define a suitable reward function
        rewards.append(reward)
    # print("reward: ", rewards)
    

    
    # Compute returns
    returns = compute_returns(rewards)
    returns = torch.tensor(returns)

    # Determine the outcome of the episode
    win_record.append(game.winner)
    episode_rewards.append(np.mean(rewards))
    
    # Collect individual loss tensors in a list
    policy_loss_terms = []
    for log_prob, R in zip(saved_log_probs, returns):
        loss = -log_prob * R
        policy_loss_terms.append(loss.unsqueeze(0))

    # Concatenate and sum to compute the total policy loss for the episode
    total_policy_loss = torch.cat(policy_loss_terms).sum()
    policy_losses.append(total_policy_loss.item())  # Recording scalar loss

    # Use total_policy_loss for gradient computation
    optimizer.zero_grad()
    total_policy_loss.backward()
    optimizer.step()

    interval = num_episodes // 10
    if episode > 0 and (episode) % interval == 0:
        random_agent_benchmark(policy_network, random_start=False)
