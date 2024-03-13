# Make sure all necessary imports are included
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import random

from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output

from games import TicTacToe
from SimplePolicyNetwork import SimplePolicyNetwork, makeValidMove

from benchmarks import random_agent_benchmark

import argparse

if torch.cuda.is_available():
    print("CUDA (GPU support) is available in this system.")
else:
    print("CUDA is not available. The program will run on CPU.")

parser = argparse.ArgumentParser(description='trains an rl agent to play tic tac toe against and random agent')
parser.add_argument('--eps', type=int, default=10000, help='Number of episodes')
args = parser.parse_args()

game = TicTacToe()
# Initialize the policy network
board_size = (3, 3)  # For Tic Tac Toe
num_moves = game.board.numel() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_network = SimplePolicyNetwork(board_size, num_moves).to(device)
optimizer = optim.Adam(policy_network.parameters(), lr=0.0001)


def compute_returns(rewards, gamma=1.0):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, device=device, dtype=torch.float)


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


num_episodes = args.eps

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
        # print(f"{episode}:{game.board}")
        if game.turn == 1:  # Policy network's turn
            action_log_prob = makeValidMove(game, policy_network)
            saved_log_probs.append(action_log_prob)
        else:  # Random agent's turn
            valid_moves = game.get_valid_moves_indices().to(device)
            weights = torch.ones_like(valid_moves, dtype=torch.float)
            action_index = torch.multinomial(weights, 1)
            action = valid_moves[action_index].item()
            game.make_move_from_index(action)
        reward = get_reward(game) 
        rewards.append(reward)
    
    # print("reward: ", rewards)


    # Compute returns
    returns = compute_returns(rewards)
    returns = torch.tensor(returns)

    # Determine the outcome of the episode
    win_record.append(game.winner)
    episode_rewards.append(torch.tensor(rewards, device=device, dtype=torch.float).mean().item())
    
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
        random_agent_benchmark(policy_network, random_start=True)