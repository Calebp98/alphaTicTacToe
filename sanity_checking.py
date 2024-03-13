# Make sure all necessary imports are included
import numpy as np
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

from benchmarks import play_game_with_random

game = TicTacToe()
# Initialize the policy network
board_size = (3, 3)  # For Tic Tac Toe
num_moves = board_size[0] * board_size[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplePolicyNetwork(board_size, num_moves).to(device)




wins = {1: 0, 2: 0, 0: 0}  # 1: Policy Network, 2: Random Agent, 0: Draw
num_simulations = 1000000

# Use tqdm for progress feedback in a console application
for _ in tqdm(range(num_simulations)):
    winner = play_game_with_random(model, game)
    wins[winner] += 1
    game.reset()

# Plotting win rates
labels = ['Policy Network Wins', 'Random Agent Wins', 'Draws']
values = [wins[1], wins[2], wins[0]]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'red', 'green'])
plt.title('Total Wins')
plt.ylabel('Number of Games')
plt.savefig('plot.png') 
plt.close()