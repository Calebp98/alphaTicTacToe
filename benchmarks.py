from games import TicTacToe
import random
import torch
from tqdm import tqdm

from SimplePolicyNetwork import makeValidMove

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# randomly intialiased policy net vs random actions
def play_game_with_random(policy_model, game, random_start=True):
    if random_start:
        game.turn = random.choice([1, 2])
    
    while not game.is_game_over():
        if game.turn == 1:  # Policy network's turn
            makeValidMove(game, policy_model)
        else:  # Random agent's turn
            valid_moves = game.get_valid_moves_indices().to(device)
            weights = torch.ones_like(valid_moves, dtype=torch.float)
            action_index = torch.multinomial(weights, 1)
            action = valid_moves[action_index].item()
            game.make_move_from_index(action)
    return game.winner

def random_agent_benchmark(model, total_games=500, random_start=True):
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

    win_rate = (win_count / total_games) * 100
    loss_rate = (loss_count / total_games) * 100
    draw_rate = (draw_count / total_games) * 100

    # Update the tqdm postfix to display metrics
    message = f"Win Rate: {win_rate}%, Loss Rate: {loss_rate}%, Draw Rate: {draw_rate}%"
    tqdm.write(message)
    return {'win_rate': win_rate, 'loss_rate': loss_rate, 'draw_rate': draw_rate}



# parallel test (not working)
    
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from games import TicTacToe
from SimplePolicyNetwork import makeValidMove, SimplePolicyNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume your play_game_with_random and other necessary functions are defined here

def run_single_game(model_state_dict, random_start=True):
    # Initialize model and game inside the worker process
    policy_model = SimplePolicyNetwork().to(device)
    policy_model.load_state_dict(model_state_dict)
    policy_model.eval()  # Set model to evaluation mode
    
    game = TicTacToe()
    outcome = play_game_with_random(policy_model, game, random_start=random_start)
    return outcome

def random_agent_benchmark_parallel(model, total_games=500, random_start=True, num_workers=4):
    model_state_dict = model.state_dict()  # Transfer model state instead of the model itself

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_game, model_state_dict, random_start) for _ in range(total_games)]
        
        win_count = 0
        loss_count = 0
        draw_count = 0

        for future in as_completed(futures):
            outcome = future.result()
            if outcome == 1:
                win_count += 1
            elif outcome == 2:
                loss_count += 1
            elif outcome == 0:
                draw_count += 1


    print(f"Win Rate: {win_count / total_games * 100}%")
    print(f"Loss Rate: {loss_count / total_games * 100}%")
    print(f"Draw Rate: {draw_count / total_games * 100}%")

# Example usage
# model = SimplePolicyNetwork().to(device)
# random_agent_benchmark_parallel(model, total_games=500, random_start=True, num_workers=4)
