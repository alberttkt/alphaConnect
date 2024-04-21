import sys
import torch
from .game import Game
import random
from game import ConnectState
from .game_choice import GameChoice


def show_board(state, file=sys.stdout):
    a = state.get_tensors()[0]
    for i in range(5, -1, -1):
        for j in range(7):
            c = "." if a[i][j] == -1 else "X" if a[i][j] == 1 else "O"
            print(c, end=" ", file=file)
        print(file=file)


def play_game(agents, starting_state=None):
    gamestate = Game(agents, starting_state)
    while not gamestate.has_ended():
        gamestate = gamestate.play()
    show_board(gamestate.state)
    return gamestate.reward()[0]


def play_connect4_games(agent1, agent2, n=100, random_start_agent=True):
    previous_game = GameChoice.get_game()
    GameChoice.set_game(ConnectState)
    dict = {"agent1": 0, "agent2": 0, "draws": 0}
    agents = [agent1, agent2]
    for i in range(n):
        print(f"Game {i}")
        print(dict)

        if random_start_agent:
            if random.choice([True, False]):
                result = play_game(agents)
            else:
                result = play_game(agents[::-1])
                result = -result
        else:
            result = play_game(agents)
        if result == 1:
            dict["agent1"] += 1
        elif result == -1:
            dict["agent2"] += 1
        else:
            dict["draws"] += 1

    GameChoice.set_game(previous_game)
    return dict


def state_to_supervised_input(state):
    tensor = torch.tensor(state.get_tensors()[0])
    output = torch.zeros((3, 6, 7))
    output[0] = (tensor == -1).float()
    if state.player == 0:
        output[1] = (tensor == 0).float()
        output[2] = (tensor == 1).float()
    else:
        output[1] = (tensor == 1).float()
        output[2] = (tensor == 0).float()
    return output
