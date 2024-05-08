import sys
import torch
from .game import Game
import random
from game import Connect4State
from .game_choice import GameChoice


full_board = torch.tensor(
    [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
)


def show_board(state, file=sys.stdout):
    a = state.get_tensors()[0]
    for i in range(5, -1, -1):
        for j in range(7):
            c = "." if a[i][j] == -1 else "X" if a[i][j] == 1 else "O"
            print(c, end=" ", file=file)
        print(file=file)


def play_game(agents, starting_state=None, greedy=True):
    gamestate = Game(agents, starting_state)
    while not gamestate.has_ended():
        gamestate = gamestate.play(greedy=greedy)
    show_board(gamestate.state)
    return gamestate.reward()[0]


def play_connect4_games(agent1, agent2, n=100, random_start_agent=True, greedy=True):
    previous_game = GameChoice.get_game()
    GameChoice.set_game(Connect4State)
    dict = {"agent1": 0, "agent2": 0, "draws": 0}
    agents = [agent1, agent2]
    for i in range(n):
        print(f"Game {i}")
        print(dict)

        if random_start_agent:
            if random.choice([True, False]):
                result = play_game(agents, greedy=greedy)
            else:
                result = play_game(agents[::-1], greedy=greedy)
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


def chinese_checkers_to_supervised_input(state, piece_to_move):
    pieces = state.to_json()["pieces"]
    current_player_pieces = pieces[state.player]
    other_player_pieces = pieces[1 - state.player]

    output = torch.zeros((5, 17, 19))
    # output is 5 layers of size 20*20 representing the board
    # layer 0: empty squares of the board
    # layer 1: current player pieces
    # layer 2: other player pieces
    # layer 3: piece to move
    # layer 4: possible actions

    output[0] = full_board

    for i in range(17):
        for j in range(19):
            if [i, j] in current_player_pieces:
                output[1][i][j] = 1
                output[0][i][j] = 0
            elif [i, j] in other_player_pieces:
                output[2][i][j] = 1
                output[0][i][j] = 0

    output[3][piece_to_move[0]][piece_to_move[1]] = 1

    piece_index = state.to_json()["pieces"][state.player].index(piece_to_move)
    for action in state.actions:
        piece = action.to_json()
        if piece["index"] == piece_index:
            x = piece["x"]
            y = piece["y"]
            output[4][x][y] = 1

    if state.player == 1:  # if the player is 1, we flip the board
        output = [[list(i) for i in zip(*layer)] for layer in output]

    return torch.tensor(output)
