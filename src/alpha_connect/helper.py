import sys
from .game import Game
import random
from game import Connect4State


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
    # show_board(gamestate.state)
    return gamestate.reward()[0]


def play_2players_games(agent1, agent2, n=100, random_start_agent=True, greedy=True):
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

    return dict
