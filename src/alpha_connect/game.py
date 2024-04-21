import random
from .agent import Agent
from .game_choice import GameChoice


class Game:
    def __init__(self, agents: list[Agent], state=None):
        self.agents = agents
        if state is None:
            state = GameChoice.get_game().sample_initial_state()
            assert len(agents) == GameChoice.get_number_players()
        self.state = state

    def reset(self):
        return Game(self.agents)

    def has_ended(self):
        return self.state.has_ended

    def play(self):
        moves_proba_dict, _ = self.agents[self.state.player].play(self.state)
        max_proba = max(moves_proba_dict.values())
        moves = [
            move for move in moves_proba_dict if moves_proba_dict[move] == max_proba
        ]
        return Game(self.agents, random.choice(moves).sample_next_state())

    def get_value(self, column, row):
        return self.state.get_tensors()[0][row][column]

    def reward(self):
        return self.state.reward
