from src.alpha_connect.agent import Agent
from game import ConnectState


State = ConnectState


class Connect4Game:
    def __init__(
        self, agent1: Agent, agent2: Agent, state: State = State.sample_initial_state()
    ):
        self.agent1 = agent1
        self.agent2 = agent2
        self.state = state

    def reset(self):
        return Connect4Game(self.agent1, self.agent2)

    def has_ended(self):
        return self.state.has_ended

    def play(self):
        if self.state.player == 0:
            moves_proba_dict, value = self.agent1.play(self.state)
        else:
            moves_proba_dict, value = self.agent2.play(self.state)

        for move in sorted(moves_proba_dict, key=moves_proba_dict.get, reverse=True):
            return Connect4Game(self.agent1, self.agent2, move.sample_next_state())

        raise Exception("No moves available")

    def get_value(self, column, row):
        return self.state.get_tensors()[0][row][column]

    def reward(self):
        return self.state.reward
