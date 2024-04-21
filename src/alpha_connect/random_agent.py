from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _play_logic(self, state):
        nb_actions = len(state.actions)
        actions = list(state.actions)
        d = {action.to_json(): 1.0 / nb_actions for action in actions}
        return d, 0
