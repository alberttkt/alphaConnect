from .agent import Agent


class OneMoveAheadAgent(Agent):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _play_logic(self, state):
        actions = list(state.actions)
        for action in actions:
            if action.sample_next_state().has_ended:
                return {action: 1.0}, 1
        possible = set(actions)
        for action in actions:
            for action2 in action.sample_next_state().actions:
                if action2.sample_next_state().has_ended:
                    possible.remove(action)
                    break
        if len(possible) == 0:
            possible = set(actions)

        nb_actions = len(possible)
        actions = list(possible)

        d = {action: 1.0 / nb_actions for action in actions}
        return d, 1
