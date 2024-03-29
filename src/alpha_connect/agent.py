from abc import ABC, abstractmethod
import math
import random


class Agent(ABC):
    @abstractmethod
    def _play_logic(self, state) -> tuple[dict[str, float], float]:
        ...

    def play(self, state) -> tuple[dict[str, float], float]:
        moves_proba_dict, value = self._play_logic(state)
        return Agent.filter_moves(state, moves_proba_dict), value

    def sample_move(self, state):
        moves_proba_dict, _ = self.play(state)
        return random.choices(
            list(moves_proba_dict.keys()), weights=moves_proba_dict.values(), k=1
        )[0]

    @staticmethod
    def filter_moves(state, moves_proba_dict: dict[str, float]) -> dict[str, float]:
        res = {}
        for action in state.actions:
            res[action] = moves_proba_dict.get(action.to_json(), 0)
        total = sum([math.exp(prob) for prob in res.values()])
        res = {k: math.exp(v) / total for k, v in res.items()}
        return res
