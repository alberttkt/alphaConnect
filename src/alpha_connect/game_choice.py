from game import Connect4State, ChineseCheckersState, BounceState
import torch
from .chinese_checkers_helper import (
    output_to_logits_chinese_checkers,
    chinese_checkers_to_supervised_inputs,
    evaluate_chinese_checkers,
)
from .connect4_helper import connect4state_to_supervised_input
from .bounce_helper import (
    bounce_to_supervised_inputs,
    output_to_logits_bounce,
    outputs_to_values_bounce,
)


class GameChoice:
    __instance = None

    def __init__(self) -> None:
        if self.__class__.__instance != None:
            raise Exception("This class is a singleton!")
        self.game = Connect4State
        self.__class__.__instance = self

    @classmethod
    def get_instance(cls):
        if cls.__instance == None:
            GameChoice()
        return cls.__instance

    @staticmethod
    def set_game(new_game):
        GameChoice.get_instance().game = new_game

    @staticmethod
    def get_game():
        return GameChoice.get_instance().game

    @staticmethod
    def get_number_players():
        return len(GameChoice.get_instance().game.sample_initial_state().reward)

    @staticmethod
    def get_reward(state, player):
        if state.has_ended:
            return state.reward[player]
        if GameChoice.get_instance().game == ChineseCheckersState:
            return evaluate_chinese_checkers(state)[player]
        return 0

    @staticmethod
    def get_state_to_supervised_input(state) -> list:
        if GameChoice.get_instance().game == Connect4State:
            return connect4state_to_supervised_input(state)
        elif GameChoice.get_instance().game == ChineseCheckersState:
            return chinese_checkers_to_supervised_inputs(state)
        elif GameChoice.get_instance().game == BounceState:
            return bounce_to_supervised_inputs(state)

    @staticmethod
    def get_input_shape():
        if GameChoice.get_instance().game == Connect4State:
            return (-1, 3, 6, 7)
        elif GameChoice.get_instance().game == ChineseCheckersState:
            return (-1, 5, 17, 17)
        elif GameChoice.get_instance().game == BounceState:
            return (-1, 6, 9, 6)

    @staticmethod
    def model_output_to_proba_dict(outputs, states) -> list[dict]:
        if GameChoice.get_instance().game == Connect4State:
            policies = []
            for i, state in enumerate(states):
                d = {}
                policy = outputs[i]
                for i in state.actions:
                    d[i] = policy[i.to_json()]
                policies.append(d)
            return policies
        elif GameChoice.get_instance().game == ChineseCheckersState:
            return output_to_logits_chinese_checkers(outputs, states)
        elif GameChoice.get_instance().game == BounceState:
            return output_to_logits_bounce(outputs, states)

    @staticmethod
    def model_output_to_policy(outputs, states) -> list[list]:
        if GameChoice.get_instance().game == Connect4State:
            return outputs
        elif GameChoice.get_instance().game == ChineseCheckersState:
            dicts = output_to_logits_chinese_checkers(outputs, states)
            policies = []
            for i in range(len(dicts)):
                actions = states[i].actions
                d = dicts[i]
                p = []
                for action in actions:
                    p.append(d[action])
                policies.append(p)
            return policies
        elif GameChoice.get_instance().game == BounceState:
            dicts = output_to_logits_bounce(outputs, states)
            policies = []
            for i in range(len(dicts)):
                actions = states[i].actions
                d = dicts[i]
                p = []
                for action in actions:
                    p.append(d[action])
                policies.append(p)
            return policies

    @staticmethod
    def model_value_output_to_policy(outputs, states=None) -> list:
        if GameChoice.get_instance().game == Connect4State:
            return outputs
        elif GameChoice.get_instance().game == ChineseCheckersState:
            values = []
            for i in range(0, len(outputs), 10):
                values.append(
                    max(outputs[i : i + 10])
                )  # we take the max of the 10 values
            return torch.stack(values)
        elif GameChoice.get_instance().game == BounceState:
            return outputs_to_values_bounce(outputs, states)
