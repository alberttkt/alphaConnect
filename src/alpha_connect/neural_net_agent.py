import torch
from .agent import Agent
from .game_choice import GameChoice
from .model import AlphaZeroModelConnect4
from torch import nn


class NeuralNetAgent(Agent):
    def __init__(
        self,
        model=None,
    ):
        if model is None:
            model = AlphaZeroModelConnect4.load_from_checkpoint(
                "/Users/alberttroussard/Documents/alpha-connect/data/supervised.ckpt"
            )
        self.model = model.to("mps")

    def _play_logic(self, state):
        self.model.eval()
        inp = GameChoice.get_state_to_supervised_input(state)
        y_hat, value = self.model(
            torch.stack(inp)
            .to("mps")
            .type(torch.float32)
            .view(GameChoice.get_input_shape())
        )
        # TODO: fix
        distribution = GameChoice.model_output_to_proba_dict(y_hat, [state])[0]

        self.model.train()

        return distribution, float(max(value))
