import torch
from .agent import Agent
from .helper import state_to_supervised_input
from .model import AlphaZeroModel
from torch import nn


class NeuralNetAgent(Agent):
    def __init__(
        self,
        model: AlphaZeroModel = None,
    ):
        if model is None:
            model = AlphaZeroModel.load_from_checkpoint(
                "/Users/alberttroussard/Documents/alpha-connect/data/supervised.ckpt"
            )
        self.model = model.to("mps")

    def _play_logic(self, state):
        self.model.eval()
        input = state_to_supervised_input(state)
        y_hat, value = self.model(
            torch.tensor(input, device="mps").type(torch.float32).view(1, 3, 6, 7)
        )
        distribution = nn.functional.softmax(y_hat, dim=1).flatten()
        d = {i: distribution[i].item() for i in range(len(distribution))}
        self.model.train()
        return d, float(value)
