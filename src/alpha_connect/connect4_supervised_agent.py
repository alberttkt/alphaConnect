import torch
from src.alpha_connect.agent import Agent
from src.alpha_connect.connect4_helper import state_to_supervised_input
from src.alpha_connect.supervised_model import AlphaZeroModel
from torch import nn


class Connect4SupervisedAgent(Agent):
    def __init__(
        self,
        model_path="/Users/alberttroussard/Documents/alpha-connect/data/supervised.ckpt",
    ):
        self.model = AlphaZeroModel.load_from_checkpoint(model_path).to("mps")
        self.model.eval()

    def _play_logic(self, state):
        input = state_to_supervised_input(state)
        y_hat, value = self.model(
            torch.tensor(input, device="mps").type(torch.float32).view(1, 3, 6, 7)
        )
        distribution = nn.functional.softmax(y_hat, dim=1).flatten()
        d = {i: distribution[i].item() for i in range(len(distribution))}
        return d, float(value)
