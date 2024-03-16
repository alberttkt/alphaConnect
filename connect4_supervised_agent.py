import torch 
from connect4_agent import Connect4Agent
import lightning as L
from supervised_model import MyModel
import torch.nn as nn


class Connect4SupervisedAgent(Connect4Agent):
    def __init__(self, model_path="epoch=7-step=1024.ckpt"):
        self.model = MyModel.load_from_checkpoint(model_path).to('mps')
        self.model.eval()
        

    def play(self, state):
        tensor = state.get_tensors()[0].flatten()

        y_hat = self.model(torch.tensor(tensor, device='mps').type(torch.float32))
        distribution = nn.functional.softmax(y_hat, dim=0)
        d = {i: distribution[i].item() for i in range(len(distribution))}
        return d
    