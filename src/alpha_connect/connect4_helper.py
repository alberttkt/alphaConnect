import torch


def connect4state_to_supervised_input(state):
    tensor = torch.tensor(state.get_tensors()[0])
    output = torch.zeros((3, 6, 7))
    output[0] = (tensor == -1).float()
    if state.player == 0:
        output[1] = (tensor == 0).float()
        output[2] = (tensor == 1).float()
    else:
        output[1] = (tensor == 1).float()
        output[2] = (tensor == 0).float()
    return [output]
