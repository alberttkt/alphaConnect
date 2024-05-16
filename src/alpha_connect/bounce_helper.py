from math import exp
import torch


def bounce_to_supervised_inputs(state):
    base = torch.zeros((6, 9, 6))
    board = state.get_tensors()[0]
    # add 1 rows of zeros to the top of the board and 1 row of zeros to the bottom of the board to represent the 'goals'
    board = torch.cat((torch.zeros(1, 6), torch.tensor(board), torch.zeros(1, 6)))
    base[0] = (board == 0).float()  # free slots
    base[1] = (board == 1).float()  # pieces 1
    base[2] = (board == 2).float()  # pieces 2
    base[3] = (board == 3).float()  # pieces 3

    outputs = []
    start_point_to_actions = categorise_actions(state)

    for start_point, actions in sorted(start_point_to_actions.items()):
        output = base.clone()
        output[4][start_point[1] + 1][start_point[0]] = 1
        for action in actions:
            end_point = [action.to_json()["to"]["x"], action.to_json()["to"]["y"]]
            output[5][end_point[1] + 1][end_point[0]] = 1

        # flip the board if player is 1
        if state.player == 1:
            output = torch.flip(output, [1])

        outputs.append(output)

    return outputs


def output_to_logits_bounce(outputs, states):
    policies = []

    for _, state in enumerate(states):
        d = {}
        start_point_to_actions = categorise_actions(state)
        for _, actions in sorted(start_point_to_actions.items()):
            logits = outputs[0]
            outputs = outputs[1:]
            for action in actions:
                end_point = [action.to_json()["to"]["x"], action.to_json()["to"]["y"]]
                if end_point[1] in [-1, 7]:
                    d[action] = logits[6 * 9]
                d[action] = logits[(end_point[1]) * 6 + end_point[0]]

        # softmax
        sum_exp = sum([exp(logit) for logit in d.values()])
        d = {k: exp(v) / sum_exp for k, v in d.items()}
        policies.append(d)
    return policies


def outputs_to_values_bounce(outputs, states):
    values = []
    for state in states:
        start_point_to_actions = categorise_actions(state)
        l = len(start_point_to_actions)
        values.append(max(outputs[:l]))
        outputs = outputs[l:]
    return torch.stack(values)


def categorise_actions(state):
    start_point_to_actions = {}
    for action in state.actions:
        start_point = (
            action.to_json()["from"]["x"],
            action.to_json()["from"]["y"],
        )  # tuple or list?
        start_point_to_actions[start_point] = start_point_to_actions.get(
            start_point, []
        ) + [action]
    return start_point_to_actions
