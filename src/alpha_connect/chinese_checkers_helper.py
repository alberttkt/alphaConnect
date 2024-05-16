from math import exp, sqrt

import torch


logits_order = [
    [0, 12],
    [1, 11],
    [1, 12],
    [2, 10],
    [2, 11],
    [2, 12],
    [3, 9],
    [3, 10],
    [3, 11],
    [3, 12],
    [4, 4],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [4, 9],
    [4, 10],
    [4, 11],
    [4, 12],
    [4, 13],
    [4, 14],
    [4, 15],
    [4, 16],
    [5, 4],
    [5, 5],
    [5, 6],
    [5, 7],
    [5, 8],
    [5, 9],
    [5, 10],
    [5, 11],
    [5, 12],
    [5, 13],
    [5, 14],
    [5, 15],
    [6, 4],
    [6, 5],
    [6, 6],
    [6, 7],
    [6, 8],
    [6, 9],
    [6, 10],
    [6, 11],
    [6, 12],
    [6, 13],
    [6, 14],
    [7, 4],
    [7, 5],
    [7, 6],
    [7, 7],
    [7, 8],
    [7, 9],
    [7, 10],
    [7, 11],
    [7, 12],
    [7, 13],
    [8, 4],
    [8, 5],
    [8, 6],
    [8, 7],
    [8, 8],
    [8, 9],
    [8, 10],
    [8, 11],
    [8, 12],
    [9, 3],
    [9, 4],
    [9, 5],
    [9, 6],
    [9, 7],
    [9, 8],
    [9, 9],
    [9, 10],
    [9, 11],
    [9, 12],
    [10, 2],
    [10, 3],
    [10, 4],
    [10, 5],
    [10, 6],
    [10, 7],
    [10, 8],
    [10, 9],
    [10, 10],
    [10, 11],
    [10, 12],
    [11, 1],
    [11, 2],
    [11, 3],
    [11, 4],
    [11, 5],
    [11, 6],
    [11, 7],
    [11, 8],
    [11, 9],
    [11, 10],
    [11, 11],
    [11, 12],
    [12, 0],
    [12, 1],
    [12, 2],
    [12, 3],
    [12, 4],
    [12, 5],
    [12, 6],
    [12, 7],
    [12, 8],
    [12, 9],
    [12, 10],
    [12, 11],
    [12, 12],
    [13, 4],
    [13, 5],
    [13, 6],
    [13, 7],
    [14, 4],
    [14, 5],
    [14, 6],
    [15, 4],
    [15, 5],
    [16, 4],
]

full_board = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def output_to_logits_chinese_checkers(outputs, states):
    policies = []

    for i, state in enumerate(states):
        d = {}
        actions = state.actions
        index_to_action = {}
        for action in actions:
            index_to_action[action.to_json()["index"]] = index_to_action.get(
                action.to_json()["index"], []
            ) + [action]
        logits = outputs[i * 10 : (i + 1) * 10]
        pieces = state.to_json()["pieces"][state.player]
        for index, _ in enumerate(pieces):
            piece_logits = logits[index]
            index_actions = index_to_action.get(index, [])
            for action in index_actions:
                logit_idx = logits_order.index(
                    [action.to_json()["x"], action.to_json()["y"]]
                )
                d[action] = piece_logits[logit_idx]

        # softmax
        sum_exp = sum([exp(logit) for logit in d.values()])
        d = {k: exp(v) / sum_exp for k, v in d.items()}
        policies.append(d)
    return policies


def chinese_checkers_to_supervised_input(state, piece_to_move):
    pieces = state.to_json()["pieces"]
    current_player_pieces = pieces[state.player]
    other_player_pieces = pieces[1 - state.player]

    output = torch.zeros((5, 17, 17))
    # output is 5 layers of size 17*17 representing the board
    # layer 0: empty squares of the board
    # layer 1: current player pieces
    # layer 2: other player pieces
    # layer 3: piece to move
    # layer 4: possible actions

    output[0] = full_board

    for i in range(17):
        for j in range(17):
            if [i, j] in current_player_pieces:
                output[1][i][j] = 1
                output[0][i][j] = 0
            elif [i, j] in other_player_pieces:
                output[2][i][j] = 1
                output[0][i][j] = 0

    output[3][piece_to_move[0]][piece_to_move[1]] = 1

    piece_index = state.to_json()["pieces"][state.player].index(piece_to_move)
    for action in state.actions:
        piece = action.to_json()
        if piece["index"] == piece_index:
            x = piece["x"]
            y = piece["y"]
            output[4][x][y] = 1

    if state.player == 1:  # if the player is 1, we flip the board
        output = torch.flip(output, [1])

    return torch.tensor(output)


def chinese_checkers_to_supervised_inputs(state):
    res = []
    for piece in state.to_json()["pieces"][state.player]:
        res.append(chinese_checkers_to_supervised_input(state, piece))

    return res


def evaluate_chinese_checkers(state):
    player1_goal = (12, 12)
    player2_goal = (4, 4)
    player1 = state.to_json()["pieces"][0]
    player2 = state.to_json()["pieces"][1]
    player1_score = sum(
        [
            sqrt((player1_goal[0] - i) ** 2 + (player1_goal[1] - j) ** 2)
            for i, j in player1
        ]
    )
    player2_score = sum(
        [
            sqrt((player2_goal[0] - i) ** 2 + (player2_goal[1] - j) ** 2)
            for i, j in player2
        ]
    )

    return (
        [1, -1]
        if player1_score > player2_score
        else [-1, 1]
        if player1_score < player2_score
        else [0, 0]
    )


star = [
    range(12, 13),
    range(11, 13),
    range(10, 13),
    range(9, 13),
    range(4, 17),
    range(4, 16),
    range(4, 15),
    range(4, 14),
    range(4, 13),
    range(3, 13),
    range(2, 13),
    range(1, 13),
    range(0, 13),
    range(4, 8),
    range(4, 7),
    range(4, 6),
    range(4, 5),
]

spaces = [
    "             ",
    "            ",
    "           ",
    "          ",
    " ",
    "  ",
    "   ",
    "    ",
    "     ",
    "    ",
    "   ",
    "  ",
    " ",
    "          ",
    "           ",
    "            ",
    "             ",
]


def show_board(state):
    pieces = state.to_json()["pieces"]
    player1 = pieces[0]
    player2 = pieces[1]
    for i in range(17):
        level = star[i]
        space = spaces[i]
        print(space, end="")
        for j in range(17):
            if j in level:
                if [i, j] in player1:
                    print(" X", end="")
                elif [i, j] in player2:
                    print(" O", end="")
                else:
                    print(" .", end="")
        print()
