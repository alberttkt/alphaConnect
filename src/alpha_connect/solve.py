from functools import reduce
import random
from game import Connect4State


WIDTH = 7
HEIGHT = 6

State = Connect4State


class Position:
    def __init__(self, state) -> None:
        self.state = state

    def can_play(self, column) -> bool:
        return self.state.get_tensors()[0][-1][column] == -1

    def play(self, column) -> None:
        for i in self.state.actions:
            if i.to_json() == column:
                self.state = i.sample_next_state()
                return

    def is_winning_move(self, column) -> bool:
        for i in self.state.actions:
            if i.to_json() == column:
                return i.sample_next_state().has_ended
        return False

    def nb_moves(self) -> int:
        count = 0
        for i in range(HEIGHT):
            count += reduce(
                lambda x, y: x + y,
                [1 if j != -1 else 0 for j in self.state.get_tensors()[0][i]],
            )
        return count

    def __hash__(self) -> int:
        return hash(str(self.state.get_tensors()[0]))

    def to_bitboards(self) -> (int, int):
        player1 = int(
            "0b"
            + "".join(
                [
                    str(1 if j == 0 else 0)
                    for i in self.state.get_tensors()[0]
                    for j in i
                ]
            ),
            2,
        )
        player2 = int(
            "0b"
            + "".join(
                [
                    str(1 if j == 1 else 0)
                    for i in self.state.get_tensors()[0]
                    for j in i
                ]
            ),
            2,
        )
        return player1, player2

    @classmethod
    def tensor_from_bitboards(cls, player1, player2):
        a = [[-1 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        for i in range(HEIGHT - 1, -1, -1):
            for j in range(WIDTH - 1, -1, -1):
                if player1 & 1:
                    a[i][j] = 0
                elif player2 & 1:
                    a[i][j] = 1
                player1 >>= 1
                player2 >>= 1
        return [a]


class Solver:
    def __init__(self):
        self.memo = {}  # A simple dictionary can be used as a transposition table
        self.column_order = [
            3,
            4,
            2,
            5,
            1,
            6,
            0,
        ]  # For WIDTH=7, adjust accordingly for other sizes

    def negamax(self, position, alpha=-HEIGHT * WIDTH, beta=HEIGHT * WIDTH):
        if position.nb_moves() == WIDTH * HEIGHT:  # Check for draw game
            return 0, -1

        for x in self.column_order:  # Check if current player can win next move
            if position.can_play(x) and position.is_winning_move(x):
                return (WIDTH * HEIGHT + 1 - position.nb_moves()) // 2, x

        if (
            position.to_bitboards() in self.memo
        ):  # Look up the position in the transposition table
            return self.memo[position.to_bitboards()]

        best_move = -1
        max_score = (
            WIDTH * HEIGHT - 1 - position.nb_moves()
        ) // 2  # Upper bound of our score
        if beta > max_score:
            beta = max_score  # Adjust beta if necessary
            if alpha >= beta:
                return (
                    beta,
                    -1,
                )  # Prune the exploration if the alpha-beta window is empty

        for x in (
            self.column_order
        ):  # Compute the score of all possible next moves and keep the best one
            if position.can_play(x):
                position_copy = Position(position.state)
                position_copy.play(x)  # Play move x

                # check that this move is not a losing move
                for i in self.column_order:
                    if position_copy.can_play(i) and position_copy.is_winning_move(i):
                        continue

                temp_score, _ = self.negamax(
                    position_copy, -beta, -alpha
                )  # Negate the score and reverse the alpha-beta window
                score = -temp_score

                if score >= beta:
                    self.memo[position.to_bitboards()] = beta, x
                    return (
                        score,
                        x,
                    )  # Prune the exploration if we find a move better than what we were looking for
                if score > alpha:
                    best_move = x
                    alpha = score  # Adjust alpha if necessary

        self.memo[position.to_bitboards()] = alpha, best_move
        return alpha, best_move


def show_board(state):
    a = state.get_tensors()[0]
    for i in range(5, -1, -1):
        for j in range(7):
            c = "." if a[i][j] == -1 else "X" if a[i][j] == 1 else "O"
            print(c, end=" ")
        print()


State = Connect4State

state = State.sample_initial_state()
good_game = False
while not good_game:
    state = State.sample_initial_state()
    nb_moves = 15
    for _ in range(nb_moves):
        if state.has_ended:
            break
        state = random.choice(state.actions).sample_next_state()
    if state.has_ended:
        continue
    good_game = True

show_board(state)

position = Position(state)
p1, p2 = position.to_bitboards()
print(p1, p2)
print(Position.tensor_from_bitboards(p1, p2))
print(state.get_tensors()[0])
