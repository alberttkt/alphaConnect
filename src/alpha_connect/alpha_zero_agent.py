import math

import numpy as np

from .agent import Agent
from .random_agent import RandomAgent


class MCTSNode:
    c_puct = 1.0

    def __init__(self, state, player, prior, action=None, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.prior = prior
        self.children = []
        self.value_sum = 0
        self.visit_count = 0
        self.action = action  # Store the action that led to this node. uselfull to give the final move

    def select_child(self):
        # Use the UCB1 formula to select a child node

        scores = {c: self.ucb_score(c) for c in self.children}
        return max(scores, key=scores.get)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, child):
        if self.visit_count == 0:
            return child.prior  # if the node has not been visited, trust the network

        prior_score = (
            MCTSNode.c_puct
            * child.prior
            * math.sqrt(self.visit_count)
            / (child.visit_count + 1)
        )
        value_score = -child.value()
        return value_score + prior_score

    def expand_node(self, player, probs):
        for action, prob in probs.items():
            new_state = action.sample_next_state()
            child_node = MCTSNode(
                state=new_state,
                player=1 - player,
                action=action,
                prior=prob,
                parent=self,
            )
            self.children.append(child_node)

    def get_distribution(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        action_to_visit_counts = {
            child.action: child.visit_count for child in self.children
        }

        if temperature == 0:
            return {max(action_to_visit_counts, key=action_to_visit_counts.get): 1.0}
        elif temperature == float("inf"):
            return {
                action: 1.0 / len(action_to_visit_counts)
                for action in action_to_visit_counts
            }
        else:
            counts = np.array(
                [
                    count ** (1 / temperature)
                    for count in action_to_visit_counts.values()
                ]
            )
            probs = counts / counts.sum()
            a = {
                action.to_json(): prob
                for action, prob in zip(action_to_visit_counts.keys(), probs)
            }

            return a

    def update(self, value, player):
        self.visit_count += 1
        self.value_sum += value if player == self.player else -value

    def expanded(self):
        return self.children == []

    def has_parent(self):
        return self.parent is not None

    def is_leaf(self):
        return self.children == []

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return " Prior: {} Count: {} Value: {} {}".format(
            prior, self.visit_count, self.value(), self.state.to_json()
        )


class AlphaZeroAgent(Agent):
    def __init__(
        self,
        inner_agent: Agent = RandomAgent(),
        iterations=200,
        temperature=1,
    ):
        self.iterations = iterations
        self.inner_agent = inner_agent
        self.temperature = temperature

    def _play_logic(self, initial_state):
        root_node = MCTSNode(initial_state, initial_state.player, 0)

        move_probabilities, value = self.inner_agent.play(initial_state)
        root_node.expand_node(initial_state.player, move_probabilities)

        for _ in range(self.iterations):
            node = root_node
            state = initial_state

            # Selection
            while not node.is_leaf():
                node = node.select_child()

            state = node.state

            # Simulation
            if not state.has_ended:
                move_probabilities, value = self.inner_agent.play(state)
                node.expand_node(state.player, move_probabilities)
                player = state.player
            else:
                value = state.reward[initial_state.player]
                player = initial_state.player

            # Backpropagation
            while node.has_parent():
                node.update(value, player)
                node = node.parent
            root_node.update(value, player)

        move_probabilities = root_node.get_distribution(self.temperature)

        return move_probabilities, 1
