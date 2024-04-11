import math
import random
from collections import defaultdict

from src.alpha_connect.agent import Agent
from src.alpha_connect.connect4_random_agent import Connect4RandomAgent


class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.action = action  # Store the action that led to this node. uselfull to give the final move

    def select_child(self):
        # Use the UCB1 formula to select a child node
        visits = (
            self.visits
            if self.visits != 0
            else sum(c.visits for c in self.children) + 1
        )

        return max(
            self.children,
            key=lambda c: float("inf")
            if c.visits == 0
            else c.wins / c.visits + 2 * math.sqrt(math.log(visits) / c.visits),
        )

    def expend_node(self):
        for action in self.state.actions:
            new_state = action.sample_next_state()
            child_node = MCTSNode(state=new_state, action=action, parent=self)
            self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += 1 if result == 1 else 0

    def is_leaf(self):
        return self.children == []

    def has_parent(self):
        return self.parent is not None


class Connect4MCTSAgent(Agent):
    def __init__(self, iterations=10000, inner_agent: Agent = Connect4RandomAgent()):
        self.iterations = iterations
        self.inner_agent = inner_agent
        self.random_agent = Connect4RandomAgent()

    def _play_logic(self, initial_state):
        root_node = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = root_node
            state = initial_state

            # Selection
            while not node.is_leaf():
                node = node.select_child()

            state = node.state

            # Expansion
            node.expend_node()

            # Simulation
            while not state.has_ended:
                action = self.random_agent.sample_move(state)
                state = action.sample_next_state()

            reward = int(state.reward[initial_state.player])

            # Backpropagation
            while node.has_parent():
                node.update(reward)
                node = node.parent

        move_probabilities = {
            int(child.action.to_json()): child.wins / child.visits
            for child in root_node.children
        }

        return move_probabilities, 1
