import math
import random
from collections import defaultdict

from agent import Agent
from connect4_random_agent import Connect4RandomAgent

depth_factor = 0.1


class MCTSNode:
    def __init__(self, state, action=None, parent=None, value=0.5):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.untried_actions = list(self.state.actions)
        self.action = action  # Store the action that led to this node. uselfull to give the final move
        self.value = value

    def select_child(self, cpuct):
        # Use the UCB1 formula to select a child node, including cpuct for exploration
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            # Calculating the UCB1 value
            ucb1_value = child.wins / child.visits + cpuct * math.sqrt(math.log(self.visits) / child.visits)
            if ucb1_value > best_value:
                best_value = ucb1_value
                best_child = child
        return best_child

    def add_child(self, action, value):
        new_state = action.sample_next_state()
        child_node = MCTSNode(state=new_state, action=action, parent=self, value=value)

        self.untried_actions.remove(action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result


class Connect4AlphaZeroAgent(Agent):
    def __init__(self, iterations=10000, cpuct=2.0, inner_agent: Agent = Connect4RandomAgent()):
        self.iterations = iterations
        self.cpuct = cpuct
        self.inner_agent = inner_agent

    def _play_logic(self, initial_state):
        root_node = MCTSNode(state=initial_state)

        for _ in range(self.iterations):
            node = root_node
            state = initial_state

            # Selection
            while node.untried_actions == [] and node.children != []:
                node = node.select_child(self.cpuct)
                state = node.state

            # Expansion
            if node.untried_actions != []:
                action_probabilities, value = self.inner_agent.play(state)
                action_probabilities = {k: v for k, v in action_probabilities.items() if k in node.untried_actions}
                action = max(action_probabilities, key=action_probabilities.get)

                state = action.sample_next_state()
                node = node.add_child(action, value)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += value  # Assuming value is the reward
                value = -value  # Negate the value for the opponent
                node = node.parent

        # Compute move probabilities based on visits
        move_probabilities = {child.action.to_json(): child.visits for child in root_node.children}
        total_visits = sum(move_probabilities.values())
        move_probabilities = {k: v / total_visits for k, v in move_probabilities.items()}

        return move_probabilities, 1
