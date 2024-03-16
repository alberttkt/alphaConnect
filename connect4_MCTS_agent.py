import math
import random
from collections import defaultdict

from connect4_agent import Connect4Agent
from connect4_random_agent import Connect4RandomAgent


depth_factor = 0.1

class MCTSNode:
    def __init__(self, state, action=None, parent=None,):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.untried_actions = list(self.state.actions)
        self.action = action  # Store the action that led to this node. uselfull to give the final move

    def select_child(self):
        # Use the UCB1 formula to select a child node
        return max(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))
    
    def add_child(self, action):
        new_state = action.sample_next_state()
        child_node = MCTSNode(state=new_state, action=action, parent=self)

        self.untried_actions.remove(action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

class Connect4MCTSAgent(Connect4Agent):
    def __init__(self, iterations=10000,inner_agent:Connect4Agent=Connect4RandomAgent()):
        self.iterations = iterations
        self.inner_agent = inner_agent

    
    def play(self, initial_state):
        root_node = MCTSNode(state=initial_state)

        for _ in range(self.iterations):
            node = root_node
            state = initial_state

            choosed_action = None

            # Selection
            while node.untried_actions == [] and node.children != []:
                node = node.select_child()
                choosed_action = node.action.to_json()
                state = node.state

            # Expansion
            if node.untried_actions != []:
                action = random.choice(node.untried_actions)
                if choosed_action is None:
                    choosed_action = action.to_json()
                state = action.sample_next_state()
                node = node.add_child(action)

            # Simulation
            game_length = node.depth
            while not state.has_ended:
                action = self.inner_agent.sample_move(state)
                state = action.sample_next_state()
                game_length += 1

            

            reward = int(state.reward[initial_state.player])*(depth_factor**(game_length-1))

            if game_length == 2 and int(state.reward[initial_state.player]) == -1:
                reward -= 100
                
            if game_length == 1 and int(state.reward[initial_state.player]) == 1:
                reward += 100

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        move_probabilities = {int(child.action.to_json()): child.wins/child.visits for child in root_node.children}
        # softmax
        total = sum([math.exp(prob) for prob in move_probabilities.values()])
        move_probabilities = {action: math.exp(prob) / total for action, prob in move_probabilities.items()}

        return move_probabilities
