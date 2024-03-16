from connect4_agent import Connect4Agent
import random

class Connect4RandomAgent(Connect4Agent):
    def __init__(self,verbose=False):
        self.verbose = verbose
        
    def play(self, state):
        l= len(state.actions)
        actions = list(state.actions)
        random.shuffle(actions)
        d= {action.to_json(): 1.0/l for action in actions}
        return d