from agent import Agent
import random

class Connect4RandomAgent(Agent):
    def __init__(self,verbose=False):
        self.verbose = verbose
        
    def _play_logic(self, state):
        l= len(state.actions)
        actions = list(state.actions)
        random.shuffle(actions)
        d= {action.to_json(): 1.0/l for action in actions}
        return d