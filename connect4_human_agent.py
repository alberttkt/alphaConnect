from agent import Agent
import time

class Connect4HumanAgent(Agent):
    next_moves = []
    def __init__(self,verbose=False):
        self.verbose = verbose
        
        
    def _play_logic(self, state):
        if self.verbose:
            print(state)
        
        d= {action.to_json(): 0.0 for action in state.actions}
        if not len(self.__class__.next_moves)>0:
            print("No move selected")
            return d
        d[self.__class__.next_moves.pop(0)]=1.0
        return d
    
    @classmethod
    def add_waiting_move(cls,move):
        cls.next_moves.append(move)

    