
from abc import abstractmethod
import random


class Connect4Agent:
  
    @abstractmethod
    def play(self, state) -> dict:
        pass
        
    
    def sample_move(self, state):
        moves_proba_dict = self.play(state)
        threshold = random.random()
        cumulative = 0
        for move in moves_proba_dict:
            cumulative += moves_proba_dict[move]
            if cumulative > threshold:
                for action in state.actions:
                    if action.to_json() == int(move):
                        return action
        