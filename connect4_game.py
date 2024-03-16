from connect4_agent import Connect4Agent
from game import *
from connect4_helper import *

State = ConnectState


class Connect4Game():
    
    def __init__(self,agent1:Connect4Agent,agent2:Connect4Agent,state:State=State.sample_initial_state()):
        self.agent1 = agent1
        self.agent2 = agent2
        self.state = state


    def reset(self):
        return Connect4Game(self.agent1,self.agent2)
    
    def has_ended(self):
        return self.state.has_ended
    
    def play(self):
        if self.state.player == 0:
            moves_proba_dict = self.agent1.play(self.state)
        else:
            moves_proba_dict = self.agent2.play(self.state)

        idx2action = {action.to_json():action for action in self.state.actions}
        print(moves_proba_dict)
        for move in sorted(moves_proba_dict, key=moves_proba_dict.get, reverse=True):
            if int(move) in idx2action:
                return Connect4Game(self.agent1,self.agent2,idx2action[int(move)].sample_next_state())
        print("No valid moves")
        return self
    
    def get_value(self,column,row):
        return self.state.get_tensors()[0][row][column]

        
    
    def show(self):
        show_board(self.state)