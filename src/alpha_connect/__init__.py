__version__ = "0.1.0"

from .trainer import Trainer
from .alpha_zero_agent import AlphaZeroAgent
from .neural_net_agent import NeuralNetAgent
from .MCTS_agent import MCTSAgent
from .random_agent import RandomAgent
from .human_agent import HumanAgent
from .one_move_ahead_agent import OneMoveAheadAgent
from .game import Game
from .helper import *
from .model import AlphaZeroModelConnect4, MyDataset
from .game_choice import GameChoice
