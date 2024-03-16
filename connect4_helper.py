import sys
from game import *

def show_board(state,file=sys.stdout):
    a = state.get_tensors()[0]
    for i in range(5,-1,-1):
        for j in range(7):
            c = "." if a[i][j] == -1 else "X" if a[i][j] == 1 else "O"
            print(c, end=' ', file=file)
        print(file=file)