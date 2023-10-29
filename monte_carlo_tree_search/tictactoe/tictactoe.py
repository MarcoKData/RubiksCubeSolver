"""
    Monte-Carlo-Tree-Search + NN for TicTacToe
"""
from copy import deepcopy
from mcts import *


# TicTacToe Board class
class Board():
    def __init__(self, board=None):
        # define Players
        self.player_1 = "x"
        self.player_2 = "o"
        self.empty_square = "."

        # define board position
        self.position = {}

        # init (reset) board
        self.init_board()

        # create copy of previous board-state if available
        if board is not None:
            self.__dict__ = deepcopy(board.__dict__)
    
    # init or reset board
    def init_board(self):
        # loop over board rows
        for row in range(3):
            # loop over board cols
            for col in range(3):
                # set every board square to empty square
                self.position[row, col] = self.empty_square
    
    def make_move(self, row, col):
        # create new board instance
        board = Board(self)

        # make move
        board.position[row, col] = self.player_1

        # switch players
        (board.player_1, board.player_2) = (board.player_2, board.player_1)

        # return new board state
        return board
    
    def is_draw(self):
        for row, col in self.position:
            if self.position[row, col] == self.empty_square:
                return False
        
        # by default return True (draw)
        return True
    
    def is_win(self):
        # vertrical sequence detection
        for col in range(3):
            winning_sequence = []
            for row in range(3):
                if self.position[row, col] == self.player_2:
                    winning_sequence.append((row, col))
                
                if len(winning_sequence) == 3:
                    return True

        # horizontal sequence detection
        for row in range(3):
            winning_sequence = []
            for col in range(3):
                if self.position[row, col] == self.player_2:
                    winning_sequence.append((row, col))
                
                if len(winning_sequence) == 3:
                    return True

        # 1st diagonal sequence detection
        winning_sequence = []
        for row in range(3):
            col = row
            if self.position[row, col] == self.player_2:
                winning_sequence.append((row, col))

            if len(winning_sequence) == 3:
                return True

        # 1st diagonal sequence detection
        winning_sequence = []
        for row in range(3):
            col = 3 - row - 1
            if self.position[row, col] == self.player_2:
                winning_sequence.append((row, col))

            if len(winning_sequence) == 3:
                return True

        # return False by default
        return False
    
    # generate legal moves to play in the current position
    def generate_states(self):
        # define states list (move list - list of available actions)
        actions = []

        for row in range(3):
            for col in range(3):
                if self.position[row, col] == self.empty_square:
                    actions.append(self.make_move(row, col))
        
        return actions
    
    def game_loop(self):
        print("Habe die Ehre ;)\nMoves als col/row bzw. x/y...\n(exit to quit)")
        
        # create MCTS instance
        mcts = MCTS()

        while True:
            print(self)
            user_input = input("> ")

            if user_input == "exit":
                break

            # skip empty input
            if user_input == "":
                continue

            # parse user input (format x, y / col, row)
            try:
                col = int(user_input.split(",")[0].strip()) - 1
                row = int(user_input.split(",")[1].strip()) - 1

                # check if move is legal
                if self.position[row, col] != self.empty_square:
                    print("Illegal Move!")
                    continue

                # make the move
                self = self.make_move(row, col)
                print(self)

                # make AI move here
                best_move = mcts.search(self)
                self = best_move.board

                if self.is_win():
                    print(f"{self.player_2} won the game!")
                    break
                elif self.is_draw():
                    print("Draw!")
                    break

            except Exception as e:
                print(e)
                print("Illegal Command!")
 
    def __str__(self):
        board_str = ""

        for row in range(3):
            # loop over board cols
            for col in range(3):
                board_str += f" {self.position[row, col]} "

            board_str += "\n"
        
        if self.player_1 == "x":
            board_str = "\n----------------\n\"x\" to move:\n----------------\n\n" + board_str
        elif self.player_1 == "o":
            board_str = "\n----------------\n\"o\" to move:\n----------------\n\n" + board_str

        return board_str


if __name__ == "__main__":
    board = Board()
    board.game_loop()
