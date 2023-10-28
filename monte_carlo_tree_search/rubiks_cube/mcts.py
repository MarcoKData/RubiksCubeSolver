from pycuber import Cube
from help_functions import cube_is_solved, get_children, flatten_one_hot
import random
import math
from keras import Model
import numpy as np


class TreeNode():
    def __init__(self, cube: Cube, parent: Cube, move_made: str):
        self.cube = cube
        self.move_made = move_made

        if cube_is_solved(self.cube):
            self.is_terminal = True
        else:
            self.is_terminal = False

        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.visits = 0
        self.score = 0
        self.children = {}


class MCTS_CUBE():
    def __init__(self, model: Model) -> None:
        self.model = model

    def search(self, cube: Cube) -> str:
        """
        Takes in a cube and calculates the best move to take.

        Args:
            cube (Cube): Cube-State at the current moment

        Returns:
            str: One move out of ["F", "R", "L", "U", "D", "B", "F'", "R'", "L'", "U'", "D'", "B'"]
        """
        self.root = TreeNode(cube=cube, parent=None, move_made=None)

        for _ in range(10):
            node = self.select(self.root)
            score = self.rollout(node.cube)
            self.backpropagate(node, score)

        try:
            return self.get_best_move(self.root, 0)
        except:
            return None
    
    def select(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_move(node, 2)
            else:
                return self.expand(node)

        return node

    def expand(self, node: TreeNode):
        moves_states = get_children(node.cube)
        for move, state in moves_states:
            if str(state) not in node.children:
                new_node = TreeNode(state, node, move)
                node.children[str(state)] = new_node

                if len(moves_states) == len(node.children):
                    node.is_fully_expanded = True

                return new_node
    
    def get_best_move(self, node: TreeNode, exploration_constant: float):
        best_score = float("-inf")
        best_moves = []

        # loop over child nodes of that node
        for child_node in node.children.values():
            # get move score using UCT1 formula
            move_score = child_node.score / child_node.visits + exploration_constant * math.sqrt(math.log(node.visits) / child_node.visits)

            if move_score > best_score:
                # better move has been found
                best_score = move_score
                best_moves = [child_node]
            elif move_score == best_score:
                # found as good move as already available
                best_moves.append(child_node)

        # return one of the best moves randomly
        sampled_best_move = random.choice(best_moves)
        return sampled_best_move

    def rollout(self, cube: Cube) -> float:
        # random moves until terminal state is reached (num_iterations in infinite game)
        predicted_v_value = self.model.predict([flatten_one_hot(cube)])[0][0][0]
        while predicted_v_value < 0.3:
            cube = random.choice(get_children(cube))
            predicted_v_value = self.model.predict([flatten_one_hot(cube)])[0][0][0]
            print("It value:", predicted_v_value)
            if predicted_v_value < 0:
                return predicted_v_value

        return predicted_v_value

    def backpropagate(self, node: TreeNode, score: float):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent
