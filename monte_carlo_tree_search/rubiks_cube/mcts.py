from pycuber import Cube
from .help_functions import *
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
    def __init__(self, model: Model, num_iterations_per_move = 20, iteration_limit_depth = 20, init_v_value_threshold = 0.0) -> None:
        self.model = model
        self.num_iterations_per_move = num_iterations_per_move
        self.v_value_threshold = init_v_value_threshold
        print(f"Set v-value threshold to {self.v_value_threshold} (init)")
        self.excluded_cube_state_str = None
        self.iteration_limit_depth = iteration_limit_depth

        self.move_scores_history = []

    def exclude_cube_state_for_next_round(self, cube: Cube) -> None:
        self.excluded_cube_state_str = str(cube)

    def search(self, cube: Cube) -> str:
        """
        Takes in a cube and calculates the best move to take.

        Args:
            cube (Cube): Cube-State at the current moment

        Returns:
            str: One move out of ["F", "R", "L", "U", "D", "B", "F'", "R'", "L'", "U'", "D'", "B'"]
        """
        self.root = TreeNode(cube=cube, parent=None, move_made=None)

        for i in range(self.num_iterations_per_move):
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{self.num_iterations_per_move}")

            node = self.select(self.root)
            score = self.rollout(node.cube)
            self.backpropagate(node, score)

        return self.get_best_move(self.root, 0, print_score=True, append_to_history=True)

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
    
    def get_best_move(self, node: TreeNode, exploration_constant: float, print_score=False, append_to_history=False):
        best_score = float("-inf")
        best_moves = []

        # loop over child nodes of that node
        for child_node in node.children.values():
            if str(child_node.cube) == self.excluded_cube_state_str:
                continue

            # get move score using UCT1 formula
            move_score = child_node.score / child_node.visits + exploration_constant * math.sqrt(math.log(node.visits) / child_node.visits)
            """if print_score:
                print("move score:", move_score)"""

            if move_score > best_score:
                # better move has been found
                best_score = move_score
                best_moves = [child_node]
            elif move_score == best_score:
                # found as good move as already available
                best_moves.append(child_node)

        # return one of the best moves randomly
        sampled_best_move = random.choice(best_moves)
        if print_score:
            print("Score for move:", best_score)
        if append_to_history:
            self.move_scores_history.append(best_score)

        return sampled_best_move

    def rollout(self, cube: Cube) -> float:
        # indices in predicted policy
        MOVES = ["F", "B", "U", "D", "L", "R", "F'", "B'", "U'", "D'", "L'", "R'"]

        predicted_v_value, policy = self.model.predict([flatten_one_hot(cube)], verbose=0)
        predicted_v_value = predicted_v_value[0][0]
        policy = policy[0]

        values = [predicted_v_value]

        CONSIDER_BEST_N_FROM_POLICY = 3
        SLIDING_WINDOW_SIZE = 3
        counter = 0
        if len(self.move_scores_history) >= SLIDING_WINDOW_SIZE:
            trailing_window = self.move_scores_history[-SLIDING_WINDOW_SIZE:]
            threshold = sum(trailing_window) / len(trailing_window)

            if self.v_value_threshold < threshold:
                self.v_value_threshold = threshold
                print(f"Set v-value threshold to {threshold}")

        while predicted_v_value < self.v_value_threshold:
            idxes_to_consider = list(np.argpartition(list(policy), -CONSIDER_BEST_N_FROM_POLICY))[-CONSIDER_BEST_N_FROM_POLICY:]
            values_for_idxes_to_consider = np.partition(list(policy), -CONSIDER_BEST_N_FROM_POLICY)[-CONSIDER_BEST_N_FROM_POLICY:]
            values_for_idxes_to_consider = values_for_idxes_to_consider / values_for_idxes_to_consider.sum()

            all_children = get_children(cube)
            children_to_consider = [all_children[idx] for idx in idxes_to_consider]

            idx_child_to_choose = np.random.choice(idxes_to_consider, p=values_for_idxes_to_consider)
            _, cube = all_children[idx_child_to_choose]

            predicted_v_value, policy = self.model.predict([flatten_one_hot(cube)], verbose=0)
            predicted_v_value = predicted_v_value[0][0]
            policy = policy[0]

            values.append(predicted_v_value)

            counter += 1
            if counter > self.iteration_limit_depth:
                break

        return max(values)

    def backpropagate(self, node: TreeNode, score: float):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent
