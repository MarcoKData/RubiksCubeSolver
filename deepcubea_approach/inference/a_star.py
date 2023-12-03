from keras.models import Model
from pycuber import Cube
import rubiks_utils as r_utils
import data_utils as data
from .batch_dive_tree import BatchDiveTree
from typing import List


class Node():
    def __init__(self, cube: Cube, parent=None, move=None) -> None:
        self.cube = cube
        self.parent = parent

        self.g = 0
        self.h = 0
        self.f = 0

        self.move_that_got_me_here = move

    def __eq__(self, __o: object) -> bool:
        return str(self.cube) == str(__o.cube)

    def __str__(self) -> str:
        return str(self.cube)


class AStarGraph():
    def __init__(self) -> None:
        self.open = []
        self.closed = []
    
    def remove_node_from_open(self, node):
        temp = []
        for node_it in self.open:
            if node_it != node:
                temp.append(node_it)
        
        self.open = temp



def solve_with_batch_dive(start_cube: Cube, model: Model, batch_depth: int = 3, prune_to_best_n: int = 3, width_per_layer: int = 5,
                          max_num_iterations: int = 999_999, print_stuff: bool = False) -> List:
    tree = BatchDiveTree(model)
    tree.add_root(start_cube)

    is_solved = False
    it_counter = 0
    while not is_solved:
        if print_stuff:
            print(f"{it_counter + 1} (max {max_num_iterations})...")
        if it_counter >= max_num_iterations:
            break

        for i in range(batch_depth):
            if print_stuff:
                print(f"Expanding {i + 1}/{batch_depth}!")
            is_solved = tree.expand_layer(width_to_expand=width_per_layer)
            if is_solved:
                break
        
        if print_stuff:
            print("Scoring leafs...")
        best_leaf = tree.get_best_leaf()
        if is_solved:
            return tree.get_path_to_node(best_leaf)

        if print_stuff:
            print(f"Best leaf's score: {best_leaf.cost_to_go}")
            print(f"Best leaf's cube:\n{best_leaf.cube}")
        tree.prune_tree_to_best_n_leafs(n=prune_to_best_n)

        it_counter += 1

    return None


def solve_with_a_star(start_cube: Cube, model: Model, max_num_iterations: int = 999_999) -> List:
    graph = AStarGraph()
    graph.open.append(Node(cube=start_cube, parent=None, move=None))

    it_counter = 0
    while len(graph.open) > 0:
        it_counter += 1
        print(f"{it_counter}...")
        if it_counter >= max_num_iterations:
            return None

        # get the current node
        current_node = graph.open[0]
        for node in graph.open:
            if node.f < current_node.f:
                current_node = node

        graph.remove_node_from_open(current_node)
        graph.closed.append(current_node)

        if r_utils.is_final_cube_state(current_node.cube):
            # FOUND GOAL!!
            sequence = []
            while current_node.move_that_got_me_here is not None:
                sequence.append(current_node.move_that_got_me_here)
                current_node = current_node.parent
            sequence.reverse()

            return sequence
        
        # did not find the goal --> generate children
        children = r_utils.get_children(current_node.cube)
        for move, child_cube in children:
            node_child = Node(cube=child_cube, parent=current_node, move=move)
            if node_child in graph.closed:
                continue

            node_child.g = current_node.g + 1
            cube_to_pred = data.flatten_one_hot(node_child.cube).reshape((1, -1))
            node_child.h = model(cube_to_pred).numpy()[0][0]
            node_child.f = node_child.g + node_child.h

            for node in graph.open:
                if node == node_child and node_child.g > node.g:
                    continue
            
            graph.open.append(node_child)
    
    return None
