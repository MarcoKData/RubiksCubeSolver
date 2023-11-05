from keras.models import Model
import numpy as np
from pycuber import Cube
import rubiks_utils as r_utils
import data_utils as data
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
            node_child.h = model.predict(cube_to_pred, verbose=0)[0][0]
            node_child.f = node_child.g + node_child.h

            for node in graph.open:
                if node == node_child and node_child.g > node.g:
                    continue
            
            graph.open.append(node_child)
    
    return None
