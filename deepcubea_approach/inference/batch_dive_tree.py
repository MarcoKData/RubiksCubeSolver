from pycuber import Cube
import rubiks_utils as r_utils
import data_utils as data
from keras import Model
import numpy as np


class BatchDiveNode():
    def __init__(self, id_value: int, cube: Cube, parent_id: int, how_did_i_get_here: str) -> None:
        self.id = id_value
        self.cube = cube
        self.parent_id = parent_id
        self.is_leaf = True
        self.how_did_i_get_here = how_did_i_get_here
        self.cost_to_go = 999_999
        self.is_root = False
    
    def __eq__(self, __o: object) -> bool:
        return self.id == __o.id


class BatchDiveTree():
    def __init__(self, model: Model) -> None:
        self.next_node_id = 0
        self.nodes = []
        self.model = model
        self.root = None
        self.former_best_leafs = []

        cube = Cube()
        first_children = r_utils.get_children(cube)
        self.str_cubes_one_shuffle = [str(c) for _, c in first_children]
        self.str_cubes_two_shuffles = []
        for _, cube_one_shuffle in first_children:
            self.str_cubes_two_shuffles += [str(c) for _, c in r_utils.get_children(cube_one_shuffle)]

    def add_root(self, cube: BatchDiveNode):
        node = BatchDiveNode(id_value=self.next_node_id, cube=cube, parent_id=None, how_did_i_get_here=None)
        node.is_root = True
        self.nodes.append(node)
        self.root = node

    def expand_layer(self):
        leafs = [node for node in self.nodes if node.is_leaf]
        for node in leafs:
            if r_utils.is_final_cube_state(node.cube):
                print("is final")
                return True

            self.expand_node(node)

        return False

    def expand_node(self, node: BatchDiveNode):
        children = r_utils.get_children(node.cube)
        for move, child in children:
            if move == r_utils.get_reversed_action(node.how_did_i_get_here):
                continue

            new_node = BatchDiveNode(id_value=self.next_node_id, cube=child, parent_id=node.id, how_did_i_get_here=move)
            self.nodes.append(new_node)
            self.next_node_id += 1
        node.is_leaf = False

    def score_leafs(self, one_is_solved=False):
        leafs = [node for node in self.nodes if node.is_leaf]
        if one_is_solved:
            for leaf in leafs:
                if r_utils.is_final_cube_state(leaf.cube):
                    return leaf

        flattened_leafs = np.array([data.flatten_one_hot(node.cube) for node in leafs])
        preds = self.model.predict(flattened_leafs, verbose=0)
        for i in range(len(leafs)):
            c = leafs[i].cube
            if str(c) in self.str_cubes_one_shuffle:
                cost_to_go_value = 1
            elif str(c) in self.str_cubes_two_shuffles:
                cost_to_go_value = 2
            else:
                cost_to_go_value = preds[i][0] + np.random.rand() / 1_000

            leafs[i].cost_to_go = cost_to_go_value

        leafs.sort(key=lambda leaf: leaf.cost_to_go)
        best_leaf = leafs[0]

        return best_leaf
    
    def get_node_by_id(self, id_value: int):
        for node in self.nodes:
            if node.id == id_value:
                return node
        
        return None

    def get_path_to_node(self, node: BatchDiveNode):
        path = []
        while not node.is_root:
            path.append(node.how_did_i_get_here)
            node = self.get_node_by_id(node.parent_id)
        path.reverse()

        return path
    
    def prune_tree_to_best_n_leafs(self, n: int):
        all_leafs = [node for node in self.nodes if node.is_leaf and node not in self.former_best_leafs]
        all_leafs.sort(key=lambda node: node.cost_to_go)

        best_n = all_leafs[:n]
        remaining_nodes = [self.root]
        for node in best_n:
            self.former_best_leafs.append(node)
            node_it = node
            node_path = [node_it]

            while node_it.parent_id is not None:
                node_it = self.get_node_by_id(node_it.parent_id)
                if node_it != self.root:
                    node_path.append(node_it)

            remaining_nodes += node_path
        
        self.nodes = remaining_nodes

    def __str__(self) -> str:
        all_leafs = [node for node in self.nodes if node.is_leaf]
        return f"Tree with {len(self.nodes)} Nodes and {len(all_leafs)} Leafs."
