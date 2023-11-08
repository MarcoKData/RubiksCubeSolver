from pycuber import Cube
import rubiks_utils as r_utils
import data_utils as data
from keras import Model


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
            if move == node.how_did_i_get_here:
                continue

            new_node = BatchDiveNode(id_value=self.next_node_id, cube=child, parent_id=node.id, how_did_i_get_here=move)
            self.nodes.append(new_node)
            self.next_node_id += 1
        node.is_leaf = False

    def score_leafs(self):
        leafs = [node for node in self.nodes if node.is_leaf]
        for node in leafs:
            if r_utils.is_final_cube_state(node.cube):
                return node

            score = self.model(data.flatten_one_hot(node.cube).reshape((1, -1))).numpy()[0][0]
            node.cost_to_go = score

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
        all_leafs = [node for node in self.nodes if node.is_leaf]
        all_leafs.sort(key=lambda node: node.cost_to_go)

        best_n = all_leafs[:n]
        remaining_nodes = [self.root]
        for node in best_n:
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
