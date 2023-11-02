from pycuber import Cube
from .help_functions import get_children_moves, flatten_one_hot, cube_is_solved
from keras.models import Model
from typing import List


class Node():
    def __init__(self, cube: Cube, score: float) -> None:
        self.cube = cube
        self.score = score

        self.is_leaf = False
        self.is_dead = False
        self.parent = None
        self.move_to_get_here = None

    def set_is_leaf(self, value: bool) -> None:
        self.is_leaf = value
    
    def kill(self):
        self.is_dead = True
    
    def set_parent(self, parent):
        self.parent = parent
    
    def set_move_to_get_here(self, move: str):
        self.move_to_get_here = move


class Edge():
    def __init__(self, from_node: Node, to_node: Node, move_used: str) -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.move_used = move_used


class Graph():
    def __init__(self, root: Cube, model: Model, take_best_n: int) -> None:
        self.model = model
        score_root = self.model.predict([flatten_one_hot(root)], verbose=0)[0][0]

        self.root = Node(cube=root, score=score_root)
        self.root.set_is_leaf(True)

        self.nodes = [self.root]
        self.edges = []

        self.take_best_n = take_best_n
        self.counter_until_cut = 0
    
    def add_node(self, cube: Cube, score: float = None) -> None:
        if score is None:
            score = self.model.predict([flatten_one_hot(cube)], verbose=0)[0][0]
        self.nodes.append(Node(cube, score))

    def expand_node(self, node: Node) -> None:
        children_moves = get_children_moves(node.cube)
        children_scores = []
        for cube, move in children_moves:
            score = self.model.predict([flatten_one_hot(cube)], verbose=0)[0][0]
            children_scores.append((cube, move, score))
        children_scores.sort(key=lambda x: x[-1])

        best_n = children_scores[:self.take_best_n]
        for cube, move, score in best_n:
            new_node = Node(cube, score)
            new_node.set_is_leaf(True)
            new_node.set_parent(node)
            new_node.set_move_to_get_here(move)
            self.nodes.append(new_node)

            edge = Edge(from_node=node, to_node=new_node, move_used=move)
            self.edges.append(edge)
        
        node.set_is_leaf(False)

    def expand_layer(self) -> bool:
        all_nodes = [el for el in self.nodes]
        could_expand_node = False
        for node in all_nodes:
            if node.is_leaf and not node.is_dead:
                self.expand_node(node)
                could_expand_node = True

        self.counter_until_cut += 1
        if self.counter_until_cut == 3:
            self.cut_tree_to_best_leaf()
            self.counter_until_cut = 0
        
        return could_expand_node

    def get_best_leaf_score_in_tree(self) -> float:
        best_score = 9999999
        for node in self.nodes[1:]:
            if node.score < best_score and node.is_leaf:
                best_score = node.score
        
        return best_score
    
    def cut_tree_to_best_leaf(self):
        best_score = self.get_best_leaf_score_in_tree()
        for node in self.nodes:
            if node.is_leaf and not node.is_dead:
                if node.score > best_score:
                    node.kill()
        
        nodes_alive = [node for node in self.nodes if node.is_leaf and not node.is_dead]
        if len(nodes_alive) > 1:
            for node in nodes_alive[1:]:
                node.kill()

    def cube_is_solved(self):
        for node in self.nodes:
            if node.is_leaf and not node.is_dead:
                if cube_is_solved(node.cube):
                    return True
        
        return False

    def get_solving_sequence(self) -> List:
        sequence = []
        for node in self.nodes:
            if node.is_leaf and not node.is_dead:
                if cube_is_solved(node.cube):
                    # found solved cube
                    sequence.append(node.move_to_get_here)
                    node_it = node
                    while node_it.parent is not None:
                        node_it = node_it.parent
                        if node_it.move_to_get_here is not None:
                            sequence.append(node_it.move_to_get_here)
                break

        sequence.reverse()
        return sequence


    def __str__(self) -> str:
        leaf_not_dead = [node for node in self.nodes if node.is_leaf and not node.is_dead]
        return f"Graph with {len(self.nodes)} Nodes and {len(self.edges)} Edges (best leaf score: {self.get_best_leaf_score_in_tree()}, {len(leaf_not_dead)} alive leafs)."
