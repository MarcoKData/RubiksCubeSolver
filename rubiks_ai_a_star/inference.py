from .model import build_model
from .help_functions import *
from .graph import Graph


def solve_random_cube(model_weights_path, shuffle_length=15):
    # Idea: traverse through tree until you find BETTER score than current one
    model = build_model()
    model.load_weights(model_weights_path)

    cube = generate_sample_cube(n_shuffles=shuffle_length)
    print("Randomly shuffled cube:")
    print(cube)

    graph = Graph(root=cube, model=model, take_best_n=2, max_living_leafs=3)

    step_counter = 0
    while not cube_is_solved(cube):
        print(f"{step_counter + 1}...")
        could_expand_node = graph.expand_layer()
        print(graph)
        if graph.cube_is_solved() or not could_expand_node:
            break

        step_counter += 1

    if could_expand_node:
        print("\n\nCube is solved!")
        solving_sequence = graph.get_solving_sequence()
        print(f"Solving sequence: {solving_sequence}")
    else:
        print("\n\nCould not solve Cube!")
