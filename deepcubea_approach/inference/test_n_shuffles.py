import numpy as np
from keras.models import Model
from typing import Tuple
import data_utils as data
from .a_star import solve_with_batch_dive


def get_max_n_shuffles_solved_plus_solution_lens(model: Model) -> Tuple:
    solved_n_shuffles = []
    solution_lens = []
    n_shuffles = 1
    while True:
        print(f"Testing n_shuffles: {n_shuffles}")
        cube = data.get_single_scrambled_cube(num_scrambles=n_shuffles)

        # sequence = inference.solve_with_a_star(cube, model, max_num_iterations=30)
        sequence = solve_with_batch_dive(
            start_cube=cube,
            model=model,
            batch_depth=4,
            prune_to_best_n=2,
            width_per_layer=5,
            max_num_iterations=3
        )
        if sequence is None:
            break

        solved_n_shuffles.append(n_shuffles)
        solution_lens.append(len(sequence))
        n_shuffles += 1
        print(f"Sequence: {sequence}")

    solution_lens = np.array(solution_lens)

    return n_shuffles - 1, int(solution_lens.max())
