from .help_functions import *
from .model import build_model
import numpy as np


def solve_cube(cube, model):
    max_steps_until_solved = 50
    step_counter = 0
    steps = []
    solved = False
    while not cube_is_solved(cube) and step_counter < max_steps_until_solved:
        step_counter += 1
        flattened_cube = flatten_one_hot(cube)
        _, p = model.predict([flattened_cube], verbose=0)

        action = inv_action_map[np.argmax(p)]
        steps.append(action)
        cube(action)
    
    if step_counter < max_steps_until_solved:
        solved = True
    
    if solved:
        return steps, solved
    else:
        return [], solved


def single_random_cube_solve(n_shuffles, model_path):
    cube = generate_sample_cube(n_shuffles=n_shuffles)

    model = build_model()
    model.load_weights(model_path)

    steps, solved = solve_cube(cube, model)

    return steps, solved


def inference(load_path, max_steps_until_solved):
    model = build_model()
    model.load_weights(load_path)

    length_solving_seqs = []
    solved_n_shuffles = []
    n_shuffles = 0
    while True:
        n_shuffles += 1
        print(f"Testing n_shuffles = {n_shuffles}")

        try_counter = 0
        successfully_solved = False
        cube = generate_sample_cube(n_shuffles=n_shuffles)

        while successfully_solved is False and try_counter < 5:
            steps = []
            step_counter = 0
            while not cube_is_solved(cube) and step_counter < max_steps_until_solved:
                step_counter += 1
                flattened_cube = flatten_one_hot(cube)
                _, p = model.predict([flattened_cube], verbose=0)

                action = inv_action_map[np.argmax(p)]
                steps.append(action)
                cube(action)

            if step_counter < max_steps_until_solved:
                print(f"Solved with Sequence {steps}!")
                successfully_solved = True
            else:
                print("FAILURE!!")
            
            if len(steps) > 0:
                try_counter += 1
        
        if successfully_solved is True:
            length_solving_seqs.append(len(steps))
            solved_n_shuffles.append(n_shuffles)
        else:
            break

    max_length_solving_seq = np.array(length_solving_seqs).max()
    max_solved_n_shuffles = np.array(solved_n_shuffles).max()

    return max_length_solving_seq, max_solved_n_shuffles
