from help_functions import *
from model import build_model
import os
import numpy as np


MAX_STEPS = 50

load_path = os.path.join(".", "models", "model.h5")
model = build_model()
model.load_weights(load_path)

length_solving_seqs = []
for n_shuffles in range(1, 10):
    print(f"Testing n_shuffles = {n_shuffles}")
    cube = generate_sample_cube(n_shuffles=n_shuffles)

    try_counter = 0
    successfully_solved = False

    while successfully_solved is False and try_counter < 5:
        steps = []
        step_counter = 0
        while not cube_is_solved(cube) and step_counter < MAX_STEPS:
            # print(f"Step {step_counter + 1} (max {MAX_STEPS})...")
            step_counter += 1
            flattened_cube = flatten_one_hot(cube)
            v, p = model.predict([flattened_cube], verbose=0)

            action = inv_action_map[np.argmax(p)]
            steps.append(action)
            cube(action)

        if step_counter < MAX_STEPS:
            print(f"Solved with Sequence {steps}!")
            successfully_solved = True
        else:
            print("FAILURE!!")
        
        if len(steps) > 0:
            try_counter += 1
    
    if successfully_solved is True:
        length_solving_seqs.append(len(steps))
    else:
        break

print(length_solving_seqs)
max_length_solving_seq = np.array(length_solving_seqs).max()
print(f"Successfully solved cubes up to max_length_solving_seq = {max_length_solving_seq}")
