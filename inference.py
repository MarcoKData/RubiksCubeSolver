from help_functions import *
from model import build_model
import os
import numpy as np


N_SHUFFLES = 3
MAX_STEPS = 50

load_path = os.path.join(".", "models", "model.h5")
model = build_model()
model.load_weights(load_path)

cube = generate_sample_cube(n_shuffles=N_SHUFFLES)

steps = []
step_counter = 1
while not cube_is_solved(cube) and step_counter < MAX_STEPS:
    print(f"Step {step_counter} (max {MAX_STEPS})...")
    step_counter += 1
    flattened_cube = flatten_one_hot(cube)
    v, p = model.predict([flattened_cube], verbose=0)

    action = inv_action_map[np.argmax(p)]
    steps.append(action)
    cube(action)

if step_counter < MAX_STEPS:
    print(f"Solved with Sequence {steps}!")
else:
    print("FAILURE!!")
