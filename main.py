from train import train
from inference import *
import os


load_path = os.path.join(".", "models", "model-5.h5")

# train the model
"""train(
    n_epochs=1,
    n_iterations=20,
    n_samples=5,
    save_weights=False,
    load_weights=False,
    load_path=os.path.join(".", "models", "model.h5"),
    save_path=os.path.join(".", "models", "model-save-path.h5")
)"""


# check how far the model can get (n_shuffles) with random cubes
"""max_length_solving_seq, max_solved_n_shuffles = inference(
    load_path=load_path,
    max_steps_until_solved=50
)

print("Successfully solved cubes up to...")
print(f"max_length_solving_seq = {max_length_solving_seq}")
print(f"max_solved_n_shuffles = {max_solved_n_shuffles}")"""


# solve single random cube with specified n_shuffles
"""steps, solved = single_random_cube_solve(n_shuffles=5, model_path=load_path)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve cube!")"""


# solve specific cube (user defined)
f = ["yellow", "yellow", "yellow", "green", "green", "white", "green", "green", "white"]
r = ["red", "blue", "blue", "orange", "orange", "white", "orange", "orange", "white"]
l = ["yellow", "green", "orange", "yellow", "red", "white", "red", "red", "white"]
b = ["yellow", "yellow", "red", "blue", "blue", "red", "blue", "blue", "white"]
u = ["blue", "orange", "orange", "red", "yellow", "yellow", "green", "green", "green"]
d = ["orange", "orange", "blue", "white", "white", "blue", "green", "red", "red"]

cube = make_cube_from_flattened_sides(f, r, l, b, u, d)
print(cube)

model = build_model()
model.load_weights(load_path)

steps, solved = solve_cube(cube, model)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve!")
