from train import train
from inference import *
import os


"""train(
    n_epochs=1,
    n_iterations=20,
    n_samples=5,
    save_weights=False,
    load_weights=False,
    load_path=os.path.join(".", "models", "model.h5"),
    save_path=os.path.join(".", "models", "model-save-path.h5")
)"""

load_path = os.path.join(".", "models", "model-4.h5")

"""max_length_solving_seq, max_solved_n_shuffles = inference(
    load_path=load_path,
    max_steps_until_solved=50
)

print("Successfully solved cubes up to...")
print(f"max_length_solving_seq = {max_length_solving_seq}")
print(f"max_solved_n_shuffles = {max_solved_n_shuffles}")"""


steps, solved = single_random_cube_solve(n_shuffles=5, model_path=load_path)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve cube!")
