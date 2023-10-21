from train import train
from inference import inference
import os


"""train(
    n_epochs=50,
    n_iterations=20,
    n_samples=100,
    save_weights=False,
    load_weights=True
)"""

load_path = os.path.join(".", "models", "model-N6-cubes.h5")

max_length_solving_seq, max_solved_n_shuffles = inference(
    load_path=load_path,
    max_steps_until_solved=50
)

print("Successfully solved cubes up to...")
print(f"max_length_solving_seq = {max_length_solving_seq}")
print(f"max_solved_n_shuffles = {max_solved_n_shuffles}")
