import inference
import data_utils as data
import model_utils as m_utils
import numpy as np


model = m_utils.build_model()
model.load_weights("/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model.h5")

solved_n_shuffles = []
solution_lens = []
n_shuffles = 1
while True:
    print(f"Testing n_shuffles: {n_shuffles}")
    cube = data.get_single_scrambled_cube(num_scrambles=n_shuffles)

    sequence = inference.solve_with_a_star(cube, model, max_num_iterations=30)
    if sequence is None:
        break

    solved_n_shuffles.append(n_shuffles)
    solution_lens.append(len(sequence))
    n_shuffles += 1
    print(f"Sequence: {sequence}")

solution_lens = np.array(solution_lens)

print(f"\nSolved up to n_shuffles = {n_shuffles}")
print(f"solution lens: {solution_lens}")
print(f"Maximum solution length: {solution_lens.max()}")
