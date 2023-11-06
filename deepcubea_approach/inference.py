import inference
import data_utils as data
import model_utils as m_utils


model = m_utils.build_model()
model.load_weights("/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model_copy.h5")

solved_n_shuffles = []
n_shuffles = 1
while True:
    print(f"Testing n_shuffles: {n_shuffles}")
    cube = data.get_single_scrambled_cube(num_scrambles=n_shuffles)

    sequence = inference.solve_with_a_star(cube, model, max_num_iterations=30)
    if sequence is None:
        break

    solved_n_shuffles.append(n_shuffles)
    n_shuffles += 1
    print(f"Sequence: {sequence}")

print(f"Solved n_shuffles: {solved_n_shuffles}")
