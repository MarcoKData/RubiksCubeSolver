import inference
import model_utils as m_utils


model = m_utils.build_model_simple()
model.load_weights("/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/classic-training/simple/model.h5")

n_shuffles, solution_lens_max = inference.get_max_n_shuffles_solved_plus_solution_lens(model)

print(f"\nSolved up to n_shuffles = {n_shuffles}")
print(f"Maximum solution length: {solution_lens_max}")
