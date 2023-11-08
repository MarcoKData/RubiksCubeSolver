import model_utils as m_utils
import data_utils as data
import inference
import pycuber as pc


PATH_MODEL = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model.h5"

model = m_utils.build_model()
model.load_weights(PATH_MODEL)

cube = pc.Cube()
cube("F")
cube("R")
cube("D")
cube("U'")
cube("R'")
cube("L")

print(cube)

sequence = inference.solve_with_batch_dive(cube, model, max_num_iterations=50, batch_depth=3, prune_to_best_n=2)
print("Sequence:", sequence)
for move in sequence:
    cube(move)

print("RESULT:")
print(cube)


"""cubes = data.get_scrambled_cubes(num_sequences=1, max_num_scrambles=25)
for cube in cubes:
    flattened = data.flatten_one_hot(cube).reshape((1, -1))
    cost_to_go = model(flattened).numpy()[0][0]
    print(cost_to_go)"""
