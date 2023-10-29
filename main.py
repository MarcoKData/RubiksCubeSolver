import rubiks_ai as ai
import parse_cube as parser
import os
from matplotlib import image
import matplotlib.pyplot as plt
import pycuber as pc
import monte_carlo_tree_search as mcts_nn


load_path = os.path.join(".", "models", "model.h5")

# train the model
"""ai.train(
    n_epochs=1,
    n_iterations=20,
    n_samples=5,
    save_weights=False,
    load_weights=False,
    load_path=os.path.join(".", "models", "model.h5"),
    save_path=os.path.join(".", "models", "model-save-path.h5")
)"""


# check how far the model can get (n_shuffles) with random cubes
"""max_length_solving_seq, max_solved_n_shuffles = ai.inference(
    load_path=load_path,
    max_steps_until_solved=50
)

print("Successfully solved cubes up to...")
print(f"max_length_solving_seq = {max_length_solving_seq}")
print(f"max_solved_n_shuffles = {max_solved_n_shuffles}")"""


# solve single random cube with specified n_shuffles
"""steps, solved = ai.single_random_cube_solve(n_shuffles=5, model_path=load_path)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve cube!")"""


# solve specific cube (user defined)
"""
cube = parser.get_sample_cube_solved()

model = ai.build_model()
model.load_weights(load_path)

steps, solved = ai.solve_cube(cube, model, use_probability_dist=True)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve!")"""


# solve specific cube from images
"""f_path = os.path.join(".", "example_imgs", "f.png")
b_path = os.path.join(".", "example_imgs", "b.png")
d_path = os.path.join(".", "example_imgs", "d.png")
l_path = os.path.join(".", "example_imgs", "l.png")
r_path = os.path.join(".", "example_imgs", "r.png")
u_path = os.path.join(".", "example_imgs", "u.png")

f_img = image.imread(f_path)
b_img = image.imread(b_path)
d_img = image.imread(d_path)
l_img = image.imread(l_path)
r_img = image.imread(r_path)
u_img = image.imread(u_path)

cube = parser.cube_from_side_imgs(f_img, r_img, l_img, b_img, u_img, d_img)
print(cube)

model = ai.build_model()
model.load_weights(load_path)

steps, solved = ai.solve_cube(cube, model, use_probability_dist=True)
if solved:
    print("Solved! Steps:")
    print(steps)
else:
    print("Failed to solve!")"""


# solve cube with mcts + nn
cube = pc.Cube()

MIX_SEQUENCE_BENCHMARK = ["F", "L", "L", "U", "B", "B", "L", "R", "U", "B", "L", "F", "R", "L", "B"]
MIX_SEQUENCE_MEDIUM_PLUS = ["F", "L", "L", "U", "B", "B", "L", "R", "U", "B"]
MIX_SEQUENCE_MEDIUM = ["F", "L", "L", "U", "B", "B", "L", "R"]
MIX_SEQUENCE_SHORT = ["F", "L", "L", "U", "B", "B"]
MIX_SEQUENCE_VERY_SHORT = ["F", "L", "L", "U"]

MIX_SEQUENCE_MEDIUM_WILD = ["F'", "L", "L", "U", "B", "U'", "L", "R'"]

cube = mcts_nn.execute_sequence(cube, MIX_SEQUENCE_MEDIUM_WILD)
cube_original = cube.copy()

solution = mcts_nn.solve_with_mcts(cube, load_path, max_moves=500, num_iterations_per_move=50, iteration_limit_depth=20, init_v_value_threshold=10.0)
print("Solution:", solution)


# parse cube and apply given sequence
"""f_path = os.path.join(".", "example_imgs", "2", "f.jpeg")
b_path = os.path.join(".", "example_imgs", "2", "b.jpeg")
d_path = os.path.join(".", "example_imgs", "2", "d.jpeg")
l_path = os.path.join(".", "example_imgs", "2", "l.jpeg")
r_path = os.path.join(".", "example_imgs", "2", "r.jpeg")
u_path = os.path.join(".", "example_imgs", "2", "u.jpeg")

f_img = image.imread(f_path)
b_img = image.imread(b_path)
d_img = image.imread(d_path)
l_img = image.imread(l_path)
r_img = image.imread(r_path)
u_img = image.imread(u_path)

cube = parser.cube_from_side_imgs(f_img, r_img, l_img, b_img, u_img, d_img)
print(cube)

sequence = ["R'", "L", "B'", "F'", "U'", "D", "U", "D'", "B'", "F", "U", "L", "L", "F"]

for move in sequence:
    cube = cube(move)

print(cube)"""
