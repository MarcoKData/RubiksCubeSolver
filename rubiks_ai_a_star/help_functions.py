import pycuber as pc
import random
from collections import Counter
import numpy as np


action_map = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11}
color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}
inv_action_map = {v: k for k, v in action_map.items()}


def generate_sequence(n_shuffles=5):
    cube = pc.Cube()
    transformations = [random.choice(list(action_map.keys())) for _ in range(n_shuffles)]

    # shuffle the cube
    formula = pc.Formula(transformations)
    cube(formula)

    # reverse the formula and save every cube and distance to solved
    cubes = []
    distances_to_solved = []
    
    formula.reverse()
    for i in range(len(formula)):
        cube(formula[i])
        cubes.append(cube.copy())
        distances_to_solved.append(len(formula) - i - 1)
    
    return cubes, distances_to_solved


def generate_cube_sequences(n_samples, sequence_length=25):
    cubes = []
    distances_to_solved = []

    for _ in range(n_samples):
        cubes_sequence, distances_sequence = generate_sequence(sequence_length)
        cubes.extend(cubes_sequence)
        distances_to_solved.extend(distances_sequence)

    return cubes, distances_to_solved


def flatten_one_hot(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.extend(color_list_map[x[i][j].colour])

    return flat


def flatten(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.append(x[i][j].colour)

    return flat


def perc_solved_cube(cube):
    flat = flatten(cube)
    perc_side = [largest_equal_share(flat[i:(i + 9)]) for i in range(0, 9 * 6, 9)]

    return np.mean(perc_side)


def largest_equal_share(data):
    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]

    return max(probs)


def get_reward_for_cube(cube):
    flattened = flatten(cube)
    perc_side = [largest_equal_share(flattened[i:(i + 9)]) for i in range(0, 9 * 6, 9)]

    mean_solved = np.mean(perc_side)
    reward = 2 * int(mean_solved > 0.99) - 1

    return reward


def get_all_possible_actions_rewards_cube(cube):
    successor_cubes = []
    successor_rewards = []

    for action in action_map.keys():
        cube_copy = cube.copy()
        cube_copy = cube_copy(action)
        successor_cubes.append(flatten_one_hot(cube_copy))
        successor_rewards.append(get_reward_for_cube(cube_copy))

    return successor_cubes, successor_rewards


def get_actions_rewards(cubes):
    # cubes_next_rewards: shape = (n, 12) --> rewards for each possible action to consider --> here: -1 if not solved, 1 if solved
    # flat_next_states: cubes resulting from all possible actions flattened to lowest level possible (like cubes_flat) --> for predicting purposes --> will be reshaped to (n, 12) later
    # cubes_flat: all cubes flattened to the lowest possible level --> 1D with flattened one-hot-encoded colors --> shape = (n, 54 fields * 6 colors) = (n, 324)

    cube_next_reward = []
    flat_next_states = []
    cube_flat = []

    for c in cubes:
        flat_cubes, rewards = get_all_possible_actions_rewards_cube(c)
        cube_next_reward.append(rewards)
        flat_next_states.extend(flat_cubes)
        cube_flat.append(flatten_one_hot(c))

    return cube_next_reward, flat_next_states, cube_flat


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



## INFERENCE
def generate_sample_cube(n_shuffles=5):
    cube = pc.Cube()
    transformations = []
    for _ in range(n_shuffles):
        if len(transformations) == 0:
            list_of_actions = list(action_map.keys())
        else:
            list_of_actions = [action for action in list(action_map.keys()) if action != transformations[-1]]
        
        transformations.append(random.choice(list_of_actions))

    formula = pc.Formula(transformations)
    cube(formula)

    return cube


def cube_is_solved(cube):
    return perc_solved_cube(cube) > 0.99
