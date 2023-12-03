import pycuber as pc
import random
from .process_data import flatten_one_hot
import numpy as np
from typing import Any, Tuple
import time


ACTIONS = ["F", "B", "R", "L", "U", "D", "F'", "B'", "R'", "L'", "U'", "D'"]


def generate_sequence(num_scrambles: int):
    cube_sequence = []
    cube = pc.Cube()

    for _ in range(num_scrambles):
        cube_copy = cube.copy()
        cube_sequence.append(cube_copy)
        cube(random.choice(ACTIONS))

    cube_sequence.reverse()

    return cube_sequence


def generate_flattened_sequence_with_distances(num_scrambles: int):
    cube_sequence = []
    distances = []

    cube = pc.Cube()
    for i in range(num_scrambles):
        cube_copy = cube.copy()
        cube_sequence.append(flatten_one_hot(cube_copy))
        distances.append(i)
        cube(random.choice(ACTIONS))

    cube_sequence.reverse()
    distances.reverse()

    return cube_sequence, distances


def get_scrambled_cubes_flattened_with_distances(num_sequences: int, num_scrambles: int, print_progress: bool = False, idle: float = None) -> Tuple:
    cubes_flattened = []
    distances = []

    n = int(num_sequences / 5)
    for i in range(num_sequences):
        if print_progress and i % n == 0:
            print(f"{i + 1}/{num_sequences}")
    
        cubes_sequence, distances_sequence = generate_flattened_sequence_with_distances(num_scrambles)
        cubes_flattened.extend(cubes_sequence)
        distances.extend(distances_sequence)

        if idle:
            time.sleep(idle)

    return np.array(cubes_flattened), np.array(distances)


def get_scrambled_cubes(num_sequences: int, max_num_scrambles: int) -> np.array:
    cubes = []
    for _ in range(num_sequences):
        num_scrambles = random.randint(1, max_num_scrambles)
        cubes.extend(generate_sequence(num_scrambles))
    
    return np.array(cubes)
