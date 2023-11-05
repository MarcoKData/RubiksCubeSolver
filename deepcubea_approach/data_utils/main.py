import pycuber as pc
import random
from .process_data import flatten_one_hot
import numpy as np


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


def get_scrambled_cubes(num_sequences: int, max_num_scrambles: int) -> pc.Cube:
    cubes = []
    for _ in range(num_sequences):
        num_scrambles = random.randint(1, max_num_scrambles)
        cubes.extend(generate_sequence(num_scrambles))
    
    return np.array(cubes)
