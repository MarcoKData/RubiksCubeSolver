from typing import List
import numpy as np
from keras.utils import to_categorical


COLOR_LIST_MAP = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}
ACTION_MAP = {'F': 0, 'B': 1, 'U': 2, 'D': 3, 'L': 4, 'R': 5, "F'": 6, "B'": 7, "U'": 8, "D'": 9, "L'": 10, "R'": 11}


def flatten_one_hot(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.extend(COLOR_LIST_MAP[x[i][j].colour])

    return flat


def actions_to_one_hot(actions: List) -> np.array:
    actions_num = [ACTION_MAP[action] for action in actions]
    actions_one_hot = to_categorical(actions_num, num_classes=len(ACTION_MAP))

    return actions_one_hot
