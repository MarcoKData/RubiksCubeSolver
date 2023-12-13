from .main import get_scrambled_cubes_flattened_with_distances
import math
import numpy as np
import json


def get_one_hot_dict(values: np.array) -> np.array:
    res = {}
    
    uniques = np.unique(values)
    uniques.sort()
    maximum = uniques.max()

    for unique_value in uniques:
        one_hot = np.zeros(maximum + 1)
        one_hot[int(unique_value)] = 1
        res[int(unique_value)] = one_hot
    
    return res


def categorize_distance(distance: int, up_to: int, step_width: int) -> int:
    max_class = math.floor(up_to / step_width)
    
    class_value = math.ceil(distance / step_width)
    if class_value >= max_class:
        class_value = max_class

    return class_value


def get_data_complexity_classes_f(num_scrambles: int, num_sequences: int, up_to: int = 25, step_width: int = 3):
    cubes, distances = get_scrambled_cubes_flattened_with_distances(
        num_scrambles=num_scrambles,
        num_sequences=num_sequences
    )

    distances = [categorize_distance(dist, up_to, step_width) for dist in distances]
    num_classes = len(np.unique(distances))

    one_hot_dict = get_one_hot_dict(distances)
    distances_one_hot = np.array([one_hot_dict[value] for value in distances])

    return cubes, distances_one_hot, num_classes
