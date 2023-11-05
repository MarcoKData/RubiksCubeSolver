import numpy as np
from pycuber import Cube
from keras.models import Model
import data_utils as data
from .predict import predict_cost_to_go_from_cube


def get_updated_cost_to_go_value(cube: Cube, model: Model) -> float:
    all_child_values = []

    for action in data.ACTIONS:
        cube_copy = cube.copy()
        cube_copy(action)
        predicted_cost_to_go_successor = predict_cost_to_go_from_cube(cube_copy, model)

        all_child_values.append(predicted_cost_to_go_successor + 1)

    return np.min(all_child_values)
