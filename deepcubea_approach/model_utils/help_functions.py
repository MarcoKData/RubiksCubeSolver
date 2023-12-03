import numpy as np
from pycuber import Cube
from keras.models import Model
import data_utils as data
from .predict import predict_cost_to_go_from_cube


def get_inverse_action(action: str) -> str:
    return (action + "'").replace("''", "")


def get_updated_cost_to_go_value(cube: Cube, model: Model) -> float:
    min_pred_cost_to_go = 999_999

    for action in data.ACTIONS:
        cube(action)
        predicted_cost_to_go_successor = predict_cost_to_go_from_cube(cube, model)
        cube(get_inverse_action(action))

        if predicted_cost_to_go_successor < min_pred_cost_to_go:
            min_pred_cost_to_go = predicted_cost_to_go_successor

    return min_pred_cost_to_go + 1
