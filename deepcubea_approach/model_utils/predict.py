from keras.models import Model
from pycuber import Cube
import data_utils as data
import rubiks_utils as r_utils


def predict_cost_to_go_from_cube(cube: Cube, model: Model) -> float:
    if r_utils.is_final_cube_state(cube):
        return 0.0

    cube_flattened = data.flatten_one_hot(cube)
    cube_flattened = cube_flattened.reshape((1, -1))
    pred = model.predict(cube_flattened, verbose=0)[0][0]

    return pred
