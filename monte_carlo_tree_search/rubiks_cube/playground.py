from pycuber import Cube
from mcts import MCTS_CUBE
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


def cube_is_solved(cube: Cube):
    is_solved = True

    faces = ["F", "L", "R", "U", "D", "B"]
    for face_ltr in faces:
        face = cube.get_face(face_ltr)
        center_stone = face[1][1]
        for row in face:
            for stone in row:
                if stone != center_stone:
                    is_solved = False
    
    return is_solved


def execute_sequence(cube, sequence):
    for move in sequence:
        cube = cube(move)
    
    return cube


def acc(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())


def build_model(learning_rate=1e-4):
    inputs = layers.Input(shape=(324,))  # 324 because of maximally flattened cube-grid (one-hot for colors)

    x = layers.Dense(1024)(inputs)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(50)(x)
    x = layers.LeakyReLU()(x)

    out_value = layers.Dense(1, name="value")(x)
    out_policy = layers.Dense(12, activation="softmax", name="policy")(x)

    model = Model(
        inputs=inputs,
        outputs=[out_value, out_policy]
    )

    model.compile(
        loss={"value": "mae", "policy": "sparse_categorical_crossentropy"},
        optimizer=Adam(learning_rate=learning_rate),
        metrics={"policy": acc}
    )

    # model.summary()

    return model


cube = Cube()
cube = execute_sequence(cube, ["F", "L", "L"])
cube_original = cube.copy()

load_path = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/models/model.h5"
model = build_model()
model.load_weights(load_path)

mcts = MCTS_CUBE(model=model)

moves_to_make = []

move_node_obj = mcts.search(cube)
move = move_node_obj.move_made
cube = cube(move)
moves_to_make.append(move)

move_node_obj = mcts.search(cube)
move = move_node_obj.move_made
cube = cube(move)
moves_to_make.append(move)

move_node_obj = mcts.search(cube)
move = move_node_obj.move_made
cube = cube(move)
moves_to_make.append(move)

print(moves_to_make)
