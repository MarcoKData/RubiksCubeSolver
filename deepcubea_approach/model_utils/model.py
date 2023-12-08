from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


def build_model_simple(learning_rate=1e-3):
    inputs = layers.Input(shape=(324,))

    x = layers.Dense(1024)(inputs)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(50)(x)
    x = layers.LeakyReLU()(x)

    outputs = layers.Dense(1)(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        loss="mae",
        optimizer=Adam(learning_rate=learning_rate)
    )

    return model


def build_model_simple_cc(num_classes: int, learning_rate=1e-3):
    inputs = layers.Input(shape=(324,))

    x = layers.Dense(1024)(inputs)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(50)(x)
    x = layers.LeakyReLU()(x)

    out = layers.Dense(num_classes)(x)
    out = layers.Activation("softmax")(out)

    model = Model(
        inputs=inputs,
        outputs=out
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    return model


def residual_block(x: np.array):
    fx = layers.Dense(1024)(x)
    fx = layers.LeakyReLU()(fx)
    x = layers.BatchNormalization()(fx)

    fx = layers.Dense(1024)(fx)
    fx = layers.LeakyReLU()(fx)
    x = layers.BatchNormalization()(fx)

    out = layers.Add()([fx, x])
    out = layers.LeakyReLU()(out)
    out = layers.BatchNormalization()(out)

    return out


def build_model_residual(learning_rate=1e-3):
    inputs = layers.Input(shape=(324,))

    x = layers.Dense(5120)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    
    for _ in range(4):
        x = residual_block(x)

    outputs = layers.Dense(1)(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        loss="mae",
        optimizer=Adam(learning_rate=learning_rate)
    )

    return model


def build_model(learning_rate=1e-3):
    return build_model_simple(learning_rate)
