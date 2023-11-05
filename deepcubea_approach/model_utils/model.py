from keras import layers
from keras.models import Model
from keras.optimizers import Adam


def build_model(learning_rate=1e-3):
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
