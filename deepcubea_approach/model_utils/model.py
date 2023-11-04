from keras import layers
from keras.models import Model
from keras.optimizers import Adam


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

    outputs = layers.Dense(12, activation="softmax")(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy"
    )

    # model.summary()

    return model
