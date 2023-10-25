import keras.backend as K
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# from blog-post: https://towardsdatascience.com/learning-to-solve-a-rubiks-cube-from-scratch-using-reinforcement-learning-381c3bac5476
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
