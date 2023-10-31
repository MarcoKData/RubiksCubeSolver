from .model import build_model
from .help_functions import *
import numpy as np
from datetime import datetime
import json
from sklearn.model_selection import train_test_split


def save_training_times(path, seconds_trained):
    with open(path, "r") as file:
        training_times = json.load(file)
    
    training_times.append(seconds_trained)

    with open(path, "w") as file:
        file.write(json.dumps(training_times, indent=4))
    
    print("Saved training times!")


def train(
    n_batches,
    n_samples_per_batch,
    n_epochs,
    save_training_times_path,
    save_weights=True,
    load_weights=True,
    save_path="model.h5",
    load_path="model.h5"):
    model = build_model()
    if load_weights:
        model.load_weights(load_path)
        print("Loaded Model Weights!")
    else:
        print("Skipped loading Model Weights.")

    cubes = []
    steps_until_solved = []
    cubes, distances = generate_cube_sequences(n_samples=n_batches, sequence_length=n_samples_per_batch)
    cubes_fully_flattened = np.array([flatten_one_hot(cube) for cube in cubes])
    distances = np.array(distances)

    X_train, X_test, y_train, y_test = train_test_split(cubes_fully_flattened, distances, test_size=0.3, random_state=187)

    print(f"X_train.shape, y_train.shape: {X_train.shape, y_train.shape}")

    model = build_model()
    model.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        batch_size=n_samples_per_batch,
        validation_data=(X_test, y_test)
    )

    preds = model.predict(X_test)
    preds = [pred[0] for pred in preds]
    print("preds first 10:", preds[:10])
    print("y_test fiorst 10:", y_test[:10])
