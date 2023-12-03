import data_utils as data
import model_utils as m_utils
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime


def append_mae_to_times_mae(path_times_mae: str, mae: float, seconds_iteration: float):
    with open(path_times_mae, "r") as file:
        times_mae = json.load(file)
    
    if len(times_mae.keys()) == 0:
        training_time_by_now = 0
    else:
        training_time_by_now = float(list(times_mae.keys())[-1])
    
    times_mae[training_time_by_now + seconds_iteration] = mae

    with open(path_times_mae, "w") as file:
        file.write(json.dumps(times_mae, indent=4))


def train_classic(model_type: str, model_path: str, n_training_iterations: int, num_cube_sequences: int, num_scrambles_per_sequence: int, num_epochs_per_dataset: int,
                  path_times_mae: str, path_metrics_n_shuffles: str):
    if model_type == "simple":
        model = m_utils.build_model_simple()
    elif model_type == "complex":
        model = m_utils.build_model_residual()

    model.summary()

    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("\nLoaded weights!")
    else:
        print("\nDid not load weights, training new model!")

    for iteration in range(n_training_iterations):
        print(f"\n{iteration + 1}/{n_training_iterations}...")

        print("Getting cubes...")
        cubes_flattened, distances = data.get_scrambled_cubes_flattened_with_distances(
            num_sequences=num_cube_sequences,
            num_scrambles=num_scrambles_per_sequence,
            idle=0.0
        )

        X_train, X_test, y_train, y_test = train_test_split(cubes_flattened, distances, test_size=0.3)

        print("Training model...")
        t0 = datetime.now()
        model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=num_epochs_per_dataset,
            validation_data=(X_test, y_test),
            verbose=1
        )
        t1 = datetime.now()

        model.save_weights(model_path)
        print("Saved weights!")

        seconds_iteration = (t1 - t0).total_seconds()

        m_utils.test_deviation_single_cubes(
            time_since_last_save=seconds_iteration,
            path_to_model=model_path,
            path_to_times_mae=path_times_mae,
            path_to_times_metrics_n_shuffles=path_metrics_n_shuffles,
            model_type=model_type,
            iterations_per_n_shuffles=50,
            idle=0.0
        )


    print("\n########\nFinished training!\n########\n")


if __name__ == "__main__":
    train_classic(
        model_type="complex",
        model_path="./saved_models/classic-training/complex/model.h5",
        n_training_iterations=10,
        num_cube_sequences=5,
        num_scrambles_per_sequence=10,
        num_epochs_per_dataset=5,
        path_times_mae="./saved_models/classic-training/complex/times_mae.json",
        path_metrics_n_shuffles="./saved_models/classic-training/complex/times_metrics_n_shuffles.json"
    )
