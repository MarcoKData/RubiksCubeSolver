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


MODEL_TYPE = "complex"


MODEL_DIR = f"/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/classic-training/{MODEL_TYPE}"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
PATH_TIMES_MAE = os.path.join(MODEL_DIR, "times_mae.json")
PATH_METRICS_N_SHUFFLES = os.path.join(MODEL_DIR, "times_metrics_n_shuffles.json")

if os.path.exists(MODEL_PATH):
    LOAD_WEIGHTS = True
else:
    LOAD_WEIGHTS = False

SAVE_WEIGHTS = True

N_ITERATIONS = 20
NUM_CUBE_SEQUENCES = 500
NUM_SCRAMBLES_PER_SEQUENCE = 20
NUM_EPOCHS_PER_DATASET = 25


if MODEL_TYPE == "simple":
    model = m_utils.build_model_simple()
elif MODEL_TYPE == "complex":
    model = m_utils.build_model_residual()

model.summary()

if LOAD_WEIGHTS:
    model.load_weights(MODEL_PATH)
    print("\nLoaded weights!")
else:
    print("\nDid not load weights, training new model!")


for iteration in range(N_ITERATIONS):
    print(f"\n{iteration + 1}/{N_ITERATIONS}...")

    print("Getting cubes...")
    cubes_flattened, distances = data.get_scrambled_cubes_flattened_with_distances(
        num_sequences=NUM_CUBE_SEQUENCES,
        num_scrambles=NUM_SCRAMBLES_PER_SEQUENCE,
        idle=0.01
    )

    X_train, X_test, y_train, y_test = train_test_split(cubes_flattened, distances, test_size=0.3)

    print("Training model...")
    t0 = datetime.now()
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=NUM_EPOCHS_PER_DATASET,
        validation_data=(X_test, y_test),
        verbose=1
    )
    t1 = datetime.now()

    if SAVE_WEIGHTS:
        model.save_weights(MODEL_PATH)
        print("Saved weights!")
    else:
        print("Did not save weights!")

    seconds_iteration = (t1 - t0).total_seconds()

    m_utils.test_deviation_single_cubes(
        time_since_last_save=seconds_iteration,
        path_to_model=MODEL_PATH,
        path_to_times_mae=PATH_TIMES_MAE,
        path_to_times_metrics_n_shuffles=PATH_METRICS_N_SHUFFLES,
        iterations_per_n_shuffles=50,
        idle=0.05
    )


print("\n########\nFinished training!\n########\n")
