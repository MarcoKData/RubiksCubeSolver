import model_utils as m_utils
import data_utils as data
import numpy as np
import json


PATH_TO_TIMES = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/training_seconds.json"
PATH_TO_TIMES_METRICS = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/times_mae.json"
PATH_TO_MODEL = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model_copy.h5"
PATH_TO_TIMES_METRICS_NEW = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/times_metrics.json"

def test_deviation_single_cubes(
    path_to_model: str = PATH_TO_MODEL,
    path_to_times_metrics: str = PATH_TO_TIMES_METRICS,
    path_to_times: str = PATH_TO_TIMES,
    iterations_per_n_shuffles: int = 30,
    n_shuffles_lower: int = 3,
    n_shuffles_upper: int = 15
):
    model = m_utils.build_model()
    model.load_weights(path_to_model)

    preds = []
    y_true = []
    metrics_per_n_shuffles = {}
    for n_shuffles in range(n_shuffles_lower, n_shuffles_upper + 1):
        sum_pred = 0
        min_pred = 999_999
        max_pred = -999_999
        for _ in range(iterations_per_n_shuffles):
            sample_cube = data.get_single_scrambled_cube(num_scrambles=n_shuffles)
            sample_cube = data.flatten_one_hot(sample_cube).reshape((1, -1))

            pred = model(sample_cube).numpy()[0][0]
            sum_pred += pred

            if pred > max_pred:
                max_pred = pred
            if pred < min_pred:
                min_pred = pred

            preds.append(pred)
            y_true.append(n_shuffles)
        
        if n_shuffles in [3, 8, 12, 15]:
            mean_pred = sum_pred / iterations_per_n_shuffles
            metrics_per_n_shuffles[n_shuffles] = {
                "mean_pred": float(mean_pred),
                "min_pred": float(min_pred),
                "max_pred": float(max_pred)
            }

    preds = np.array(preds)
    y_true = np.array(y_true)

    random_idxes = [np.random.randint(0, len(preds)) for _ in range(10)]
    random_preds = [preds[i] for i in random_idxes]
    random_y_true = [y_true[i] for i in random_idxes]

    print("10 random y_true:", random_y_true)
    print("corresponding preds:", random_preds)

    mae = np.mean(np.abs(preds - y_true))
    print("mae:", mae)

    with open(path_to_times, "r") as file:
        training_seconds = json.load(file)

    total_seconds_trained = sum(training_seconds)

    # MAE ONLY
    with open(path_to_times_metrics, "r") as file:
        times_metrics = json.load(file)

    times_metrics[total_seconds_trained] = mae

    with open(path_to_times_metrics, "w") as file:
        file.write(json.dumps(times_metrics, indent=4))

    print("Wrote results to times_metrics!")
    # END MAE ONLY

    # MAE AND METRICS PER N_SHUFFLES
    with open(PATH_TO_TIMES_METRICS_NEW, "r") as file:
        times_metrics = json.load(file)

    times_metrics[total_seconds_trained] = {
        "mae": mae,
        "metrics_per_n_shuffles": metrics_per_n_shuffles
    }

    with open(PATH_TO_TIMES_METRICS_NEW, "w") as file:
        file.write(json.dumps(times_metrics, indent=4))

    print("Wrote results to times_metrics!")
    # END MAE AND METRICS PER N_SHUFFLES
