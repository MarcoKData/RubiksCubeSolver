import json
import matplotlib.pyplot as plt
import numpy as np


MODEL_TYPE = "simple"

PATH_TO_N_SHUFFLES_DATA = f"/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/classic-training/{MODEL_TYPE}/times_metrics_n_shuffles.json"
PATH_TO_MAE_OVER_TIME = f"/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/classic-training/{MODEL_TYPE}/times_mae.json"


def get_mvg_avg(details_lists: np.array, window_size: int = 10, key_y: str = "abs_error_mean", key_x: str = "seconds_trained"):
    x, y = [], []
    y_values = np.array(details_lists[key_y])
    x_values = np.array(details_lists[key_x])

    for i in range(len(y_values) - window_size):
        window = y_values[i:i + window_size]
        y.append(window.mean())
        x.append(x_values[i + window_size])

    return x, y


def analyze_n_shuffles_performance():
    with open(PATH_TO_N_SHUFFLES_DATA, "r") as file:
        metrics = json.load(file)
    
    sample = list(metrics.values())[0]["metrics_per_n_shuffles"]
    means_mins_maxes = {}
    for n_shuffles in sample.keys():
        means_mins_maxes[int(n_shuffles)] = {
            "seconds_trained": [],
            "hours_trained": [],
            "means": [],
            "mins": [],
            "maxes": [],
            "abs_error_mean": []
        }

    for seconds_trained, single_metrics in list(metrics.items()):
        metrics_per_n_shuffles = single_metrics["metrics_per_n_shuffles"]
        for n_shuffles, details in metrics_per_n_shuffles.items():
            means_mins_maxes[int(n_shuffles)]["seconds_trained"].append(float(seconds_trained))
            means_mins_maxes[int(n_shuffles)]["hours_trained"].append(float(seconds_trained) / 3600.0)
            means_mins_maxes[int(n_shuffles)]["means"].append(details["mean_pred"])
            means_mins_maxes[int(n_shuffles)]["mins"].append(details["min_pred"])
            means_mins_maxes[int(n_shuffles)]["maxes"].append(details["max_pred"])
            means_mins_maxes[int(n_shuffles)]["abs_error_mean"].append(np.abs(details["mean_pred"] - int(n_shuffles)))

    plt.figure(figsize=(10, 7))
    MVG_AVG_WINDOW_SIZE = 1
    plt.title(f"MAE by n_shuffles ({MVG_AVG_WINDOW_SIZE} iterations moving average)")
    plt.xlabel("Hours trained")
    plt.ylabel("Mean Absolute Error")

    for n_shuffles, details_lists in means_mins_maxes.items():
        if n_shuffles in [3, 8, 12, 15]:
            x, y = get_mvg_avg(details_lists, window_size=MVG_AVG_WINDOW_SIZE, key_x="hours_trained", key_y="abs_error_mean")
            plt.plot(x, y, label=str(n_shuffles))

    plt.legend()
    plt.show()


analyze_n_shuffles_performance()
