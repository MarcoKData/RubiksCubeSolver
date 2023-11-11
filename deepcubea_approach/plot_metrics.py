import json
import matplotlib.pyplot as plt
import numpy as np


PATH_TO_N_SHUFFLES_DATA = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/times_metrics.json"


def get_mvg_avg(values: np.array, window_size: int = 10):
    res = []
    for i in range(len(values) - window_size):
        window = values[i:i + window_size]
        res.append(window.mean())
    
    return res


def analyze_n_shuffles_performance():
    with open(PATH_TO_N_SHUFFLES_DATA, "r") as file:
        metrics = json.load(file)
    
    sample = list(metrics.values())[0]["metrics_per_n_shuffles"]
    means_mins_maxes = {}
    for n_shuffles in sample.keys():
        means_mins_maxes[int(n_shuffles)] = {
            "means": [],
            "mins": [],
            "maxes": [],
            "abs_error_mean": []
        }

    for _, single_metrics in list(metrics.items()):
        metrics_per_n_shuffles = single_metrics["metrics_per_n_shuffles"]
        for n_shuffles, details in metrics_per_n_shuffles.items():
            means_mins_maxes[int(n_shuffles)]["means"].append(details["mean_pred"])
            means_mins_maxes[int(n_shuffles)]["mins"].append(details["min_pred"])
            means_mins_maxes[int(n_shuffles)]["maxes"].append(details["max_pred"])
            means_mins_maxes[int(n_shuffles)]["abs_error_mean"].append(details["mean_pred"] - int(n_shuffles))

    print(json.dumps(means_mins_maxes, indent=4))

    plt.figure(figsize=(10, 7))
    MVG_AVG_WINDOW_SIZE = 50
    plt.title(f"MAE by n_shuffles ({MVG_AVG_WINDOW_SIZE}-window moving average)")

    for n_shuffles, details_lists in means_mins_maxes.items():
        if n_shuffles in [3, 8, 12, 15]:
            values = np.array(details_lists["abs_error_mean"])
            mvg_avg = get_mvg_avg(values, window_size=MVG_AVG_WINDOW_SIZE)
            plt.plot(range(len(mvg_avg)), mvg_avg, label=str(n_shuffles))

    plt.legend()
    plt.show()


analyze_n_shuffles_performance()
