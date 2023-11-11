import json
import matplotlib.pyplot as plt


PATH_TO_N_SHUFFLES_DATA = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/times_metrics.json"


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
    plt.title("Error of the Means by n_shuffles")
    total_range = range(len(metrics.keys()))
    for n_shuffles, details_lists in means_mins_maxes.items():
        if n_shuffles in [3, 8, 12, 15]:
            plt.plot(total_range, details_lists["abs_error_mean"], label=str(n_shuffles))
    plt.legend()
    plt.show()


analyze_n_shuffles_performance()
