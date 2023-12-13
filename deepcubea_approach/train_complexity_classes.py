import data_utils as data
import model_utils as m_utils
from sklearn.model_selection import train_test_split
import os
import numpy as np
from datetime import datetime
import json
from typing import Any


def append_metric_to_file(metric: Any, path_metrics_file: str, dt_total_seconds: float) -> None:
    if not os.path.exists(path_metrics_file):
        metrics_time = {}
    else:
        with open(path_metrics_file, "r") as file:
            metrics_time = json.load(file)

    metrics_time[dt_total_seconds] = metric

    with open(path_metrics_file, "w") as file:
        file.write(json.dumps(metrics_time, indent=4))


def train_complexity_classes(
    n_iterations: int,
    num_seq_per_it: int,
    num_scrablmes: int,
    up_to_num_scrambles_for_classes: int,
    step_width_num_scrambles: int,
    load_weights: bool,
    model_base_dir: str,
    epochs_per_iteration: int
):
    # only mock to get num_classes
    _, _, num_classes = data.get_data_complexity_classes_f(
        num_sequences=10,
        num_scrambles=30,
        up_to=25,
        step_width=3
    )

    model = m_utils.build_model_simple_cc(num_classes)
    if load_weights:
        model.load_weights(os.path.join(model_base_dir, str(num_classes), "model.h5"))
        print("Loaded weights!")

    t_start = datetime.now()
    for i in range(n_iterations):
        print(f"{i + 1}/{n_iterations}...")
        cubes, distance_classes, num_classes = data.get_data_complexity_classes_f(
            num_sequences=num_seq_per_it,
            num_scrambles=num_scrablmes,
            up_to=up_to_num_scrambles_for_classes,
            step_width=step_width_num_scrambles
        )
        X_train, X_test, y_train, y_test = train_test_split(cubes, distance_classes, test_size=0.3)
        
        model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs_per_iteration,
            batch_size=32,
            validation_data=(X_test, y_test)
        )

        model_dir = os.path.join(model_base_dir, str(num_classes))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        model.save(os.path.join(model_dir, "model.h5"))

        t_inter = datetime.now()
        dt = (t_inter - t_start).total_seconds()

        metrics_classes = evaluate_complexity_classes(
            num_seq=200,
            num_scrablmes=num_scrablmes,
            up_to_num_scrambles_for_classes=up_to_num_scrambles_for_classes,
            step_width_num_scrambles=step_width_num_scrambles,
            model_base_dir=model_base_dir
        )

        path_metrics_file = os.path.join(model_base_dir, str(num_classes), "metrics.json")
        path_metrics_file_mean_acc_only = os.path.join(model_base_dir, str(num_classes), "mean_accs.json")

        append_metric_to_file(metrics_classes, path_metrics_file, dt)
        append_metric_to_file(metrics_classes["mean-acc"], path_metrics_file_mean_acc_only, dt)


def evaluate_complexity_classes(num_seq: int, num_scrablmes: int, up_to_num_scrambles_for_classes: int, step_width_num_scrambles: int, model_base_dir: str):
    cubes, distance_classes, num_classes = data.get_data_complexity_classes_f(
        num_sequences=num_seq,
        num_scrambles=num_scrablmes,
        up_to=up_to_num_scrambles_for_classes,
        step_width=step_width_num_scrambles
    )

    model = m_utils.build_model_simple_cc(num_classes)
    model_dir = os.path.join(model_base_dir, str(num_classes))
    model.load_weights(os.path.join(model_dir, "model.h5"))

    metrics_classes = {}
    accs = []
    for class_value in range(num_classes):
        cubes_class = []
        distance_classes_class = []

        for i in range(len(cubes)):
            if np.argmax(distance_classes[i]) == class_value:
                cubes_class.append(cubes[i])
                distance_classes_class.append(distance_classes[i])

        metrics = m_utils.get_metrics(cubes_class, distance_classes_class, model)
        accs.append(metrics["acc"])
        metrics_classes[class_value] = metrics
    
    metrics_classes["mean-acc"] = np.array(accs).mean()
    metrics_classes["min-acc"] = np.array(accs).min()
    metrics_classes["max-acc"] = np.array(accs).max()

    return metrics_classes


if __name__ == "__main__":
    NUM_SCRAMBLES = 25
    UP_TO_NUM_SCRAMBLES_CLASSES = 25
    STEP_WIDTH_NUM_SCRAMBLES = 3
    MODEL_BASE_DIR = "./saved_models/complexity_classes"

    train_complexity_classes(
        n_iterations=2,
        num_seq_per_it=100,
        epochs_per_iteration=10,
        num_scrablmes=NUM_SCRAMBLES,
        up_to_num_scrambles_for_classes=UP_TO_NUM_SCRAMBLES_CLASSES,
        step_width_num_scrambles=STEP_WIDTH_NUM_SCRAMBLES,
        model_base_dir=MODEL_BASE_DIR,
        load_weights=False
    )
