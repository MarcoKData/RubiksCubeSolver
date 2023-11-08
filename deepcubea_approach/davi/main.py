import model_utils as m_utils
import data_utils as d_utils
import data_utils as data
import numpy as np
from datetime import datetime
import json
import time
import shutil


def train(
    batch_size: int,
    max_num_scrambles: int,
    training_iterations: int,
    convergence_check_freq: int,
    error_threshold: float,
    model_path: str = None,
    model_backup_path: str = None,
    path_training_times: str = None,
    path_to_training_times_metrics: str = None,
    load_model_weights: bool = False
) -> None:
    model_learn = m_utils.build_model()
    model_improve = m_utils.build_model()

    if model_path is not None and load_model_weights:
        model_learn.load_weights(model_path)
        model_improve.load_weights(model_path)

    for m in range(training_iterations):
        t0 = datetime.now()
        print(f"\n\n######## {m + 1}/{training_iterations} ########\n")
        X_cubes = data.get_scrambled_cubes(batch_size, max_num_scrambles)
        X, y = [], []

        len_X_cubes = len(X_cubes)
        for i, cube in enumerate(X_cubes):
            time.sleep(0.02)
            if (i + 1) % int(len_X_cubes * 0.2) == 0 or i == 0:
                print(f"{i + 1}/{len_X_cubes}...")
            value = m_utils.get_updated_cost_to_go_value(cube, model_improve)

            y.append(value)
            X.append(d_utils.flatten_one_hot(cube))
        
        X = np.array(X)
        y = np.array(y)
        print("Mean y for training:", y.mean())

        hist = model_learn.fit(X, y, epochs=1, verbose=0)
        loss = hist.history["loss"][-1]
        print("Loss:", loss)

        if model_path is not None:
            model_learn.save_weights(model_path)
            print("Saved model!")
            time.sleep(1.0)
            if model_backup_path is not None:
                shutil.copyfile(model_path, model_backup_path)
                print("## Copied model to backup file! ##")

        print(m + 1, convergence_check_freq, (m + 1) % convergence_check_freq == 0)
        print(loss, error_threshold, loss < error_threshold)

        if (m + 1) % convergence_check_freq == 0:
            model_improve.set_weights(model_learn.get_weights())
            print("Updated models!")

        t1 = datetime.now()
        dt = (t1 - t0).total_seconds()

        if path_training_times is not None:
            with open(path_training_times, "r") as file:
                times = json.load(file)
            
            times.append(dt)

            with open(path_training_times, "w") as file:
                file.write(json.dumps(times, indent=4))
            
            print("Saved times successfully!")
        
        if (m + 1) % convergence_check_freq == 0:
            m_utils.test_deviation_single_cubes(
                path_to_model=model_backup_path,
                path_to_times=path_training_times,
                path_to_times_metrics=path_to_training_times_metrics
            )
            print("#### Tested Deviation on Single Cubes! ####")
