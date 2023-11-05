import model_utils as m_utils
import data_utils as d_utils
import data_utils as data
import numpy as np


def train(batch_size: int, max_num_scrambles: int, training_iterations: int, convergence_check_freq: int, error_threshold: float, model_path: str = None, load_model_weights: bool = False) -> None:
    model_learn = m_utils.build_model()
    model_improve = m_utils.build_model()

    if model_path is not None and load_model_weights:
        model_learn.load_weights(model_path)
        model_improve.load_weights(model_path)

    for m in range(training_iterations):
        print(f"\n\n######## {m + 1}/{training_iterations} ########\n")
        X_cubes = data.get_scrambled_cubes(batch_size, max_num_scrambles)
        X, y = [], []

        len_X_cubes = len(X_cubes)
        for i, cube in enumerate(X_cubes):
            if (i + 1) % int(len_X_cubes * 0.2) == 0 or i == 0:
                print(f"{i + 1}/{len_X_cubes}...")
            value = m_utils.get_updated_cost_to_go_value(cube, model_improve)
            
            y.append(value)
            X.append(d_utils.flatten_one_hot(cube))
        
        X = np.array(X)
        y = np.array(y)

        hist = model_learn.fit(X, y, epochs=1, verbose=0)
        loss = hist.history["loss"][-1]
        print("Loss:", loss)

        if model_path is not None:
            model_learn.save_weights(model_path)
            print("Saved model!")

        print(m + 1, convergence_check_freq, (m + 1) % convergence_check_freq == 0)
        print(loss, error_threshold, loss < error_threshold)

        if ((m + 1) % convergence_check_freq == 0) and (loss < error_threshold):
            model_improve.set_weights(model_learn.get_weights())
            print("Updated models!")
