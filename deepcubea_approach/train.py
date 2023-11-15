import davi
import os


MODEL_DIR = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/complex"

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.h5")
MODEL_BACKUP_SAVE_PATH = os.path.join(MODEL_DIR, "model_copy.h5")
TIMES_SAVE_PATH = os.path.join(MODEL_DIR, "training_seconds.json")
TIMES_METRICS_SAVE_PATH = os.path.join(MODEL_DIR, "times_mae.json")
TIMES_METRICS_NEW_SAVE_PATH = os.path.join(MODEL_DIR, "times_metrics.json")

davi.train(
    batch_size=8,
    max_num_scrambles=25,
    training_iterations=999_999,
    convergence_check_freq=10,
    error_threshold=0.5,
    model_path=MODEL_SAVE_PATH,
    model_backup_path=MODEL_BACKUP_SAVE_PATH,
    path_training_times=TIMES_SAVE_PATH,
    path_to_training_times_metrics=TIMES_METRICS_SAVE_PATH,
    path_to_times_metrics_new=TIMES_METRICS_NEW_SAVE_PATH,
    load_model_weights=False
)
