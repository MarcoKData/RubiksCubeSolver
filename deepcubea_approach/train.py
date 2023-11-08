import davi


MODEL_SAVE_PATH = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model.h5"
MODEL_BACKUP_SAVE_PATH = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model_copy.h5"
TIMES_SAVE_PATH = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/training_seconds.json"
TIMES_METRICS_SAVE_PATH = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/times_mae.json"

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
    load_model_weights=True
)
