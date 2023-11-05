import davi


MODEL_SAVE_PATH = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model.h5"
print("OKEEEEEE LET'S GO...")
davi.train(
    batch_size=8,
    max_num_scrambles=25,
    training_iterations=999_999,
    convergence_check_freq=5,
    error_threshold=0.5,
    model_path=MODEL_SAVE_PATH,
    load_model_weights=True
)
