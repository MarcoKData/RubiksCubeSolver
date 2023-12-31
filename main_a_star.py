import rubiks_ai_a_star as ai


MODEL_PATH = "./models/a_star/model_one_run.h5"

"""while True:
    ai.train(
        n_batches=500,
        n_samples_per_batch=32,
        n_epochs=15,
        save_training_times_path="./times.json",
        save_weights=True,
        load_weights=True,
        save_path=MODEL_PATH,
        load_path=MODEL_PATH
    )"""

ai.solve_random_cube(
    model_weights_path=MODEL_PATH,
    shuffle_length=10
)
