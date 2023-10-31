import rubiks_ai_a_star as ai


ai.train(
    n_batches=500,
    n_samples_per_batch=16,
    n_epochs=30,
    save_training_times_path="./times.json",
    save_weights=False,
    load_weights=False
)
