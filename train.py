from model import build_model
from help_functions import *
import numpy as np
from datetime import datetime
import json


def save_training_times(path, seconds_trained):
    with open(path, "r") as file:
        training_times = json.load(file)
    
    training_times.append(seconds_trained)

    with open(path, "w") as file:
        file.write(json.dumps(training_times, indent=4))
    
    print("Saved training times!")


def train(
    n_epochs,
    n_iterations,
    n_samples,
    save_training_times_path,
    save_weights=True,
    load_weights=True,
    save_path="model.h5",
    load_path="model.h5"):
    model = build_model()
    if load_weights:
        model.load_weights(load_path)
        print("Loaded Model Weights!")
    else:
        print("Skipped loading Model Weights.")

    total_iterations = n_epochs * n_iterations
    iteration_counter = 1

    for i in range(n_epochs):
        print(f"\n\n\n######## EPOCH {i + 1}/{n_epochs} ########\n")
        with open(save_training_times_path, "r") as file:
            training_times = json.load(file)
        total_seconds_trained = np.array(training_times).sum()
        print(f"Trained {total_seconds_trained} Seconds ({total_seconds_trained / 3600} Hours) by now...")

        # generate samples
        print("Generating Dataset...")
        t0 = datetime.now()
        cubes, distances_to_solved = generate_cube_sequences(n_samples)
        # cubes_next_rewards: (n, 12), flat_next_states: (n * 12, 324), cubes_flat: (n, 324)
        cubes_next_rewards, flat_next_states, cubes_flat = get_actions_rewards(cubes)

        # iterate to improve model
        for _ in range(n_iterations):
            print(f"\n\n\ntotal {iteration_counter}/{total_iterations}...\n")
            iteration_counter += 1
            # estimate v-values
            estimated_next_v_values, _ = model.predict(np.array(flat_next_states), batch_size=1024, verbose=0)
            estimated_next_v_values = [pred[0] for pred in estimated_next_v_values]
            estimated_next_v_values = list(chunker(estimated_next_v_values, len(action_map.items())))
            
            target_v_values = []
            target_policy = []

            for rewards, predicted_state_values in zip(cubes_next_rewards, estimated_next_v_values):
                r_plus_v = 0.4 * np.array(rewards) + np.array(predicted_state_values)
                target_v = np.max(r_plus_v)
                target_p = np.argmax(r_plus_v)

                target_v_values.append(target_v)
                target_policy.append(target_p)
            
            # normalize - shape = (n,)
            target_v_values = (target_v_values-np.mean(target_v_values))/(np.std(target_v_values)+0.01)

            sample_weights = 1. / np.array(distances_to_solved)
            sample_weights = sample_weights * sample_weights.size / np.sum(sample_weights)

            history = model.fit(np.array(cubes_flat), [np.array(target_v_values), np.array(target_policy)[..., np.newaxis]],
                    epochs=1, batch_size=128, sample_weight=[sample_weights, sample_weights], verbose=0)
            value_loss = history.history["value_loss"][0]
            policy_loss = history.history["policy_loss"][0]
            policy_acc = history.history["policy_acc"][0]
            print(f"Value Loss: {value_loss}, Policy Loss: {policy_loss}, Policy Acc: {policy_acc}")
        
        if save_weights:
            model.save_weights(save_path)
            print("Saved Model Weights!")

            t1 = datetime.now()
            seconds_trained = (t1 - t0).total_seconds()
            save_training_times(save_training_times_path, seconds_trained)
        else:
            print("Skipped saving Model Weights and Training Time.")
