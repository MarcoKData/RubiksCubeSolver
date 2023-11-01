from .model import build_model
from .help_functions import *


def solve_random_cube(model_weights_path, shuffle_length=15):
    # Idea: traverse through tree until you find BETTER score than current one
    model = build_model()
    model.load_weights(model_weights_path)

    cube = generate_sample_cube(n_shuffles=shuffle_length)
    print("Randomly shuffled cube:")
    print(cube)

    solving_sequence = []
    last_move = None
    while not cube_is_solved(cube):
        print(f"Calculating step {len(solving_sequence) + 1}...")
        children_moves = get_children_moves(cube)
        children_scores = []

        reverse_last_move = get_reverse_move(last_move)
        for child, move in children_moves:
            if move != reverse_last_move:
                score = model.predict([flatten_one_hot(child)], verbose=0)
                children_scores.append((child, move, score))

        children_scores.sort(key=lambda t: t[2])
        move_to_take = children_scores[0][1]
        print(f"Score new Move {move_to_take}: {children_scores[0][2][0][0]}")
        cube = cube(move_to_take)
        solving_sequence.append(move_to_take)
        last_move = move_to_take

    print("Cube is solved!")
    print(f"Sequence: {solving_sequence}")
