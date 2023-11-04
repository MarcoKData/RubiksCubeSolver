from .constants import POSSIBLE_ACTIONS
import pycuber as pc
import random


def generate_random_cube(n_shuffles=25):
    cube = pc.Cube()
    last_action = "F"  # random action to implement structure from first move onwards
    for _ in range(n_shuffles):
        last_action_reversed = (last_action + "'").replace("''", "")
        action = random.choice([action for action in POSSIBLE_ACTIONS if action != last_action_reversed])
        cube(action)
    
    return cube
