from pycuber import Cube
from .main import ACTIONS
import random


def get_reversed_action(action: str) -> str:
    if action is None:
        return None
    
    return (action + "'").replace("''", "")


def get_single_scrambled_cube(num_scrambles: int) -> Cube:
    cube = Cube()

    last_action = None
    for _ in range(num_scrambles):
        action = random.choice(ACTIONS)
        while action == get_reversed_action(last_action):
            action = random.choice(ACTIONS)
        cube(action)

    return cube
