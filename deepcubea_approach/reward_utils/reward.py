from pycuber import Cube
from .help_funcs import *


def get_reward(cube: Cube):
    # give reward for only one side
    sides = [cube.F, cube.B, cube.L, cube.R, cube.U, cube.D]
    percentages_sides_done = []

    for side in sides:
        percentages_sides_done.append(perc_side_done(side))

    return max(percentages_sides_done)
