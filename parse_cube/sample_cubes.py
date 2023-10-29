import pycuber as pc
from .main import make_cube_from_flattened_sides


def get_sample_cube_ultimate_test():
    f = ["green", "green", "blue", "red", "green", "white", "orange", "white", "white"]
    r = ["red", "blue", "yellow", "green", "orange", "blue", "red", "green", "white"]
    l = ["red", "yellow", "yellow", "yellow", "red", "white", "green", "blue", "blue"]
    b = ["red", "blue", "white", "red", "blue", "red", "orange", "green", "white"]
    u = ["blue", "yellow", "green", "orange", "yellow", "orange", "orange", "red", "yellow"]
    d = ["yellow", "orange", "green", "white", "white", "orange", "orange", "yellow", "blue"]

    cube = make_cube_from_flattened_sides(f, r, l, b, u, d)
    print(cube)

    return cube


def get_sample_cube_solved():
    f = ["yellow", "yellow", "yellow", "green", "green", "white", "green", "green", "white"]
    r = ["red", "blue", "blue", "orange", "orange", "white", "orange", "orange", "white"]
    l = ["yellow", "green", "orange", "yellow", "red", "white", "red", "red", "white"]
    b = ["yellow", "yellow", "red", "blue", "blue", "red", "blue", "blue", "white"]
    u = ["blue", "orange", "orange", "red", "yellow", "yellow", "green", "green", "green"]
    d = ["orange", "orange", "blue", "white", "white", "blue", "green", "red", "red"]

    cube = make_cube_from_flattened_sides(f, r, l, b, u, d)
    print(cube)

    return cube
