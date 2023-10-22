import pycuber as pc
from help_functions import make_cube_from_flattened_sides


f = ["red", "blue", "yellow", "yellow", "yellow", "red", "white", "red", "blue"]
r = ["green", "green", "red", "yellow", "green", "red", "blue", "white", "white"]
l = ["yellow", "red", "white", "white", "white", "blue", "yellow", "white", "blue"]
b = ["blue", "blue", "green", "green", "blue", "red", "yellow", "yellow", "red"]
u = ["orange", "orange", "orange", "blue", "orange", "orange", "red", "red", "white"]
d = ["white", "blue", "blue", "yellow", "blue", "red", "orange", "red", "red"]

print(make_cube_from_flattened_sides(f, r, l, b, u, d))
