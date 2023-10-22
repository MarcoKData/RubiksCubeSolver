import pycuber as pc


def make_cube_from_flattened_sides(f, r, l, b, u, d):
    # order: l u f d r b
    cube_flattened = l + u + f + d + r + b
    cubies = pc.array_to_cubies(cube_flattened)
    cube = pc.Cube(cubies)
    
    return cube
