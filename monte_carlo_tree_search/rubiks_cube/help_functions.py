from pycuber import Cube


def cube_is_solved(cube: Cube):
    is_solved = True

    faces = ["F", "L", "R", "U", "D", "B"]
    for face_ltr in faces:
        face = cube.get_face(face_ltr)
        center_stone = face[1][1]
        for row in face:
            for stone in row:
                if stone != center_stone:
                    is_solved = False
    
    return is_solved


def get_children(cube):
    MOVES = ["F", "L", "R", "U", "D", "B", "F'", "L'", "R'", "U'", "D'", "B'"]

    children = []
    for move in MOVES:
        cube_copy = cube.copy()
        children.append((move, cube_copy(move)))
    
    return children


color_list_map = {'green': [1, 0, 0, 0, 0, 0], 'blue': [0, 1, 0, 0, 0, 0], 'yellow': [0, 0, 1, 0, 0, 0],
                  'red': [0, 0, 0, 1, 0, 0], 'orange': [0, 0, 0, 0, 1, 0], 'white': [0, 0, 0, 0, 0, 1]}
def flatten_one_hot(cube):
    sides = [cube.F, cube.B, cube.U, cube.D, cube.L, cube.R]
    flat = []
    for x in sides:
        for i in range(3):
            for j in range(3):
                flat.extend(color_list_map[x[i][j].colour])
    return flat
