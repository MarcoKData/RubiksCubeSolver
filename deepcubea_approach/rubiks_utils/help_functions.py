import numpy as np
import pycuber as pc


def is_final_cube_state(cube: pc.Cube) -> bool:
    is_final = True
    faces = [cube.F, cube.B, cube.L, cube.R, cube.U, cube.D]
    
    for face in faces:
        center_stone = face[1][1]
        for row in face:
            for stone in row:
                if stone != center_stone:
                    is_final = False
                    break
        
        if not is_final:
            break

    return is_final


def get_reversed_action(action: str) -> str:
    if action is None:
        return None
    
    return (action + "'").replace("''", "")


def get_children(cube: pc.Cube, excluded_moves = []):
    MOVES = ["F", "B", "U", "D", "L", "R", "F'", "B'", "U'", "D'", "L'", "R'"]
    # moves_to_use = [move for move in MOVES if move not in excluded_moves]
    moves_to_use = MOVES

    children = []
    for move in moves_to_use:
        cube_copy = cube.copy()
        children.append((move, cube_copy(move)))

    return children
