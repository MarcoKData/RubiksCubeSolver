from typing import List


def perc_side_done(side: List):
    center_stone = side[1][1]

    count_correct_colors = 0
    total_stones = 0
    for row in side:
        for stone in row:
            total_stones += 1
            if stone == center_stone:
                count_correct_colors += 1

    return count_correct_colors / total_stones
