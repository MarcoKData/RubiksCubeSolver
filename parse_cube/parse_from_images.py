import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from .main import make_cube_from_flattened_sides


# specific to image --> change when real setup done
APPROX_COLORS = {
    "red": np.array([237, 48, 48]),
    "green": np.array([88, 213, 103]),
    "blue": np.array([28, 96, 254]),
    "yellow": np.array([242, 242, 21]),
    "orange": np.array([232, 158, 20]),
    "white": np.array([255, 255, 255])
}

# specific to image --> change when real setup done
POSITIONS_MARKERS = [
    (50, 50),
    (50, 150),
    (50, 250),
    (150, 50),
    (150, 150),
    (150, 250),
    (250, 50),
    (250, 150),
    (250, 250)
]


def get_color_closest_to_pixel_value(pixel_value):
    if np.mean(pixel_value) <= 1:
        # Pixel value as floats --> scale to integers (255)
        pixel_value *= 255
    if len(pixel_value) == 4:
        # rgba --> clip to rgb
        pixel_value = pixel_value[:3]
    
    differences = {}
    for color, px_baseline in APPROX_COLORS.items():
        diff = np.abs(px_baseline - pixel_value).mean()
        differences[color] = diff
    
    differences_items = list(differences.items())
    differences_items.sort(key=lambda x: x[1])
    color = differences_items[0][0]
    
    return color


def parse_colors_from_img(img):
    colors_flattened = [get_color_closest_to_pixel_value(img[marker[0]][marker[1]]) for marker in POSITIONS_MARKERS]
    return colors_flattened

def marker_helper_function(img, marker_pos, surrounding):
    for i in range(surrounding):
        for j in range(surrounding):
            for c in range(3):
                img[marker_pos[0] + i][marker_pos[1] + j][c] = 0
    
    return img


def cube_from_side_imgs(f_img, r_img, l_img, b_img, u_img, d_img):
    cube = make_cube_from_flattened_sides(
        f=parse_colors_from_img(f_img),
        r=parse_colors_from_img(r_img),
        l=parse_colors_from_img(l_img),
        b=parse_colors_from_img(b_img),
        u=parse_colors_from_img(u_img),
        d=parse_colors_from_img(d_img)
    )

    return cube
