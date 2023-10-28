import parse_cube as parser
import os
from matplotlib import image


f_path = os.path.join(".", "example_imgs", "f.png")
b_path = os.path.join(".", "example_imgs", "b.png")
d_path = os.path.join(".", "example_imgs", "d.png")
l_path = os.path.join(".", "example_imgs", "l.png")
r_path = os.path.join(".", "example_imgs", "r.png")
u_path = os.path.join(".", "example_imgs", "u.png")

f_img = image.imread(f_path)
b_img = image.imread(b_path)
d_img = image.imread(d_path)
l_img = image.imread(l_path)
r_img = image.imread(r_path)
u_img = image.imread(u_path)

cube = parser.cube_from_side_imgs(f_img, r_img, l_img, b_img, u_img, d_img)
print(cube)
