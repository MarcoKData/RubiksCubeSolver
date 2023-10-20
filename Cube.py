import numpy as np
import matplotlib.pyplot as plt


COLOR_ID_TO_RGB = {
    1: [0, 112, 21],
    2: [120, 0, 120],
    3: [255, 255, 255],
    4: [255, 55, 0],
    5: [0, 56, 209],
    6: [255, 255, 0],
    0: [0, 0, 0]
}

ACTION_SPACE = ["F", "R", "U", "L", "B", "D", "F_dash", "R_dash", "U_dash", "L_dash", "B_dash", "D_dash"]


class Cube:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.flattened_cube = np.zeros(shape=(9, 12))

        # set up ordered/solved cube
        c = 0
        for i in range(3):
            for j in range(3):
                c += 1
                # left
                self.flattened_cube[3 + i][j] = 1
                # top
                self.flattened_cube[i][3 + j] = 2
                # middle
                self.flattened_cube[3 + i][3 + j] = 3
                # bottom
                self.flattened_cube[6 + i][3 + j] = 4
                # right
                self.flattened_cube[3 + i][6 + j] = 5
                # right-right
                self.flattened_cube[3 + i][9 + j] = 6

    def mix(self, sequence_length=10) -> None:
        sequence = []
        for _ in range(sequence_length):
            sequence.append(np.random.choice(ACTION_SPACE))
        
        flattened_cubes = self.execute_sequence(sequence)
        sequence_idxes = [ACTION_SPACE.index(el) for el in sequence]

        return flattened_cubes, sequence_idxes

    def execute_sequence(self, sequence):
        cube_states = []

        for action in sequence:
            # print(f"Executing: {action}")
            if action == "F":
                self.F()
            elif action == "R":
                self.R()
            elif action == "U":
                self.U()
            elif action == "L":
                self.L()
            elif action == "B":
                self.B()
            elif action == "D":
                self.D()
            elif action == "F_dash":
                self.F_dash()
            elif action == "R_dash":
                self.R_dash()
            elif action == "U_dash":
                self.U_dash()
            elif action == "L_dash":
                self.L_dash()
            elif action == "B_dash":
                self.B_dash()
            elif action == "D_dash":
                self.D_dash()
            
            cube_states.append(self.flattened_cube)
        
        return cube_states
    
    def execute_action_idx(self, idx):
        sequence = [ACTION_SPACE[idx]]
        self.execute_sequence(sequence)

    def F(self) -> None:
        # outer rim
        top_row_backup = np.copy(self.flattened_cube[2][3:6])
        left_row_backup = np.copy(self.flattened_cube[3:6][:, 2])
        bottom_row_backup = np.copy(self.flattened_cube[6][3:6])
        right_row_backup = np.copy(self.flattened_cube[3:6][:, 6])
        
        self.flattened_cube[3:6][:,2] = bottom_row_backup
        self.flattened_cube[6][3:6] = right_row_backup
        self.flattened_cube[3:6][:,6] = top_row_backup
        self.flattened_cube[2][3:6] = left_row_backup

        # inner area
        self.rotate_inner_area_cw(starting_point=(3, 3))
    
    def F_dash(self):
        for _ in range(3):
            self.F()

    def R(self) -> None:
        # outer rim
        right_right_backup = np.flip(np.copy(self.flattened_cube[3:6][:, 9]))

        upper_right_backup = np.flip(np.copy(self.flattened_cube[:3][:, 5]))
        self.flattened_cube[3:6][:, 9] = upper_right_backup

        middle_right_backup = np.copy(self.flattened_cube[3:6][:, 5])
        self.flattened_cube[:3][:, 5] = middle_right_backup

        lower_right_backup = np.copy(self.flattened_cube[6:9][:, 5])
        self.flattened_cube[3:6][:, 5] = lower_right_backup

        self.flattened_cube[6:9][:, 5] = right_right_backup

        # inner area
        self.rotate_inner_area_cw(starting_point=(3, 6))
    
    def R_dash(self):
        for _ in range(3):
            self.R()
    
    def L(self):
        # outer rim
        upper_right_backup = np.copy(self.flattened_cube[:3][:, 3])
        middle_right_backup = np.copy(self.flattened_cube[3:6][:, 3])
        lower_right_backup = np.copy(self.flattened_cube[6:][:, 3])
        right_right_right_backup = np.copy(self.flattened_cube[3:6][:, -1])
        
        self.flattened_cube[:3][:, 3] = np.flip(right_right_right_backup)
        self.flattened_cube[3:6][:, 3] = upper_right_backup
        self.flattened_cube[6:9][:, 3] = middle_right_backup
        self.flattened_cube[3:6][:, -1] = np.flip(lower_right_backup)

        # inner area
        self.rotate_inner_area_cw(starting_point=(3, 0))
    
    def L_dash(self):
        for _ in range(3):
            self.L()
    
    def B(self):
        # outer rim
        upper_upper_row_backup = np.copy(self.flattened_cube[0][3:6])
        left_left_row_backup = np.copy(self.flattened_cube[3:6][:, 0])
        lower_lower_row_backup = np.copy(self.flattened_cube[-1][3:6])
        right_right_row_backup = np.copy(self.flattened_cube[3:6][:, 8])

        self.flattened_cube[0][3:6] = right_right_row_backup
        self.flattened_cube[3:6][:, 0] = np.flip(upper_upper_row_backup)
        self.flattened_cube[-1][3:6] = left_left_row_backup
        self.flattened_cube[3:6][:, 8] = np.flip(lower_lower_row_backup)

        # inner area
        self.rotate_inner_area_cw(starting_point=(3, 9))
    
    def B_dash(self):
        for _ in range(3):
            self.B()

    def U(self) -> None:
        # outer rim
        middle_upper_backup = np.copy(self.flattened_cube[3][3:6])
        left_upper_backup = np.copy(self.flattened_cube[3][:3])
        right_upper_backup = np.copy(self.flattened_cube[3][6:9])
        right_right_upper_backup = np.copy(self.flattened_cube[3][9:])

        self.flattened_cube[3][9:] = left_upper_backup
        self.flattened_cube[3][6:9] = right_right_upper_backup
        self.flattened_cube[3][3:6] = right_upper_backup
        self.flattened_cube[3][:3] = middle_upper_backup

        # inner area
        self.rotate_inner_area_cw(starting_point=(0, 3))
    
    def U_dash(self):
        for _ in range(3):
            self.U()
    
    def D(self):
        # outer rim
        left_lower_backup = np.copy(self.flattened_cube[5][:3])
        middle_lower_backup = np.copy(self.flattened_cube[5][3:6])
        right_lower_backup = np.copy(self.flattened_cube[5][6:9])
        right_right_lower_backup = np.copy(self.flattened_cube[5][9:])

        self.flattened_cube[5][:3] = right_right_lower_backup
        self.flattened_cube[5][3:6] = left_lower_backup
        self.flattened_cube[5][6:9] = middle_lower_backup
        self.flattened_cube[5][9:] = right_lower_backup

        # inner area
        self.rotate_inner_area_cw(starting_point=(6, 3))
    
    def D_dash(self):
        for _ in range(3):
            self.D()

    # HELPER FUNCTIONS
    def rotate_inner_area_cw(self, starting_point):
        inner_area = np.copy(self.flattened_cube[starting_point[0]:starting_point[0] + 3][:, starting_point[1]:starting_point[1] + 3])
        inner_rotated = np.rot90(inner_area, 3)

        for i in range(3):
            for j in range(3):
                self.flattened_cube[starting_point[0] + i][starting_point[1] + j] = inner_rotated[i][j]
    
    def is_solved(self):
        solved_state = np.zeros(shape=(9, 12))

        # set up ordered/solved cube
        for i in range(3):
            for j in range(3):
                # left
                self.flattened_cube[3 + i][j] = 1
                # top
                self.flattened_cube[i][3 + j] = 2
                # middle
                self.flattened_cube[3 + i][3 + j] = 3
                # bottom
                self.flattened_cube[6 + i][3 + j] = 4
                # right
                self.flattened_cube[3 + i][6 + j] = 5
                # right-right
                self.flattened_cube[3 + i][9 + j] = 6
        
        result = True
        for i in range(self.flattened_cube.shape[0]):
            for j in range(self.flattened_cube.shape[1]):
                if self.flattened_cube[i][j] != solved_state[i][j]:
                    result = False
        
        return result

    def plot(self) -> None:
        flattened_as_img = np.zeros(shape=(9, 12, 3))
        for i in range(len(self.flattened_cube)):
            for j in range(len(self.flattened_cube[i])):
                flattened_as_img[i][j] = np.array(COLOR_ID_TO_RGB[self.flattened_cube[i][j]]) / 255.0

        plt.imshow(flattened_as_img)
        plt.show()
