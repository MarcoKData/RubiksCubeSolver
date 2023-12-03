import json
import numpy as np


PATH_TO_TRAINING_TIMES = "/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/training_seconds.json"

with open(PATH_TO_TRAINING_TIMES, "r") as file:
    seconds = json.load(file)

hours = sum(seconds) / 3600

print(f"Hours trained: {np.round(hours, 4)}")
