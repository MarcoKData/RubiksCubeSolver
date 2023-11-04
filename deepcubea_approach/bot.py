import rubiks_utils
import reward_utils
import model_utils
from typing import List, Dict, Tuple
from keras.models import Model
import numpy as np
from pycuber import Cube
from sklearn.model_selection import train_test_split


class RubiksCubeBot():
    def __init__(self, model: Model) -> None:
        self.model = model
    
    def predict_action_from_cube(self, cube: Cube) -> str:
        policy_network = self.model.predict([model_utils.flatten_one_hot(cube)], verbose=0)[0]
        action_idx_network = np.argmax(policy_network)
        action_network = rubiks_utils.POSSIBLE_ACTIONS[action_idx_network]
        
        return action_network

    def generate_samples(self, n_samples: int) -> List[Dict]:
        print("Generating samples...")
        samples = []
        for _ in range(n_samples):
            cube = rubiks_utils.generate_random_cube()
            action = self.predict_action_from_cube(cube)
            new_cube = cube.copy()
            new_cube(action)

            samples.append({
                "cube1": cube,
                "action": action,
                "cube2": new_cube
            })

        return samples
    
    def generate_cube_action_rewards(self, samples: List[Dict]) -> List[Dict]:
        print("Calculating Rewards...")
        cube_action_rewards = []
        for sample in samples:
            reward_resulting_cube = reward_utils.get_reward(sample["cube2"])
            cube_action_rewards.append({
                "cube": sample["cube1"],
                "action": sample["action"],
                "reward": reward_resulting_cube
            })
        
        return cube_action_rewards
    
    def filter_cube_action_rewards(self, cube_action_rewards: List[Dict], percentage_best: float) -> List[Dict]:
        print("Filtering Samples...")
        num_best = int(percentage_best * len(cube_action_rewards))
        cube_action_rewards.sort(key=lambda entry: entry["reward"], reverse=True)
        
        return cube_action_rewards[:num_best]
    
    def preprocess_filtered(self, filtered_c_a_r: List[Dict]) -> Tuple:
        print("Preprocessing Data for Training...")
        flattened_X_cubes = np.array([model_utils.flatten_one_hot(c_a_r["cube"]) for c_a_r in filtered_c_a_r])
        y_not_one_hot = model_utils.actions_to_one_hot([c_a_r["action"] for c_a_r in filtered_c_a_r])
        
        return flattened_X_cubes, y_not_one_hot
    
    def train_with_test_split(self, X: np.array, y: np.array, epochs: int, batch_size: int = 32, test_size: float = 0.3) -> None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=187)
        
        self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32
        )
