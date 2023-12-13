from typing import List, Dict
from keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score


def get_metrics(cubes: List, distances_classes: List, model: Model) -> Dict:
    metrics = {}

    # get Preds
    cubes = np.array(cubes)
    preds = model.predict(cubes, verbose=0)
    y_pred = [np.argmax(pred) for pred in preds]
    y_true = [np.argmax(value) for value in distances_classes]

    # Accuracy
    metrics["acc"] = accuracy_score(y_true, y_pred)

    return metrics
