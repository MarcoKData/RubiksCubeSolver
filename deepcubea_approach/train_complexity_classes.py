import data_utils as data
import model_utils as m_utils
from sklearn.model_selection import train_test_split


def train_complexity_classes():
    cubes, distance_classes, num_classes = data.get_data_complexity_classes_f(
        num_sequences=200,
        num_scrambles=30,
        up_to=25,
        step_width=3
    )
    X_train, X_test, y_train, y_test = train_test_split(cubes, distance_classes, test_size=0.3)
    for i in range(10):
        print(y_train[i])

    model = m_utils.build_model_simple_cc(num_classes)
    
    model.fit(
        x=X_train,
        y=y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
    )


if __name__ == "__main__":
    train_complexity_classes()
