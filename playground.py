import rubiks_ai as ai
from keras.utils import plot_model
import matplotlib.pyplot as plt


model = ai.build_model()
plot_model(model)
plt.show()
