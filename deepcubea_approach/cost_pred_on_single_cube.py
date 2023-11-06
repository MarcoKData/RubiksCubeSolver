import model_utils as m_utils
import data_utils as data


model = m_utils.build_model()
model.load_weights("/Users/marcokleimaier/Documents/Projekte/RubiksCubeSolver/deepcubea_approach/saved_models/model_copy.h5")

sample_cube = data.get_single_scrambled_cube(num_scrambles=5)
print(sample_cube)

sample_cube = data.flatten_one_hot(sample_cube).reshape((1, -1))
pred = model.predict(sample_cube)
print(pred)
