from bot import RubiksCubeBot
import model_utils


model = model_utils.build_model(learning_rate=1e-3)
bot = RubiksCubeBot(model=model)

samples = bot.generate_samples(n_samples=2 * 32)
cube_action_rewards = bot.generate_cube_action_rewards(samples)
filtered = bot.filter_cube_action_rewards(cube_action_rewards, percentage_best=0.25)
X, y = bot.preprocess_filtered(filtered)

bot.train_with_test_split(X, y, epochs=10)
