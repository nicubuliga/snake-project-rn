import numpy
from ple import PLE
from ple.games.snake import Snake
import numpy as np
import q_agent as agent
import json
from tensorflow import keras
model = keras.models.load_model('trained_model')

game = Snake(480, 480, 3)
gamma = 0.9

env = PLE(game, fps=30, display_screen=True, force_fps=False)
env.init()

for i in range(100):
    env.reset_game()
    over = False

    print(i)
    while not env.game_over():

        current_state = env.getGameState()
        x = agent.get_input(current_state, game, env)
        q_vals = model.predict(x.reshape(1, len(x)), batch_size=1)
        action = np.argmax(q_vals)
        # Act
        env.act(env.getActionSet()[action])
