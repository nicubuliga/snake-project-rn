import numpy
from ple import PLE
from ple.games.snake import Snake
import numpy as np
import q_agent as agent
import json

game = Snake(480, 480, 3)
gamma = 0.95


env = PLE(game, fps=30, display_screen=True, force_fps=False, reward_values={
    "positive": 1.0,
    "negative": -1.0,
    "tick": 0.0,
    "loss": -1,
    "win": 5.0
})
env.init()

nr_games = 10000


def train():
    epsilon = 1.0
    for i in range(nr_games):
        env.reset_game()

        print(i)
        while not env.game_over():
            current_state = env.getGameState()
            # Predict Q value for every possible action
            q_vals = agent.predict(current_state, game, env)
            # print(q_vals)
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 5)
            else:
                action = np.argmax(q_vals)
            # Act
            reward = env.act(env.getActionSet()[action])
            new_state = env.getGameState()
            new_q_vals = agent.predict(new_state, game, env)
            new_max_q = np.max(new_q_vals)
            # print(new_max_q)
            # Wanted output
            t = q_vals

            t[0][action] = reward + gamma * new_max_q

            agent.train(current_state, game, env, t)
        if epsilon > 0.1:
            epsilon *= 0.98
        print("EPSILON = {} --- SCORE = {}".format(epsilon, env.score()))
    agent.save_model()


train()
