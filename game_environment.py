import numpy
from ple import PLE
from ple.games.snake import Snake
import numpy as np
import q_agent as agent
import random
import json
from collections import deque

game = Snake(480, 480, 3)
gamma = 0.95
memory = deque(maxlen=2500)
epsilon = 1.0
min_epsilon = 0.01
epsilon_factor = 0.995
mini_batch_size = 500

env = PLE(game, fps=30, display_screen=False, force_fps=True, reward_values={
    "positive": 10.0,
    "negative": -100.0,
    "tick": 0.0,
    "loss": -100.0,
    "win": 5.0
})
env.init()

nr_games = 30
max_states = 10000


def experience_replay():
    global epsilon

    if len(memory) < mini_batch_size:
        return
    mini_batch = random.sample(memory, mini_batch_size)
    x_train = []
    y_train = []
    for instance in mini_batch:
        # Get max_Q(S',a)
        old_state, action, reward, new_state = instance
        old_qval = agent.predict(old_state, game, env)
        new_qval = agent.predict(new_state, game, env)
        max_q = np.max(new_qval)
        y = np.zeros((1, 5))
        y[:] = old_qval[:]
        if reward != -1:  # non-terminal state
            update = (reward + (gamma * max_q))
        else:  # terminal state
            update = reward
        y[0][action] = update
        x = agent.get_input(old_state, game, env)
        x_train.append(x.reshape(1, len(x)))
        y_train.append(y)

    agent.train_batch(x_train, y_train, batch_size=mini_batch_size)
    if epsilon > min_epsilon:
        epsilon *= epsilon_factor


def train():
    global memory
    global epsilon
    try:
        for i in range(nr_games):
            env.reset_game()
            print()

            for j in range(max_states):
                current_state = env.getGameState()
                q_vals = agent.predict(current_state, game, env)

                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, 5)
                else:
                    action = np.argmax(q_vals)

                reward = env.act(env.getActionSet()[action])
                new_state = env.getGameState()

                memory.append((current_state, action, reward, new_state))
                experience_replay()
    except:
        agent.save_model()


train()
