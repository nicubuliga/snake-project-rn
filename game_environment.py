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


def get_distance(state):
    head_snake_x = state['snake_head_x']
    head_snake_y = state['snake_head_y']
    food_x = state['food_x']
    food_y = state['food_y']

    a = np.asarray([head_snake_x, head_snake_y])
    b = np.asarray([food_x, food_y])

    return np.linalg.norm(a - b)


def train():
    global memory
    global epsilon
    try:
        for i in range(nr_games):
            env.reset_game()
            print()

            for j in range(max_states):
                if env.game_over():
                    break
                current_state = env.getGameState()
                old_distance = get_distance(current_state)
                q_vals = agent.predict(current_state, game, env)

                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, 5)
                else:
                    action = np.argmax(q_vals)

                reward = env.act(env.getActionSet()[action])
                new_state = env.getGameState()
                new_distance = get_distance(new_state)

                if new_distance <= old_distance:
                    reward += 1
                else:
                    reward -= 1

                memory.append((current_state, action, reward, new_state))
                experience_replay()

            print("Score: {}".format(env.score()))
    except:
        agent.save_model()


train()
