from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import math

model = Sequential()
# model.add(Dense(6, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(5, activation='linear'))

model.compile(optimizer="adam", loss="mse")


def predict(state, game, env):
    x = get_input(state, game, env)
    return model.predict(x.reshape(1, len(x)), batch_size=1)


def train(current_state, game, env, t):
    x = get_input(current_state, game, env)
    # print(x)
    # print(t)
    model.fit(x.reshape(1, len(x)), t.reshape(
        1, 5), epochs=1, verbose=0, batch_size=1)


def save_model():
    model.save("trained_model")


def get_input(state, game, env):
    max_error = 0.9
    min_error = 0.1
    height = 480
    width = 480
    head_x = state['snake_head_x']
    head_y = state['snake_head_y']
    food_x = state['food_x']
    food_y = state['food_y']
    is_obstacle_left = 0
    is_obstacle_right = 0
    is_obstacle_front = 0
    is_food_front = 0
    is_food_left = 0
    is_food_right = 0
    direction = (game.player.dir.x, game.player.dir.y)

    if direction == (0, -1):  # up
        if food_x < head_x:
            is_food_left = 1
        elif food_x > head_x:
            is_food_right = 1
        else:
            is_food_front = 1

        if head_y < height * min_error:
            is_obstacle_front = 1
        if head_x < width * min_error:
            is_obstacle_left = 1
        if head_x > width * max_error:
            is_obstacle_right = 1
    elif direction == (0, 1):  # down
        if food_x > head_x:
            is_food_left = 1
        elif food_x < head_x:
            is_food_right = 1
        else:
            is_food_front = 1
        if head_y > height * max_error:
            is_obstacle_front = 1
        if head_x > width * max_error:
            is_obstacle_left = 1
        if head_x < width * min_error:
            is_obstacle_right = 1
    elif direction == (-1, 0):  # left
        if food_y > head_y:
            is_food_left = 1
        elif food_y < head_y:
            is_food_right = 1
        else:
            is_food_front = 1
        if head_x < width * min_error:
            is_obstacle_front = 1
        if head_y > height * max_error:
            is_obstacle_left = 1
        if head_y < height * min_error:
            is_obstacle_right = 1
    elif direction == (1, 0):  # right
        if food_y < head_y:
            is_food_left = 1
        elif food_y > head_y:
            is_food_right = 1
        else:
            is_food_front = 1
        if head_x > width * max_error:
            is_obstacle_front = 1
        if head_y < height * min_error:
            is_obstacle_left = 1
        if head_y > height * max_error:
            is_obstacle_right = 1

    return np.asarray([is_obstacle_left, is_obstacle_front, is_obstacle_right, is_food_front, is_food_right, is_food_left])
