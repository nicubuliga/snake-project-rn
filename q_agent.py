from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math

model = Sequential()
model.add(Dense(128, input_shape=(12, ), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))
opt = Adam(learning_rate=0.00025)
model.compile(optimizer=opt, loss="mse")


def predict(state, game, env):
    x = get_input(state, game, env)
    return model.predict(x.reshape(1, len(x)))


def train_batch(x_train, y_train, batch_size):
    model.fit(x_train, y_train, epochs=1, verbose=1)


def save_model():
    model.save("trained_model")


def get_input(state, game, env):
    max_error = 0.95
    min_error = 0.05
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
        if food_x > head_x:
            is_food_right = 1
        if food_y < head_y:
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
        if food_x < head_x:
            is_food_right = 1
        if food_y > head_y:
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
        if food_y < head_y:
            is_food_right = 1
        if food_x < head_x:
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
        if food_y > head_y:
            is_food_right = 1
        if food_x > head_x:
            is_food_front = 1

        if head_x > width * max_error:
            is_obstacle_front = 1
        if head_y < height * min_error:
            is_obstacle_left = 1
        if head_y > height * max_error:
            is_obstacle_right = 1

    return np.asarray([is_obstacle_left, is_obstacle_front, is_obstacle_right, is_food_front,
                       is_food_right, is_food_left, direction[0], direction[1]])
