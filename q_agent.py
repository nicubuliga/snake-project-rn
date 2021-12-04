from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math

model = Sequential()
model.add(Dense(128, input_shape=(12,), activation='relu'))
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
    height = 480.0
    width = 480.0
    min_rate = 0.05
    max_rate = 0.95

    head_snake_x = state['snake_head_x']
    head_snake_y = state['snake_head_y']
    food_x = state['food_x']
    food_y = state['food_y']

    food_above = (food_y < head_snake_y)
    food_below = (food_y > head_snake_y)
    food_right = (food_x > head_snake_x)
    food_left = (food_x < head_snake_x)

    direction = (game.player.dir.x, game.player.dir.y)
    direction_up = (direction == (0, -1))
    direction_down = (direction == (0, 1))
    direction_left = (direction == (-1, 0))
    direction_right = (direction == (1, 0))

    wall_up = (head_snake_y <= min_rate * height)
    wall_down = (head_snake_y >= max_rate * height)
    wall_left = (head_snake_x <= min_rate * width)
    wall_right = (head_snake_y >= max_rate * width)

    return np.asarray([food_above, food_below, food_left, food_right, direction_right, direction_left, direction_up,
                       direction_down, wall_right, wall_left, wall_up, wall_down])
