from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import math

layer_sizes = [32]
model = Sequential()
for i in range(len(layer_sizes)):
    if i == 0:
        model.add(Dense(layer_sizes[i], input_shape=(13,), activation='relu'))
    else:
        model.add(Dense(layer_sizes[i], activation='relu'))
model.add(Dense(5, activation='linear'))
# rms = RMSprop()
adam = Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=adam)


def predict(state, game, env):
    x = get_input(state, game, env)
    return model.predict(x.reshape(1, len(x)))


def predict_on_batch(next_states):
    return model.predict_on_batch(next_states)


def train_batch(x_train, y_train):
    model.fit(x=x_train, y=y_train, epochs=1, verbose=0)


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

    food_above = float(food_y < head_snake_y)
    food_below = float(food_y > head_snake_y)
    food_right = float(food_x > head_snake_x)
    food_left = float(food_x < head_snake_x)

    direction = (game.player.dir.x, game.player.dir.y)
    direction_up = float(direction == (0, -1))
    direction_down = float(direction == (0, 1))
    direction_left = float(direction == (-1, 0))
    direction_right = float(direction == (1, 0))

    wall_up = float(head_snake_y <= min_rate * height)
    wall_down = float(head_snake_y >= max_rate * height)
    wall_left = float(head_snake_x <= min_rate * width)
    wall_right = float(head_snake_x >= max_rate * width)

    close_to_body = 0.0
    # suma = 0.0
    # for dist in state['snake_body']:
    #     suma += dist / 480.0
    # # suma /= len(state['snake_body'])
    for index in range(2, len(state['snake_body'])):
        if state['snake_body'][index] <= state['snake_body'][1]:
            close_to_body = 1.0
            break

    return np.asarray([food_above, food_below, food_left, food_right, direction_right, direction_left, direction_up,
                       direction_down, wall_right, wall_left, wall_up, wall_down, close_to_body])
