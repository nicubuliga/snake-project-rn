import numpy
from ple import PLE
from ple.games.snake import Snake
import numpy as np
game = Snake(480, 480, 3)


p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

nb_frames = 1000
reward = 0.0
print(p.getActionSet())
while True:
    if p.game_over():  # check if the game is over
        p.reset_game()

    obs = p.getScreenRGB()
    action_index = int(np.random.rand() * 4)
    reward = p.act(p.getActionSet()[action_index])
