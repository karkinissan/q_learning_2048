from gym_2048 import gym_2048
import random
import numpy as np
random.seed(1234)
game = gym_2048()

game.make()
game.reset()
game.render()
max_tiles = []
rewards = []
for i in range(2000):
    print(i)
    _, reward, done, _ = game.step(random.randint(0,3))
    game.render()
    max_tiles.append(game.max_tile)
    rewards.append(reward)
    print(reward)
    if done: game.reset()
print(max(max_tiles))
print(max(rewards))