from Grid_3 import Grid
from random import randint
import math
import numpy as np

default_initial_tiles = 2
default_probability = 0.9

action_dict = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

priority = [[6, 5, 4, 1],
            [5, 4, 1, 0],
            [4, 1, 0, -1],
            [1, 0, -1, -2]]

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
time_limit = 0.2
allowance = 0.05


class gym_2048():
    def __init__(self, size=4):
        self.size = size
        self.grid = None
        self.possible_new_tiles = [2, 4]
        self.probability = default_probability
        self.init_tiles = default_initial_tiles
        self.over = False
        self.action_space = 4
        self.reward = 0
        self.max_tile = 0
        self.score = 0
        pass

    def make(self):
        self.grid = Grid(self.size)
        for i in range(self.init_tiles):
            self.insert_random_tile()

    def get_state(self):
        return self.grid.map

    def reset(self):
        self.make()
        self.score = 0
        self.max_tile = 0
        self.reward = 0

    def get_new_tile_value(self):
        if randint(0, 99) < 100 * self.probability:
            return self.possible_new_tiles[0]
        else:
            return self.possible_new_tiles[1]

    def insert_random_tile(self):
        tile_value = self.get_new_tile_value()
        cells = self.grid.getAvailableCells()
        if len(cells) > 0:
            cell = cells[randint(0, len(cells) - 1)]
            self.grid.setCellValue(cell, tile_value)

    def close(self):
        self.grid = None

    def is_done(self):
        return not self.grid.canMove()

    def render(self, mode='human'):
        grid = self.grid
        if mode == "human":
            for i in range(grid.size):
                for j in range(grid.size):
                    print("%6d  " % grid.map[i][j], end="")
                print("")
            print("")
        # return self.get_state()

    def step(self, action):
        move = action
        # print(action_dict[move])
        # Validate Move
        if move is not None and 0 <= move < 4:
            moved, merge_points = self.grid.move(move)
            self.score = merge_points
            # Update maxTile
            self.max_tile = self.grid.getMaxTile()
            if moved:
                self.insert_random_tile()
            self.calculate_reward()
        else:
            print("Invalid PlayerAI Move - 1")
            exit(0)
        return self.get_state(), self.reward, self.is_done(), []

    def calculate_reward(self):
        # weighted_sum = self.calculate_weighted_sum()
        # penalty = self.calculate_penalty()
        # self.reward = weighted_sum - penalty
        self.reward = self.score

    def calculate_weighted_sum(self):
        score = 0
        for x in range(4):
            for y in range(4):
                score += (priority[x][y] * self.grid.map[x][y] * self.grid.map[x][y])
        return score

    def calculate_penalty(self):
        # Calculate clustering penalty.
        # We want two same valued tiles to be present next to each other
        # so that it is easier to merge.
        # This penalty becomes large when high valued tiles are scattered across
        # the grid, indicating that the grid is bad.

        def within_bounds(position):
            return 0 <= position['x'] < 4 and 0 <= position['y'] < 4

        penalty = 0

        # Direction vectors for up, down, left and right
        directions = ([-1, 0], [1, 0], [0, -1], [0, 1])

        for x in range(4):
            for y in range(4):
                # if grid.map[x][y] != 0:
                for i in range(4):
                    pos = {"x": x + directions[i][0],
                           "y": y + directions[i][1]}
                    if within_bounds(pos):
                        neighbour = self.grid.map[pos['x']][pos['y']]
                        # if neighbour!=0:
                        penalty += math.fabs(neighbour - self.grid.map[x][y])
        return penalty
