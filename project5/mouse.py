import numpy as np

# Grid parameters
GRID_SIZE = 5

# Elements on the grid
EMPTY = 0
MOUSE = 1
CHEESE = 2
TRAP = 3
WALL = 4
ORGANIC_CHEESE = 5

# Numbers of special elements
NUM_TRAPS = 2
NUM_WALLS = 2
NUM_ORGANIC_CHEESE = 1
NUM_CHEESE = 2

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def initialize_grid_with_cheese_types(
    grid_size=GRID_SIZE,
    num_traps=NUM_TRAPS,
    num_walls=NUM_WALLS,
    num_cheese=NUM_CHEESE,
    num_organic_cheese=NUM_ORGANIC_CHEESE,
):
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Randomly place mouse
    while True:
        mouse_pos = tuple(np.random.randint(0, grid_size, size=2))
        if grid[mouse_pos] == EMPTY:
            grid[mouse_pos] = MOUSE
            break

    # Normal cheese
    for _ in range(num_cheese):
        while True:
            pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[pos] == EMPTY:
                grid[pos] = CHEESE
                break

    # Organic cheese
    for _ in range(num_organic_cheese):
        while True:
            pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[pos] == EMPTY:
                grid[pos] = ORGANIC_CHEESE
                break

    # Traps
    for _ in range(num_traps):
        while True:
            pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[pos] == EMPTY:
                grid[pos] = TRAP
                break

    # Walls
    for _ in range(num_walls):
        while True:
            pos = tuple(np.random.randint(0, grid_size, size=2))
            if grid[pos] == EMPTY:
                grid[pos] = WALL
                break

    return grid, mouse_pos, None, None


def get_reward(pos, grid):
    if grid[pos] == CHEESE or grid[pos] == ORGANIC_CHEESE:
        return 10
    elif grid[pos] == TRAP:
        return -50
    else:
        return -0.2


def move(action, grid):
    delta = ACTION_TO_DELTA[action]
    mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])
    new_pos = (mouse_pos[0] + delta[0], mouse_pos[1] + delta[1])

    # Bounds + wall check
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
        if grid[new_pos] != WALL:
            reward = get_reward(new_pos, grid)  # reward BEFORE overwriting
            grid[mouse_pos] = EMPTY
            grid[new_pos] = MOUSE
            return grid, reward

    return grid, -0.2  # bump penalty
