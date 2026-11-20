import os
import math
import numpy as np

# grid settings (same as the fusion step)
arena_min = -1.0
resolution = 0.05               # each cell is 5 cm
n_cells = 40                    # 40 x 40 cells for the 2 m x 2 m arena


def world_to_cell(x, y):
    # x, y in metres -> (row, col) of the cell
    col = int((x - arena_min) / resolution)
    row = int((y - arena_min) / resolution)
    col = min(max(col, 0), n_cells - 1)
    row = min(max(row, 0), n_cells - 1)
    return row, col


def cell_to_world(row, col):
    # centre of a cell in metres
    x = arena_min + (col + 0.5) * resolution
    y = arena_min + (row + 0.5) * resolution
    return x, y


# the 8 moves: (row change, col change, cost)
# straight moves cost 1, diagonal moves cost sqrt(2)
moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
         (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def get_neighbours(cell, grid):
    # gives the free neighbours of a cell and the cost to move there
    row, col = cell
    result = []
    for dr, dc, cost in moves:
        r2 = row + dr
        c2 = col + dc
        # stay inside the grid
        if r2 < 0 or r2 >= n_cells or c2 < 0 or c2 >= n_cells:
            continue
        # cannot move into a blocked cell
        if grid[r2, c2] == 1:
            continue
        # no cutting corners on diagonal moves
        if dr != 0 and dc != 0:
            if grid[row + dr, col] == 1 or grid[row, col + dc] == 1:
                continue
        result.append(((r2, c2), cost))
    return result


def load_planning_data(data_folder):
    # loads the planning grid and the start and goal cells saved by a2_grid_map.py
    grid = np.load(os.path.join(data_folder, "planning_grid.npy"))
    cells = np.load(os.path.join(data_folder, "start_goal_cells.npy"))
    start = (int(cells[0]), int(cells[1]))
    goal = (int(cells[2]), int(cells[3]))
    return grid, start, goal