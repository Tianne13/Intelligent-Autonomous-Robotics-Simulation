import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from grid_map import (n_cells, resolution, arena_min,
                      world_to_cell, cell_to_world, get_neighbours)

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# ------------------------------------------------------------
# 1. load the fused map and inflate the obstacles
# ------------------------------------------------------------
fused_grid = np.load(os.path.join(data_folder, "fused_grid.npy"))

# wall, obstacles and the landmark are all blocked (1), free cells are 0
blocked = np.uint8(fused_grid > 0)

# the planner treats the robot as a point, so the obstacles are made bigger
# the e-puck radius is about 0.037 m which is less than one cell (0.05 m)
robot_radius = 0.037
inflate_cells = 1
kernel = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), np.uint8)
planning_grid = cv2.dilate(blocked, kernel)

print("Grid size:", n_cells, "x", n_cells, "cells, cell size", resolution, "m")
print("Robot radius:", robot_radius, "m, inflation:", inflate_cells * resolution, "m")
print("Blocked cells before inflation:", int(blocked.sum()))
print("Blocked cells after inflation: ", int(planning_grid.sum()))

# ------------------------------------------------------------
# 2. start and goal cells
# ------------------------------------------------------------
start_world = (-0.8, -0.8)
defined_goal = (0.8, 0.8)

# the goal comes from the camera triangulation in the fusion step
goal_file = os.path.join(data_folder, "goal_estimate.npy")
if os.path.exists(goal_file):
    goal_estimate = np.load(goal_file)
    goal_world = (float(goal_estimate[0]), float(goal_estimate[1]))
    print("Goal taken from the camera triangulation:", goal_world)
else:
    goal_world = defined_goal
    print("No goal estimate found, using the defined goal:", goal_world)


def nearest_free(cell, grid):
    # if the cell is blocked, find the closest free cell
    if grid[cell[0], cell[1]] == 0:
        return cell
    best = None
    best_dist = 9999
    for r in range(n_cells):
        for c in range(n_cells):
            if grid[r, c] == 0:
                d = math.sqrt((r - cell[0]) ** 2 + (c - cell[1]) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = (r, c)
    return best


start_cell = world_to_cell(start_world[0], start_world[1])
goal_cell = world_to_cell(goal_world[0], goal_world[1])
if planning_grid[start_cell[0], start_cell[1]] == 1:
    print("Start cell was blocked, moving it to the nearest free cell")
if planning_grid[goal_cell[0], goal_cell[1]] == 1:
    print("Goal cell was blocked, moving it to the nearest free cell")
start_cell = nearest_free(start_cell, planning_grid)
goal_cell = nearest_free(goal_cell, planning_grid)

print("Start cell (row, col):", start_cell)
print("Goal cell (row, col): ", goal_cell)

# ------------------------------------------------------------
# 3. the grid as a graph (nodes = free cells, edges = moves)
# ------------------------------------------------------------
n_nodes = 0
n_edges = 0
for r in range(n_cells):
    for c in range(n_cells):
        if planning_grid[r, c] == 0:
            n_nodes = n_nodes + 1
            n_edges = n_edges + len(get_neighbours((r, c), planning_grid))
n_edges = n_edges // 2          # every edge was counted from both ends

print("Graph nodes (free cells):", n_nodes)
print("Graph edges (8-connected):", n_edges)
print("Free space:", round(100 * n_nodes / (n_cells * n_cells), 1), "%")

# check that the goal can be reached from the start (simple flood fill)
visited = set([start_cell])
queue = [start_cell]
while len(queue) > 0:
    current = queue.pop(0)
    for (neighbour, cost) in get_neighbours(current, planning_grid):
        if neighbour not in visited:
            visited.add(neighbour)
            queue.append(neighbour)
print("Goal reachable from start:", goal_cell in visited)

# save for dijkstra and a star
np.save(os.path.join(data_folder, "planning_grid.npy"), planning_grid)
np.save(os.path.join(data_folder, "start_goal_cells.npy"),
        np.array([start_cell[0], start_cell[1], goal_cell[0], goal_cell[1]]))

# ------------------------------------------------------------
# 4. figure: planning grid and a zoom on the graph near the start
# ------------------------------------------------------------
display = np.zeros((n_cells, n_cells), dtype=np.uint8)
display[planning_grid == 1] = 1         # inflated margin
display[blocked == 1] = 2               # original blocked cells
cmap = ListedColormap(["white", "lightsalmon", "black"])

start_x, start_y = cell_to_world(start_cell[0], start_cell[1])
goal_x, goal_y = cell_to_world(goal_cell[0], goal_cell[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

# left: whole planning grid
ax1.imshow(display, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap, vmin=0, vmax=2)
ax1.plot(start_x, start_y, "gs", markersize=10, markeredgecolor="black", label="Start")
ax1.plot(goal_x, goal_y, marker="*", color="gold", markersize=16,
         markeredgecolor="black", linestyle="none", label="Goal")
handles, names = ax1.get_legend_handles_labels()
extra = [Patch(facecolor="black", label="Blocked (fused map)"),
         Patch(facecolor="lightsalmon", label="Inflated margin (1 cell)"),
         Patch(facecolor="white", edgecolor="gray", label="Free")]
ax1.legend(handles=handles + extra, loc="lower right", fontsize=8)
ax1.set_title("Planning occupancy grid (40 x 40 cells, 5 cm)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# right: zoom on the corner with the start, drawing the graph
r0, r1 = 0, 10
c0, c1 = 0, 10
zoom = display[r0:r1, c0:c1]
x_low = arena_min + c0 * resolution
x_high = arena_min + c1 * resolution
y_low = arena_min + r0 * resolution
y_high = arena_min + r1 * resolution
ax2.imshow(zoom, origin="lower", extent=[x_low, x_high, y_low, y_high],
           cmap=cmap, vmin=0, vmax=2)

for r in range(r0, r1):
    for c in range(c0, c1):
        if planning_grid[r, c] == 0:
            x, y = cell_to_world(r, c)
            for (neighbour, cost) in get_neighbours((r, c), planning_grid):
                if r0 <= neighbour[0] < r1 and c0 <= neighbour[1] < c1:
                    x2, y2 = cell_to_world(neighbour[0], neighbour[1])
                    ax2.plot([x, x2], [y, y2], color="steelblue", linewidth=0.8)
            ax2.plot(x, y, "o", color="navy", markersize=4)

ax2.plot(start_x, start_y, "gs", markersize=11, markeredgecolor="black")
ax2.set_title("Graph zoom near the start (nodes = cells, edges = moves)")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A2_planning_grid.png"), dpi=150)
plt.show()

print("Done. Figure saved in:", figures_folder)