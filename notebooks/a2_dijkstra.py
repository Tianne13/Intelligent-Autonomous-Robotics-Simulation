import os
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from grid_map import n_cells, resolution, cell_to_world, get_neighbours, load_planning_data


def dijkstra(grid, start, goal):
    # finds the cheapest path from start to goal
    # gives back the path (list of cells), the cost and the cells it expanded in order

    queue = []                                  # priority queue of (cost, cell)
    heapq.heappush(queue, (0.0, start))
    dist = {start: 0.0}                         # best cost found so far for each cell
    came_from = {start: None}                   # to rebuild the path at the end
    closed = set()                              # cells that are finished
    explored = []                               # order the cells were expanded in

    while len(queue) > 0:
        cost, current = heapq.heappop(queue)

        # the same cell can be in the queue twice, skip the old entry
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)

        # stop when the goal is taken out of the queue
        if current == goal:
            break

        for (neighbour, step_cost) in get_neighbours(current, grid):
            new_cost = cost + step_cost
            if neighbour not in dist or new_cost < dist[neighbour]:
                dist[neighbour] = new_cost
                came_from[neighbour] = current
                heapq.heappush(queue, (new_cost, neighbour))

    # no path found
    if goal not in closed:
        return None, None, explored

    # go backwards from the goal to get the path
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    path.reverse()
    return path, dist[goal], explored


# only runs when this file is run directly (not when it is imported)
if __name__ == "__main__":
    # folders
    this_folder = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.dirname(this_folder)
    data_folder = os.path.join(project_folder, "data")
    figures_folder = os.path.join(project_folder, "figures")
    os.makedirs(figures_folder, exist_ok=True)

    # load the grid, start and goal from step 6
    grid, start, goal = load_planning_data(data_folder)
    n_free = int(np.sum(grid == 0))
    print("Start cell:", start, " Goal cell:", goal)
    print("Free cells in the grid:", n_free)

    # run it once to get the result
    path, cost, explored = dijkstra(grid, start, goal)
    if path is None:
        print("No path found!")
        raise SystemExit

    # the timing for one run is very small, so run it 20 times and take the average
    runs = 20
    t_start = time.perf_counter()
    for i in range(runs):
        dijkstra(grid, start, goal)
    t_end = time.perf_counter()
    avg_time_ms = (t_end - t_start) / runs * 1000

    # results
    path_length_m = cost * resolution            # cost is in cell units
    print("\nDijkstra results:")
    print("  cells in the path:", len(path))
    print("  path cost (cell units):", round(cost, 3))
    print("  path length:", round(path_length_m, 3), "m")
    print("  nodes expanded:", len(explored))
    print("  share of free cells explored:", round(100 * len(explored) / n_free, 1), "%")
    print("  average run time:", round(avg_time_ms, 2), "ms (over", runs, "runs)")

    # straight line distance for comparison
    sx, sy = cell_to_world(start[0], start[1])
    gx, gy = cell_to_world(goal[0], goal[1])
    straight = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
    print("  straight line distance:", round(straight, 3), "m")

    # save the path for the later steps
    np.save(os.path.join(data_folder, "path_dijkstra.npy"), np.array(path))

    # ---- figure ----
    explored_map = np.full((n_cells, n_cells), np.nan)
    for (r, c) in explored:
        explored_map[r, c] = 1

    path_x = []
    path_y = []
    for (r, c) in path:
        x, y = cell_to_world(r, c)
        path_x.append(x)
        path_y.append(y)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [-1, 1, -1, 1]
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax.imshow(explored_map, origin="lower", extent=extent,
              cmap=ListedColormap(["skyblue"]), vmin=0, vmax=1, alpha=0.6)
    ax.plot(path_x, path_y, "r-", linewidth=2.5)
    ax.plot(path_x, path_y, "r.", markersize=4)
    ax.plot(sx, sy, "gs", markersize=11, markeredgecolor="black")
    ax.plot(gx, gy, marker="*", color="gold", markersize=18,
            markeredgecolor="black", linestyle="none")

    legend_items = [Patch(facecolor="black", label="Blocked (inflated)"),
                    Patch(facecolor="skyblue", label="Cells explored (" + str(len(explored)) + ")"),
                    plt.Line2D([0], [0], color="red", linewidth=2.5,
                               label="Path (" + str(round(path_length_m, 2)) + " m)"),
                    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="green",
                               markeredgecolor="black", markersize=10, label="Start"),
                    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                               markeredgecolor="black", markersize=14, label="Goal")]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=9)
    ax.set_title("Dijkstra's algorithm: explored cells and shortest path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.savefig(os.path.join(figures_folder, "A2_dijkstra_path.png"), dpi=150,
                bbox_inches="tight")
    plt.show()

    print("\nDone. Figure saved in:", figures_folder)