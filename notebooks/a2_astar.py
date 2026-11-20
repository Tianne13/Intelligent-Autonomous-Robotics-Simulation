import os
import math
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from grid_map import n_cells, resolution, cell_to_world, get_neighbours, load_planning_data
from a2_dijkstra import dijkstra        # used to check the path length


def heuristic(cell, goal, kind):
    # estimate of the cost from a cell to the goal (in cell units)
    dr = abs(cell[0] - goal[0])
    dc = abs(cell[1] - goal[1])
    if kind == "manhattan":
        return dr + dc
    return math.sqrt(dr * dr + dc * dc)         # euclidean


def a_star(grid, start, goal, kind="euclidean"):
    # same as dijkstra but the queue is ordered by f = g + h
    # gives back the path, the cost and the cells expanded in order

    queue = []                                  # entries are (f, g, cell)
    heapq.heappush(queue, (heuristic(start, goal, kind), 0.0, start))
    g_cost = {start: 0.0}                       # real cost from the start
    came_from = {start: None}
    closed = set()
    explored = []

    while len(queue) > 0:
        f, g, current = heapq.heappop(queue)

        # skip old entries of cells that are already finished
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)

        if current == goal:
            break

        for (neighbour, step_cost) in get_neighbours(current, grid):
            new_g = g + step_cost
            if neighbour not in g_cost or new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g
                came_from[neighbour] = current
                new_f = new_g + heuristic(neighbour, goal, kind)
                heapq.heappush(queue, (new_f, new_g, neighbour))

    if goal not in closed:
        return None, None, explored

    # rebuild the path backwards from the goal
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    path.reverse()
    return path, g_cost[goal], explored


def draw_result(ax, grid, explored, path, start, goal, title):
    # draws the grid, the explored cells and the path on one axis
    explored_map = np.full((n_cells, n_cells), np.nan)
    for (r, c) in explored:
        explored_map[r, c] = 1

    path_x = []
    path_y = []
    for (r, c) in path:
        x, y = cell_to_world(r, c)
        path_x.append(x)
        path_y.append(y)

    sx, sy = cell_to_world(start[0], start[1])
    gx, gy = cell_to_world(goal[0], goal[1])

    extent = [-1, 1, -1, 1]
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax.imshow(explored_map, origin="lower", extent=extent,
              cmap=ListedColormap(["skyblue"]), vmin=0, vmax=1, alpha=0.6)
    ax.plot(path_x, path_y, "r-", linewidth=2.5)
    ax.plot(path_x, path_y, "r.", markersize=4)
    ax.plot(sx, sy, "gs", markersize=11, markeredgecolor="black")
    ax.plot(gx, gy, marker="*", color="gold", markersize=18,
            markeredgecolor="black", linestyle="none")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# only runs when this file is run directly
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

    # dijkstra gives the optimal cost to compare with
    d_path, d_cost, d_explored = dijkstra(grid, start, goal)
    print("Dijkstra cost (optimal):", round(d_cost, 3), "cell units =",
          round(d_cost * resolution, 3), "m")

    results = {}
    runs = 20
    for kind in ["euclidean", "manhattan"]:
        path, cost, explored = a_star(grid, start, goal, kind)
        if path is None:
            print("No path found with", kind)
            raise SystemExit

        # average time over 20 runs
        t_start = time.perf_counter()
        for i in range(runs):
            a_star(grid, start, goal, kind)
        t_end = time.perf_counter()
        avg_time_ms = (t_end - t_start) / runs * 1000

        results[kind] = (path, cost, explored)

        print("\nA* with", kind, "heuristic:")
        print("  cells in the path:", len(path))
        print("  path length:", round(cost * resolution, 3), "m")
        print("  longer than dijkstra by:", round((cost - d_cost) * resolution, 4), "m")
        print("  nodes expanded:", len(explored))
        print("  share of free cells explored:", round(100 * len(explored) / n_free, 1), "%")
        print("  average run time:", round(avg_time_ms, 2), "ms (over", runs, "runs)")

    # save the euclidean path for the later steps
    np.save(os.path.join(data_folder, "path_astar.npy"), np.array(results["euclidean"][0]))

    # ---- figure: euclidean on the left, manhattan on the right ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5))

    p, c, e = results["euclidean"]
    draw_result(ax1, grid, e, p, start, goal,
                "A* (Euclidean): " + str(len(e)) + " cells explored, "
                + str(round(c * resolution, 2)) + " m")
    p, c, e = results["manhattan"]
    draw_result(ax2, grid, e, p, start, goal,
                "A* (Manhattan): " + str(len(e)) + " cells explored, "
                + str(round(c * resolution, 2)) + " m")

    legend_items = [Patch(facecolor="black", label="Blocked (inflated)"),
                    Patch(facecolor="skyblue", label="Cells explored"),
                    plt.Line2D([0], [0], color="red", linewidth=2.5, label="Path"),
                    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="green",
                               markeredgecolor="black", markersize=10, label="Start"),
                    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                               markeredgecolor="black", markersize=14, label="Goal")]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(os.path.join(figures_folder, "A2_astar_path.png"), dpi=150,
                bbox_inches="tight")
    plt.show()

    print("\nDone. Figure saved in:", figures_folder)