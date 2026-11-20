import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grid_map import n_cells, resolution, world_to_cell, cell_to_world, load_planning_data
from a2_dijkstra import dijkstra
from a2_astar import a_star

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# load the planning grid and the main start and goal
grid, start, goal = load_planning_data(data_folder)


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


def run_algorithm(name, grid, s, t):
    # runs one of the three planners
    if name == "Dijkstra":
        return dijkstra(grid, s, t)
    elif name == "A* Euclidean":
        return a_star(grid, s, t, "euclidean")
    else:
        return a_star(grid, s, t, "manhattan")


# ------------------------------------------------------------
# test cases (the first one is the real task)
# ------------------------------------------------------------
case_list = [("1 main", (-0.8, -0.8), (0.8, 0.8)),
             ("2 bottom", (-0.8, -0.8), (0.8, -0.8)),
             ("3 corners", (-0.8, 0.8), (0.8, -0.8)),
             ("4 middle", (-0.8, 0.0), (0.8, 0.0)),
             ("5 vertical", (0.0, 0.8), (0.0, -0.8))]

cases = []
for (name, s_world, g_world) in case_list:
    if name == "1 main":
        s_cell = start
        g_cell = goal
    else:
        s_cell = nearest_free(world_to_cell(s_world[0], s_world[1]), grid)
        g_cell = nearest_free(world_to_cell(g_world[0], g_world[1]), grid)
    cases.append((name, s_cell, g_cell))
    print("Case", name, ": start", s_cell, "goal", g_cell)

algorithm_names = ["Dijkstra", "A* Euclidean", "A* Manhattan"]
runs = 10                       # runs used for the average time

# ------------------------------------------------------------
# run everything
# ------------------------------------------------------------
rows = []
main_paths = {}
optimal_cost = {}               # dijkstra cost for each case (the optimal one)

for (case_name, s_cell, g_cell) in cases:
    for name in algorithm_names:
        path, cost, explored = run_algorithm(name, grid, s_cell, g_cell)
        if path is None:
            print("No path found for", name, "in case", case_name)
            continue

        # average time over several runs
        t_start = time.perf_counter()
        for i in range(runs):
            run_algorithm(name, grid, s_cell, g_cell)
        t_end = time.perf_counter()
        avg_ms = (t_end - t_start) / runs * 1000

        if name == "Dijkstra":
            optimal_cost[case_name] = cost

        rows.append({"case": case_name, "algorithm": name,
                     "path_length_m": cost * resolution,
                     "nodes_expanded": len(explored),
                     "time_ms": avg_ms})
        if case_name == "1 main":
            main_paths[name] = path

table = pd.DataFrame(rows)

# how much longer is each path than the optimal one
extra = []
for index, row in table.iterrows():
    best = optimal_cost[row["case"]] * resolution
    extra.append(row["path_length_m"] - best)
table["extra_vs_optimal_m"] = extra

table.to_csv(os.path.join(data_folder, "a2_comparison.csv"), index=False)

print("\nResults for every test case:")
print(table.round(4).to_string(index=False))

# average over the 5 cases
summary = table.groupby("algorithm")[["path_length_m", "nodes_expanded", "time_ms"]].mean()
summary = summary.loc[algorithm_names]
print("\nAverage over the", len(cases), "test cases:")
print(summary.round(3).to_string())

print("\nCases where the path was longer than the optimal one:")
for name in algorithm_names:
    sub = table[table["algorithm"] == name]
    count = int((sub["extra_vs_optimal_m"] > 0.0001).sum())
    print(" ", name, ":", count, "of", len(sub))

# ------------------------------------------------------------
# figure
# ------------------------------------------------------------
colours = {"Dijkstra": "tab:blue", "A* Euclidean": "tab:orange", "A* Manhattan": "tab:green"}
case_names = [c[0] for c in cases]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
ax1, ax2, ax3, ax4 = axes.flatten()

# panel 1: the three paths for the main task on the map
ax1.imshow(grid, origin="lower", extent=[-1, 1, -1, 1], cmap="gray_r", vmin=0, vmax=1)
widths = {"Dijkstra": 6, "A* Euclidean": 3.5, "A* Manhattan": 1.8}
for name in algorithm_names:
    xs = []
    ys = []
    for (r, c) in main_paths[name]:
        x, y = cell_to_world(r, c)
        xs.append(x)
        ys.append(y)
    ax1.plot(xs, ys, color=colours[name], linewidth=widths[name], alpha=0.85, label=name)
sx, sy = cell_to_world(start[0], start[1])
gx, gy = cell_to_world(goal[0], goal[1])
ax1.plot(sx, sy, "gs", markersize=11, markeredgecolor="black", label="Start")
ax1.plot(gx, gy, marker="*", color="gold", markersize=18,
         markeredgecolor="black", linestyle="none", label="Goal")
ax1.legend(loc="lower right", fontsize=8)
ax1.set_title("Paths found for the main task")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")


def bar_chart(ax, column, title, ylabel, log=False):
    # grouped bars: one group per test case, one bar per algorithm
    x = np.arange(len(case_names))
    width = 0.25
    for i in range(len(algorithm_names)):
        name = algorithm_names[i]
        values = []
        for case_name in case_names:
            sub = table[(table["algorithm"] == name) & (table["case"] == case_name)]
            values.append(float(sub[column].iloc[0]))
        ax.bar(x + (i - 1) * width, values, width, label=name, color=colours[name])
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Test case")
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y")
    ax.legend(fontsize=8)


bar_chart(ax2, "nodes_expanded", "Nodes expanded", "Cells expanded")
bar_chart(ax3, "time_ms", "Average run time (log scale)", "Time (ms)", log=True)
bar_chart(ax4, "path_length_m", "Path length", "Length (m)")

plt.suptitle("Dijkstra vs A* (Euclidean and Manhattan) on the planning grid", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A2_comparison.png"), dpi=150)
plt.show()

print("\nDone. Figure saved in:", figures_folder)
print("Table saved in:", data_folder)