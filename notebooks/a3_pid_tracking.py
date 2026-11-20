import os
import math
import numpy as np
import matplotlib.pyplot as plt

from grid_map import cell_to_world, resolution, n_cells
from robot_model import axle_length, max_v, dt
from pid_controller import make_dense_path, follow_path

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# ------------------------------------------------------------
# 1. load the planned path and turn it into waypoints in metres
# ------------------------------------------------------------
planning_grid = np.load(os.path.join(data_folder, "planning_grid.npy"))
fused_grid = np.load(os.path.join(data_folder, "fused_grid.npy"))
path_cells = np.load(os.path.join(data_folder, "path_astar.npy"))
goal_estimate = np.load(os.path.join(data_folder, "goal_estimate.npy"))

start_pose = (-0.8, -0.8, 0.0)          # same start as in webots (facing +x)
goal_point = (float(goal_estimate[0]), float(goal_estimate[1]))

# first point = real start, last point = goal from the camera,
# the cells in between are the centres of the path cells
waypoints = [(start_pose[0], start_pose[1])]
for i in range(1, len(path_cells) - 1):
    r, c = path_cells[i]
    waypoints.append(cell_to_world(r, c))
waypoints.append(goal_point)

dense_path = make_dense_path(waypoints, 0.01)
planned_length = np.sum(np.hypot(np.diff(dense_path[:, 0]), np.diff(dense_path[:, 1])))
print("Waypoints:", len(waypoints), " Planned path length:", round(planned_length, 3), "m")

# ------------------------------------------------------------
# 2. controller settings
# ------------------------------------------------------------
# first set of gains (they get tuned in the next step)
kp = 2.0
ki = 0.1
kd = 0.2
v_nom = 0.08            # forward speed (m/s)
lookahead = 0.12        # how far ahead on the path the robot aims (m)
w_limit = 2.0           # largest turn rate the controller may ask for (rad/s)

print("PID gains: Kp =", kp, " Ki =", ki, " Kd =", kd)
print("Forward speed:", v_nom, "m/s, lookahead:", lookahead, "m")
print("Initial heading error: 45 deg (robot faces +x, the path starts diagonally)")

log = follow_path(dense_path, (kp, ki, kd), start_pose, v_nom=v_nom,
                  lookahead=lookahead, goal_tol=0.03, w_limit=w_limit)

t = np.array(log["t"])
x = np.array(log["x"])
y = np.array(log["y"])
heading_error = np.array(log["heading_error"])
cross_track = np.array(log["cross_track"])
v_actual = np.array(log["v"])
w_actual = np.array(log["w"])

# ------------------------------------------------------------
# 3. results
# ------------------------------------------------------------
travelled = np.sum(np.hypot(np.diff(x), np.diff(y)))
final_distance = math.hypot(x[-1] - goal_point[0], y[-1] - goal_point[1])
print("\nGoal reached:", log["reached"])
print("Time to goal:", round(log["time"], 1), "s")
print("Distance travelled:", round(travelled, 3), "m")
print("Final distance to the goal:", round(final_distance, 3), "m")
print("Cross track error: mean", round(np.mean(np.abs(cross_track)), 4), "m, max",
      round(np.max(np.abs(cross_track)), 4), "m, rms",
      round(math.sqrt(np.mean(cross_track ** 2)), 4), "m")
print("Largest heading error after the first 2 s:",
      round(math.degrees(np.max(np.abs(heading_error[t > 2.0]))), 1), "deg")

# how close the robot gets to the obstacles in the fused map
blocked_rows, blocked_cols = np.nonzero(fused_grid)
half = resolution / 2
min_clearance = 9.9
for i in range(0, len(x), 5):
    for j in range(len(blocked_rows)):
        bx, by = cell_to_world(blocked_rows[j], blocked_cols[j])
        dx = max(abs(x[i] - bx) - half, 0)
        dy = max(abs(y[i] - by) - half, 0)
        d = math.hypot(dx, dy)
        if d < min_clearance:
            min_clearance = d
print("Smallest distance from the robot centre to an obstacle cell:",
      round(min_clearance, 3), "m (robot radius is 0.037 m)")

# ------------------------------------------------------------
# 4. figure
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
ax1, ax2, ax3, ax4 = axes.flatten()

# planned path vs actual path on the map
ax1.imshow(planning_grid, origin="lower", extent=[-1, 1, -1, 1], cmap="gray_r",
           vmin=0, vmax=1, alpha=0.8)
ax1.plot(dense_path[:, 0], dense_path[:, 1], "b--", linewidth=2, label="Planned path (A*)")
ax1.plot(x, y, "r-", linewidth=1.5, label="Robot path (PID)")
ax1.plot(start_pose[0], start_pose[1], "gs", markersize=10, markeredgecolor="black",
         label="Start")
ax1.plot(goal_point[0], goal_point[1], marker="*", color="gold", markersize=16,
         markeredgecolor="black", linestyle="none", label="Goal")
ax1.legend(loc="lower right", fontsize=8)
ax1.set_title("Path tracking with the PID controller")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# heading error
ax2.plot(t, np.degrees(heading_error), "b-")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.grid(True)
ax2.set_title("Heading error over time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Heading error (deg)")

# cross track error
ax3.plot(t, cross_track * 100, "r-")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.grid(True)
ax3.set_title("Cross track error over time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Cross track error (cm)")

# speeds
ax4.plot(t, v_actual, "g-", label="Forward speed v (m/s)")
ax4.plot(t, w_actual, "m-", label="Turn rate w (rad/s)")
ax4.grid(True)
ax4.legend()
ax4.set_title("Robot speeds over time")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Speed")

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A3_pid_tracking.png"), dpi=150)
plt.show()

print("\nDone. Figure saved in:", figures_folder)