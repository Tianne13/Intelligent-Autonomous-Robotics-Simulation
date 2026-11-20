import os
import math
import numpy as np
import pandas as pd

from grid_map import cell_to_world
from pid_controller import make_dense_path, follow_path

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")


def path_of(name):
    return os.path.join(data_folder, name)


def stats_from_log(x, y, cross_track, goal_x, goal_y):
    # distance travelled, final distance and cross track numbers (in cm)
    travelled = np.sum(np.hypot(np.diff(x), np.diff(y)))
    final_distance = math.hypot(x[-1] - goal_x, y[-1] - goal_y)
    cross = np.array(cross_track)
    return {"travelled_m": travelled, "final_distance_m": final_distance,
            "mean_cm": np.mean(np.abs(cross)) * 100,
            "max_cm": np.max(np.abs(cross)) * 100,
            "rms_cm": math.sqrt(np.mean(cross ** 2)) * 100}


# ------------------------------------------------------------
# 1. pid gains
# ------------------------------------------------------------
print("=== 1. PID gains (data/pid_gains.npy) ===")
if os.path.exists(path_of("pid_gains.npy")):
    gains = np.load(path_of("pid_gains.npy"))
    print("Kp =", gains[0], " Ki =", gains[1], " Kd =", gains[2])
else:
    gains = None
    print("pid_gains.npy not found (run a3_pid_tuning.py)")

# ------------------------------------------------------------
# 2. the webots run
# ------------------------------------------------------------
print("\n=== 2. Webots run (data/webots_run_log.npz) ===")
webots = None
if os.path.exists(path_of("webots_run_log.npz")) and os.path.exists(path_of("webots_planned_path.npy")):
    log = np.load(path_of("webots_run_log.npz"))
    planned = np.load(path_of("webots_planned_path.npy"))
    planned_length = np.sum(np.hypot(np.diff(planned[:, 0]), np.diff(planned[:, 1])))
    webots = stats_from_log(log["x"], log["y"], log["cross_track"], planned[-1][0], planned[-1][1])
    webots["time_s"] = float(log["t"][-1])

    lidar = log["lidar_min"]
    finite = lidar[np.isfinite(lidar)]
    heading_error = np.degrees(np.abs(log["heading_error"]))
    after_2s = heading_error[log["t"] > 2.0]

    print("Goal reached:", bool(log["reached"]))
    print("Gains used in the run:", log["gains"])
    print("Planned path length:", round(planned_length, 3), "m")
    print("Time to goal:", round(webots["time_s"], 1), "s")
    print("Distance travelled:", round(webots["travelled_m"], 3), "m")
    print("Final distance to the goal:", round(webots["final_distance_m"], 3), "m")
    print("Cross track error (cm): mean", round(webots["mean_cm"], 2), " max",
          round(webots["max_cm"], 2), " rms", round(webots["rms_cm"], 2))
    print("Largest heading error after 2 s:", round(float(np.max(after_2s)), 1), "deg")
    print("Average commanded speed:", round(float(np.mean(log["v"])), 3), "m/s")
    if len(finite) > 0:
        print("Smallest lidar distance:", round(float(np.min(finite)), 3), "m")
        print("Any lidar reading under 0.055 m (safety stop level):", bool(np.any(finite < 0.055)))
    else:
        print("Smallest lidar distance: no finite readings in the log")
else:
    print("webots_run_log.npz or webots_planned_path.npy not found (run the navigator in Webots)")

# ------------------------------------------------------------
# 3. the same path in the python simulation (same gains)
# ------------------------------------------------------------
print("\n=== 3. Python simulation with the same path and gains ===")
simulation = None
if os.path.exists(path_of("path_astar.npy")) and os.path.exists(path_of("goal_estimate.npy")) and gains is not None:
    path_cells = np.load(path_of("path_astar.npy"))
    goal_estimate = np.load(path_of("goal_estimate.npy"))
    start_pose = (-0.8, -0.8, 0.0)

    waypoints = [(start_pose[0], start_pose[1])]
    for i in range(1, len(path_cells) - 1):
        r, c = path_cells[i]
        waypoints.append(cell_to_world(r, c))
    waypoints.append((float(goal_estimate[0]), float(goal_estimate[1])))
    dense_path = make_dense_path(waypoints, 0.01)

    sim_log = follow_path(dense_path, (float(gains[0]), float(gains[1]), float(gains[2])), start_pose)
    simulation = stats_from_log(np.array(sim_log["x"]), np.array(sim_log["y"]),
                                sim_log["cross_track"], dense_path[-1][0], dense_path[-1][1])
    simulation["time_s"] = sim_log["time"]
    planned_length_sim = np.sum(np.hypot(np.diff(dense_path[:, 0]), np.diff(dense_path[:, 1])))
    print("Goal reached:", sim_log["reached"])
    print("Planned path length:", round(planned_length_sim, 3), "m")
    print("Time to goal:", round(simulation["time_s"], 1), "s")
    print("Distance travelled:", round(simulation["travelled_m"], 3), "m")
    print("Final distance to the goal:", round(simulation["final_distance_m"], 3), "m")
    print("Cross track error (cm): mean", round(simulation["mean_cm"], 2), " max",
          round(simulation["max_cm"], 2), " rms", round(simulation["rms_cm"], 2))
else:
    print("needs path_astar.npy, goal_estimate.npy and pid_gains.npy")

# ------------------------------------------------------------
# 4. webots against simulation
# ------------------------------------------------------------
print("\n=== 4. Webots against simulation ===")
if webots is not None and simulation is not None:
    table = pd.DataFrame({"metric": ["time to goal (s)", "distance travelled (m)",
                                     "mean cross track (cm)", "max cross track (cm)",
                                     "rms cross track (cm)"],
                          "simulation": [simulation["time_s"], simulation["travelled_m"],
                                         simulation["mean_cm"], simulation["max_cm"],
                                         simulation["rms_cm"]],
                          "webots": [webots["time_s"], webots["travelled_m"],
                                     webots["mean_cm"], webots["max_cm"],
                                     webots["rms_cm"]]})
    table["difference"] = table["webots"] - table["simulation"]
    print(table.round(3).to_string(index=False))
else:
    print("needs both the webots run and the simulation run")

# ------------------------------------------------------------
# 5. tables saved by the earlier steps
# ------------------------------------------------------------
print("\n=== 5. Gain sets on the path (data/a3_pid_tracking_compare.csv) ===")
if os.path.exists(path_of("a3_pid_tracking_compare.csv")):
    compare = pd.read_csv(path_of("a3_pid_tracking_compare.csv"))
    print(compare.round(3).to_string(index=False))
else:
    print("not found (run a3_pid_tuning.py)")

print("\n=== 6. Heading step metrics (data/a3_pid_metrics.csv) ===")
if os.path.exists(path_of("a3_pid_metrics.csv")):
    metrics = pd.read_csv(path_of("a3_pid_metrics.csv"))
    print(metrics.round(3).to_string(index=False))
else:
    print("not found (run a3_pid_tuning.py)")

print("\n=== 7. Planner comparison, main case (data/a2_comparison.csv) ===")
if os.path.exists(path_of("a2_comparison.csv")):
    planners = pd.read_csv(path_of("a2_comparison.csv"))
    print(planners[planners["case"] == "1 main"].round(3).to_string(index=False))
else:
    print("not found (run a2_compare.py)")

print("\nDone. Copy everything above into your notes.")