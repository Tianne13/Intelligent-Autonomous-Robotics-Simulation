import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grid_map import cell_to_world
from robot_model import (dt, motor_tau, wheel_radius, axle_length, max_wheel_speed,
                         vw_to_wheels, wheels_to_vw, limit_wheels, motor_step)
from pid_controller import PID, make_dense_path, follow_path

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

max_w = 2 * wheel_radius * max_wheel_speed / axle_length    # about 4.95 rad/s when spinning
step_deg = 20.0                 # size of the heading step
disturbance = -0.05             # rad/s, small constant drift (wheel speed mismatch of about 3%)


# ------------------------------------------------------------
# heading step response of the robot turning on the spot
# ------------------------------------------------------------
def heading_step(gains, dist=0.0, t_end=4.0):
    # the robot starts facing 0 deg and has to turn to step_deg
    # dist is a constant turn rate drift added to the robot (a disturbance)
    kp, ki, kd = gains
    pid = PID(kp, ki, kd, out_limit=max_w, int_limit=1.0)
    target = math.radians(step_deg)

    theta = 0.0
    wl = 0.0
    wr = 0.0
    n = int(round(t_end / dt))
    t_list = []
    theta_list = []
    for i in range(n + 1):
        t_list.append(i * dt)
        theta_list.append(theta)

        error = target - theta
        w_cmd = pid.update(error)

        # same motor model as before: wheel limit, then the lag
        wl_cmd, wr_cmd = vw_to_wheels(0.0, w_cmd)
        wl_cmd, wr_cmd = limit_wheels(wl_cmd, wr_cmd)
        wl = motor_step(wl, wl_cmd)
        wr = motor_step(wr, wr_cmd)
        v, w = wheels_to_vw(wl, wr)
        theta = theta + (w + dist) * dt
    return np.array(t_list), np.array(theta_list), target


def step_metrics(t, theta, target):
    # rise time (10% to 90%), overshoot (%), settling time (2% band), steady state error (%)
    t10 = None
    t90 = None
    for i in range(len(t)):
        if t10 is None and theta[i] >= 0.1 * target:
            t10 = t[i]
        if t90 is None and theta[i] >= 0.9 * target:
            t90 = t[i]
    if t10 is None or t90 is None:
        rise = np.nan
    else:
        rise = t90 - t10

    overshoot = max(0.0, (np.max(theta) - target) / target * 100)

    # settling time = the time after which the angle stays inside the 2% band
    band = 0.02 * target
    last_outside = -1
    for i in range(len(t)):
        if abs(theta[i] - target) > band:
            last_outside = i
    if last_outside == len(t) - 1:
        settling = np.nan               # never settled
    elif last_outside == -1:
        settling = 0.0
    else:
        settling = t[last_outside + 1]

    # steady state error = average error over the last second
    k = int(round(1.0 / dt))
    sse = np.mean(target - theta[-k:]) / target * 100
    return rise, overshoot, settling, sse


def theory_overshoot(kp, kd):
    # overshoot of the second order system tau*s^2 + (1 + kd)*s + kp = 0 (no integral)
    zeta = (1 + kd) / (2 * math.sqrt(kp * motor_tau))
    if zeta >= 1:
        return 0.0
    return 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))


rows = []
curves = {"E1": [], "E2": [], "E3": [], "E4": []}


def run_experiment(name, gains, dist, t_end):
    t, theta, target = heading_step(gains, dist, t_end)
    rise, overshoot, settling, sse = step_metrics(t, theta, target)
    kp, ki, kd = gains
    theory = theory_overshoot(kp, kd) if name in ("E1", "E2") else np.nan
    rows.append({"experiment": name, "Kp": kp, "Ki": ki, "Kd": kd,
                 "rise_time_s": rise, "overshoot_pct": overshoot,
                 "settling_time_s": settling, "steady_state_error_pct": sse,
                 "theory_overshoot_pct": theory})
    if name in curves:
        curves[name].append((gains, t, theta))


# E1: only Kp
for kp in [1, 2, 5, 10, 20]:
    run_experiment("E1", (kp, 0.0, 0.0), 0.0, 4.0)
# E2: Kp = 15 and different Kd
for kd in [0.0, 0.2, 0.5, 1.0]:
    run_experiment("E2", (15, 0.0, kd), 0.0, 4.0)
# E3: constant drift and different Ki
for ki in [0.0, 0.5, 2.0, 5.0]:
    run_experiment("E3", (5, ki, 0.3), disturbance, 6.0)
# E4: Ki too high
for ki in [30, 60, 100]:
    run_experiment("E4", (5, ki, 0.3), disturbance, 6.0)

table = pd.DataFrame(rows)
print("Heading step response (step of", step_deg, "deg, turning on the spot)")
print("E1: Kp only | E2: Kp = 15, Kd changes | E3: drift of", disturbance,
      "rad/s, Ki changes | E4: Ki too high")
print(table.round(3).to_string(index=False))

# stability limit of the integral gain for the E3/E4 settings (Routh)
ki_limit = (1 + 0.3) * 5 / motor_tau
print("\nTheory: with Kp = 5 and Kd = 0.3 the loop is stable only for Ki <", round(ki_limit, 1))

# ------------------------------------------------------------
# figure 1: step responses
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()
titles = {"E1": "E1: effect of Kp (Ki = 0, Kd = 0)",
          "E2": "E2: effect of Kd (Kp = 15, Ki = 0)",
          "E3": "E3: effect of Ki with a constant drift (Kp = 5, Kd = 0.3)",
          "E4": "E4: Ki too high (Kp = 5, Kd = 0.3)"}
for ax, name in zip([ax1, ax2, ax3, ax4], ["E1", "E2", "E3", "E4"]):
    for (gains, t, theta) in curves[name]:
        if name == "E1":
            label = "Kp = " + str(gains[0])
        elif name == "E2":
            label = "Kd = " + str(gains[2])
        else:
            label = "Ki = " + str(gains[1])
        ax.plot(t, np.degrees(theta), label=label)
    ax.axhline(step_deg, color="black", linestyle="--", linewidth=1, label="Target")
    ax.axhspan(step_deg * 0.98, step_deg * 1.02, color="gray", alpha=0.2)
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_title(titles[name])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A3_pid_step_responses.png"), dpi=150)
plt.show()

# ------------------------------------------------------------
# part 2: path tracking with different gains
# ------------------------------------------------------------
path_cells = np.load(os.path.join(data_folder, "path_astar.npy"))
goal_estimate = np.load(os.path.join(data_folder, "goal_estimate.npy"))
start_pose = (-0.8, -0.8, 0.0)

waypoints = [(start_pose[0], start_pose[1])]
for i in range(1, len(path_cells) - 1):
    r, c = path_cells[i]
    waypoints.append(cell_to_world(r, c))
waypoints.append((float(goal_estimate[0]), float(goal_estimate[1])))
dense_path = make_dense_path(waypoints, 0.01)


def track_summary(gains):
    # runs the path following and measures how good it was
    log = follow_path(dense_path, gains, start_pose)
    cross = np.array(log["cross_track"])
    w = np.array(log["w"])
    summary = {"reached": log["reached"], "time_s": log["time"],
               "mean_cm": np.mean(np.abs(cross)) * 100,
               "max_cm": np.max(np.abs(cross)) * 100,
               "rms_cm": math.sqrt(np.mean(cross ** 2)) * 100,
               "turn_rate_jitter": np.mean(np.abs(np.diff(w))),
               "max_turn_rate": np.max(np.abs(w))}
    return summary, log


# search a small grid of gains (all three terms stay on) and keep the best one
best_gains = None
best_rms = 1e9
search_rows = []
for kp in [1.5, 2.0, 3.0, 4.0]:
    for ki in [0.05, 0.1, 0.3]:
        for kd in [0.05, 0.1, 0.2]:
            summary, log = track_summary((kp, ki, kd))
            if not summary["reached"]:
                continue
            search_rows.append({"Kp": kp, "Ki": ki, "Kd": kd, "rms_cm": summary["rms_cm"],
                                "max_cm": summary["max_cm"], "time_s": summary["time_s"]})
            if summary["rms_cm"] < best_rms:
                best_rms = summary["rms_cm"]
                best_gains = (kp, ki, kd)

search = pd.DataFrame(search_rows).sort_values("rms_cm")
print("\nGrid search on the path tracking task (" + str(len(search)) + " gain sets), best 5:")
print(search.head(5).round(3).to_string(index=False))
print("Chosen gains: Kp =", best_gains[0], " Ki =", best_gains[1], " Kd =", best_gains[2])
np.save(os.path.join(data_folder, "pid_gains.npy"), np.array(best_gains))

# compare four gain sets
gain_sets = [("Low gain", (1.0, 0.0, 0.0)),
             ("First guess", (2.0, 0.1, 0.2)),
             ("Tuned", best_gains),
             ("Aggressive", (12.0, 3.0, 0.5))]
compare_rows = []
logs = {}
for name, gains in gain_sets:
    summary, log = track_summary(gains)
    row = {"set": name, "Kp": gains[0], "Ki": gains[1], "Kd": gains[2]}
    row.update(summary)
    compare_rows.append(row)
    logs[name] = log

compare = pd.DataFrame(compare_rows)
print("\nPath tracking with four gain sets:")
print(compare.round(3).to_string(index=False))

# the same four sets in the heading step test with the drift
print("\nHeading step test of the same four sets (drift", disturbance, "rad/s):")
for name, gains in gain_sets:
    t, theta, target = heading_step(gains, disturbance, 6.0)
    rise, overshoot, settling, sse = step_metrics(t, theta, target)
    print(" ", name, gains, "rise", round(rise, 3), "s, overshoot", round(overshoot, 1),
          "%, settling", round(settling, 3) if not np.isnan(settling) else "not settled",
          ", steady state error", round(sse, 2), "%")
    rows.append({"experiment": "gain set: " + name, "Kp": gains[0], "Ki": gains[1],
                 "Kd": gains[2], "rise_time_s": rise, "overshoot_pct": overshoot,
                 "settling_time_s": settling, "steady_state_error_pct": sse,
                 "theory_overshoot_pct": np.nan})

pd.DataFrame(rows).to_csv(os.path.join(data_folder, "a3_pid_metrics.csv"), index=False)
compare.to_csv(os.path.join(data_folder, "a3_pid_tracking_compare.csv"), index=False)

# ------------------------------------------------------------
# figure 2: tracking with the four gain sets
# ------------------------------------------------------------
colours = {"Low gain": "tab:blue", "First guess": "tab:orange",
           "Tuned": "tab:green", "Aggressive": "tab:red"}
fig, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(17, 5.5))
for name, gains in gain_sets:
    log = logs[name]
    label = name + " (" + str(gains[0]) + ", " + str(gains[1]) + ", " + str(gains[2]) + ")"
    bx1.plot(log["t"], np.array(log["cross_track"]) * 100, color=colours[name], label=label)
    bx2.plot(log["t"], np.degrees(log["heading_error"]), color=colours[name], label=label)
    bx3.plot(log["t"], log["w"], color=colours[name], label=label)
bx1.set_title("Cross track error")
bx1.set_ylabel("Error (cm)")
bx2.set_title("Heading error")
bx2.set_ylabel("Error (deg)")
bx3.set_title("Turn rate of the robot")
bx3.set_ylabel("w (rad/s)")
for bx in [bx1, bx2, bx3]:
    bx.axhline(0, color="black", linewidth=0.8)
    bx.grid(True)
    bx.set_xlabel("Time (s)")
bx1.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A3_pid_tracking_compare.png"), dpi=150)
plt.show()

print("\nDone. Figures saved in:", figures_folder)
print("Gains saved in:", data_folder)