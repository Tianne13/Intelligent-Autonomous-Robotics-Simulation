import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)


def main():
    # Load Webots run log and planned path
    log_file = os.path.join(data_folder, "webots_run_log.npz")
    if not os.path.exists(log_file):
        print("Error: webots_run_log.npz not found. Run the Webots simulation first.")
        return

    log = np.load(log_file)
    t = log["t"]
    xs = log["xs"] if "xs" in log else log["x"]
    ys = log["ys"] if "ys" in log else log["y"]
    cross = log["cross_track"]
    reached = log["reached"]

    dense_path_file = os.path.join(data_folder, "webots_planned_path.npy")
    dense_path = np.load(dense_path_file)
    planned_x = dense_path[:, 0]
    planned_y = dense_path[:, 1]

    # Compute metrics
    planned_length = np.sum(np.hypot(np.diff(planned_x), np.diff(planned_y)))
    actual_length = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
    total_time = t[-1]
    mean_cross = np.mean(np.abs(cross)) * 100
    max_cross = np.max(np.abs(cross)) * 100
    rms_cross = math.sqrt(np.mean(cross ** 2)) * 100
    final_error = math.hypot(xs[-1] - planned_x[-1], ys[-1] - planned_y[-1]) * 100

    # Print performance evaluation table
    table_data = {
        "Metric": [
            "Path Length (m)",
            "Time to Goal (s)",
            "Mean Tracking Error (cm)",
            "Max Tracking Error (cm)",
            "RMS Tracking Error (cm)",
            "Final Goal Distance (cm)",
            "Goal Reached Successfully"
        ],
        "Planned": [round(planned_length, 3), "-", "-", "-", "-", "-", "-"],
        "Actual (Webots)": [
            round(actual_length, 3),
            round(total_time, 1),
            round(mean_cross, 2),
            round(max_cross, 2),
            round(rms_cross, 2),
            round(final_error, 2),
            str(reached)
        ]
    }
    perf_table = pd.DataFrame(table_data)
    print("\n--- Component A4: Closed-Loop Performance Evaluation Table ---")
    print(perf_table.to_string(index=False))

    perf_table.to_csv(os.path.join(data_folder, "a4_performance_table.csv"), index=False)

    # Plot 1: Planned vs Actual Trajectory & Plot 2: Cross-track Error Over Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Trajectory overlay
    ax1.plot(planned_x, planned_y, color="tab:blue", linewidth=2.5, linestyle="--", label="Planned A* Path")
    ax1.plot(xs, ys, color="tab:red", linewidth=1.5, label="Webots Actual Trajectory")
    ax1.scatter(planned_x[0], planned_y[0], color="green", s=100, zorder=5, label="Start")
    ax1.scatter(planned_x[-1], planned_y[-1], color="gold", edgecolors="black", s=120, zorder=5, label="Goal")
    ax1.set_title("Planned vs Actual Robot Trajectory")
    ax1.set_xlabel("World X (m)")
    ax1.set_ylabel("World Y (m)")
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=9)

    # Cross-track error over time
    ax2.plot(t, cross * 100, color="tab:purple", label="Cross-Track Error")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.fill_between(t, cross * 100, color="tab:purple", alpha=0.2)
    ax2.set_title("Cross-Track Error During Webots Run")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (cm)")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "A4_trajectory_comparison.png"), dpi=150)
    plt.show()

    print("\nEvaluation complete. Figure saved in:", figures_folder)


if __name__ == "__main__":
    main()