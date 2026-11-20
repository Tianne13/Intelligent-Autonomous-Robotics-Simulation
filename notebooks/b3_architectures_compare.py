import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folders setup
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


def simulate_architecture_metrics():
    """Simulate performance metrics across varying swarm sizes for Centralised vs Distributed."""
    agent_counts = np.array([3, 5, 10, 20, 50])
    
    # Centralised: Latency scales O(N) due to bottleneck at master server; single point of failure risk is 100%
    centralised_latency = 5.0 + 2.5 * agent_counts  # ms
    centralised_bandwidth = 10.0 + 5.0 * agent_counts # KB/s
    centralised_fault_tolerance = np.zeros_like(agent_counts) # 0% fault tolerant (master fails -> all fail)

    # Distributed: Latency remains constant O(1) locally; bandwidth is peer-to-peer; 100% robust to single agent loss
    distributed_latency = np.full_like(agent_counts, 8.5, dtype=float)  # constant local processing
    distributed_bandwidth = 4.0 + 1.2 * agent_counts # localized neighbour communication
    distributed_fault_tolerance = np.full_like(agent_counts, 95.0, dtype=float) # high robustness

    return agent_counts, centralised_latency, distributed_latency, centralised_bandwidth, distributed_bandwidth


def main():
    print("Executing Step 22: Comparing Centralised vs Distributed Decision-Making Architectures...")

    agents, cent_lat, dist_lat, cent_bw, dist_bw = simulate_architecture_metrics()

    # Create summary DataFrame for reporting
    comp_df = pd.DataFrame({
        "Swarm Size (N)": agents,
        "Centralised Latency (ms)": cent_lat,
        "Distributed Latency (ms)": dist_lat,
        "Centralised Bandwidth (KB/s)": cent_bw,
        "Distributed Bandwidth (KB/s)": dist_bw
    })

    print("\n--- Architecture Comparison Table ---")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(data_folder, "b3_architecture_comparison.csv"), index=False)

    # Plotting comparison charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Processing Latency vs Swarm Size
    ax1.plot(agents, cent_lat, marker='o', color='tab:red', linewidth=2, label="Centralised Architecture")
    ax1.plot(agents, dist_lat, marker='s', color='tab:blue', linewidth=2, label="Distributed Architecture")
    ax1.set_title("Decision-Making Latency vs Swarm Size")
    ax1.set_xlabel("Number of Agents (N)")
    ax1.set_ylabel("Processing Latency (ms)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Plot 2: Communication Bandwidth vs Swarm Size
    ax2.plot(agents, cent_bw, marker='o', color='tab:red', linewidth=2, label="Centralised Architecture")
    ax2.plot(agents, dist_bw, marker='s', color='tab:blue', linewidth=2, label="Distributed Architecture")
    ax2.set_title("Communication Bandwidth vs Swarm Size")
    ax2.set_xlabel("Number of Agents (N)")
    ax2.set_ylabel("Bandwidth Consumption (KB/s)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "B3_architecture_comparison.png"), dpi=150)
    plt.show()
    print("Architecture comparison plot saved to figures/B3_architecture_comparison.png")


if __name__ == "__main__":
    main()