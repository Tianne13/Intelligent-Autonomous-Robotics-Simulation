import os
import numpy as np
import matplotlib.pyplot as plt

# Fixed random seed for reproducibility
np.random.seed(42)

# Folders setup
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)


def main():
    print("Executing Step 17: Multi-Agent Average Consensus Simulation...")

    # 1. Setup simulation parameters for 3 agents
    num_agents = 3
    num_steps = 40
    dt = 0.1

    # Initial random states (e.g., initial heading angles in radians) for each agent
    # Agent 1, Agent 2, Agent 3 start with different headings
    states = np.array([-1.2, 0.5, 2.1]) 
    
    # Define communication adjacency matrix (fully connected topology for 3 agents)
    # A[i, j] = 1 if agent i can communicate with agent j, else 0
    adj_matrix = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    # History tracking for plotting
    state_history = np.zeros((num_steps, num_agents))
    state_history[0] = states

    # 2. Iterative Average Consensus Protocol Update
    # Consensus update rule: x_i(t+1) = x_i(t) + epsilon * sum_j (x_j(t) - x_i(t))
    epsilon = 0.2  # Convergence rate parameter

    for t in range(1, num_steps):
        new_states = np.copy(states)
        for i in range(num_agents):
            sum_diff = 0.0
            neighbors = 0
            for j in range(num_agents):
                if adj_matrix[i, j] == 1:
                    sum_diff += (states[j] - states[i])
                    neighbors += 1
            if neighbors > 0:
                new_states[i] = states[i] + epsilon * sum_diff
        states = new_states
        state_history[t] = states

    # Print final consensus values
    print(f"Final consensus state achieved across agents: {states}")
    print(f"Target consensus value (exact average): {np.mean(state_history[0])}")

    # 3. Plotting Consensus Convergence
    plt.figure(figsize=(9, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(num_agents):
        plt.plot(range(num_steps), state_history[:, i], color=colors[i], linewidth=2, label=f"Agent {i+1}")

    plt.axhline(np.mean(state_history[0]), color='black', linestyle='--', linewidth=1.2, label="Target Average")
    plt.title("B2: Multi-Agent Average Consensus Convergence")
    plt.xlabel("Time Step")
    plt.ylabel("Agent State Value (radians)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_folder, "B2_consensus_convergence.png"), dpi=150)
    plt.show()
    print("Consensus plot saved to figures/B2_consensus_convergence.png")


if __name__ == "__main__":
    main()