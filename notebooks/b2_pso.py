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


def objective_function(x, y):
    """
    Objective fitness function representing target intensity or resource distribution.
    We want the swarm to locate the global peak (e.g., target location at x=0.5, y=0.5).
    """
    # Gaussian peak centered at (0.5, 0.5) with some surrounding gradient
    peak = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.3)
    return peak


def main():
    print("Executing Step 18: Multi-Agent Particle Swarm Optimisation (PSO)...")

    # 1. PSO Parameters
    num_particles = 5
    num_iterations = 30
    
    # Initialize particle positions randomly within arena [-1.0, 1.0]
    positions = np.random.uniform(-0.8, 0.8, size=(num_particles, 2))
    velocities = np.random.uniform(-0.1, 0.1, size=(num_particles, 2))

    # Initialize personal bests
    p_best_positions = np.copy(positions)
    p_best_scores = np.array([objective_function(p[0], p[1]) for p in positions])

    # Initialize global best
    g_best_idx = np.argmax(p_best_scores)
    g_best_position = np.copy(positions[g_best_idx])
    g_best_score = p_best_scores[g_best_idx]

    # History for tracking trajectories
    trajectory_history = [np.copy(positions)]

    # Hyperparameters for velocity update
    w = 0.5   # Inertia weight
    c1 = 1.5  # Cognitive parameter (personal best)
    c2 = 1.5  # Social parameter (global best)

    # 2. PSO Iterative Search Loop
    for it in range(num_iterations):
        for i in range(num_particles):
            # Random coefficients
            r1, r2 = np.random.rand(), np.random.rand()

            # Update velocity
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (p_best_positions[i] - positions[i]) +
                             c2 * r2 * (g_best_position - positions[i]))

            # Update position
            positions[i] += velocities[i]

            # Clip to arena boundaries [-1.0, 1.0]
            positions[i] = np.clip(positions[i], -1.0, 1.0)

            # Evaluate fitness
            score = objective_function(positions[i, 0], positions[i, 1])

            # Update personal best
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best_positions[i] = np.copy(positions[i])

                # Update global best
                if score > g_best_score:
                    g_best_score = score
                    g_best_position = np.copy(positions[i])

        trajectory_history.append(np.copy(positions))

    print(f"PSO Optimization Complete.")
    print(f"Global Best Target Found at Position: {g_best_position} with Fitness: {g_best_score:.4f}")

    # 3. Plotting PSO Trajectories
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create background fitness contour
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="Target Fitness Score")

    # Plot particle paths
    history_arr = np.array(trajectory_history) # shape: (iterations, particles, 2)
    colors = ['tab:red', 'tab:orange', 'tab:pink', 'tab:cyan', 'tab:purple']
    
    for i in range(num_particles):
        ax.plot(history_arr[:, i, 0], history_arr[:, i, 1], marker='o', markersize=3, color=colors[i], label=f"Agent {i+1}")
        ax.scatter(history_arr[-1, i, 0], history_arr[-1, i, 1], color=colors[i], s=80, edgecolors='black')

    ax.scatter(g_best_position[0], g_best_position[1], color='gold', marker='*', s=200, edgecolors='black', zorder=10, label="Global Optimum")

    ax.set_title("B2: Particle Swarm Optimisation (PSO) Search Trajectories")
    ax.set_xlabel("Arena X (m)")
    ax.set_ylabel("Arena Y (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_folder, "B2_pso_trajectory.png"), dpi=150)
    plt.show()
    print("PSO trajectory plot saved to figures/B2_pso_trajectory.png")


if __name__ == "__main__":
    main()