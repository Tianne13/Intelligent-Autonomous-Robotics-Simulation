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
    return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.3)


def main():
    print("Executing Step 19: Evaluating Swarm Emergent Behaviour and Convergence...")

    # Run a simulated PSO session to record swarm dynamics over time
    num_particles = 5
    num_iterations = 30
    positions = np.random.uniform(-0.8, 0.8, size=(num_particles, 2))
    velocities = np.random.uniform(-0.1, 0.1, size=(num_particles, 2))

    p_best_positions = np.copy(positions)
    p_best_scores = np.array([objective_function(p[0], p[1]) for p in positions])
    g_best_idx = np.argmax(p_best_scores)
    g_best_position = np.copy(positions[g_best_idx])
    g_best_score = p_best_scores[g_best_idx]

    w, c1, c2 = 0.5, 1.5, 1.5

    spread_history = []
    best_fitness_history = []

    for it in range(num_iterations):
        # Compute swarm spread (average distance of agents from the swarm centroid)
        centroid = np.mean(positions, axis=0)
        dispersion = np.mean([np.linalg.norm(p - centroid) for p in positions])
        spread_history.append(dispersion)
        best_fitness_history.append(g_best_score)

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (p_best_positions[i] - positions[i]) +
                             c2 * r2 * (g_best_position - positions[i]))
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], -1.0, 1.0)

            score = objective_function(positions[i, 0], positions[i, 1])
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best_positions[i] = np.copy(positions[i])
                if score > g_best_score:
                    g_best_score = score
                    g_best_position = np.copy(positions[i])

    print("Emergence analysis complete.")
    print(f"Initial Swarm Dispersion (Spread): {spread_history[0]:.4f} m")
    print(f"Final Swarm Dispersion (Spread): {spread_history[-1]:.4f} m")

    # Plotting Swarm Convergence & Emergent Behaviour Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Swarm Dispersion (Exploration to Exploitation Transition)
    ax1.plot(range(num_iterations), spread_history, color="tab:blue", marker="o", linewidth=2)
    ax1.set_title("Swarm Dispersion (Spatial Spread)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Average Distance to Centroid (m)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Best Fitness Convergence Curve
    ax2.plot(range(num_iterations), best_fitness_history, color="tab:green", marker="s", linewidth=2)
    ax2.set_title("Global Best Fitness Convergence")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fitness Score")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "B2_emergent_convergence.png"), dpi=150)
    plt.show()
    print("Emergent convergence plot saved to figures/B2_emergent_convergence.png")


if __name__ == "__main__":
    main()