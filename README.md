# Intelligent Autonomous Robotic Systems Simulation

A comprehensive robotics and multi-agent systems simulation suite built using **Webots** and **Python**, demonstrating end-to-end sensor perception, graph-based motion planning, closed-loop PID control, swarm intelligence, and neural-network decision-making.

## Project Overview

### Component A: Autonomous Mobile Robot Navigation
* **Sensor Simulation & Perception:** 2D LiDAR range scanning and RGB camera processing using OpenCV, combined via multi-modal sensor fusion into a unified occupancy grid.
* **Motion Planning:** Implementation and benchmarking of **Dijkstra's Algorithm** and **A\*** (Manhattan & Euclidean heuristics) comparing execution time, path optimality, and explored nodes.
* **Closed-Loop Control:** Kinematic robot modelling and PID controller design, including gain tuning ($K_p, K_i, K_d$) and step-response stability analysis.

### Component B: Multi-Agent Swarm & Learning System
* **Swarm Architecture:** Multi-robot Webots simulation (3+ autonomous agents) with defined roles, communication topology, and cooperative task objectives.
* **Coordination & Swarm Intelligence:** Implementation of distributed consensus and swarm optimization (PSO/ACO) to achieve emergent group behaviour.
* **Machine Learning Navigation:** PyTorch neural network trained for autonomous navigation/action selection, evaluated via accuracy metrics and confusion matrices.

## Tech Stack
* **Simulator:** Webots (R2023b)
* **Language:** Python 3.10+
* **Libraries:** NumPy, OpenCV (`cv2`), PyTorch, Matplotlib, `python-control`

## Repository Structure
* `webots_project/` – Webots world files (`.wbt`) and robot controllers
* `scripts/` – Perception, planning, control, and ML Python scripts
* `data/` – Captured sensor datasets and processed grids
* `plots/` – Matplotlib trajectories, response curves, and metrics
* `report/` – Technical documentation and evaluation

## Setup & Execution
1. Install dependencies:
   `python -m pip install numpy matplotlib opencv-python torch torchvision control seaborn plotly`
2. Run the environment verification script:
   `python scripts/step1_verify_environment.py`