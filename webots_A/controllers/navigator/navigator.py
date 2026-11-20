import os
import sys
import math
import numpy as np
import cv2
from controller import Supervisor

# folders (this file is in ASR700_Project/webots_A/controllers/navigator/)
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.normpath(os.path.join(this_folder, "..", "..", ".."))
data_folder = os.path.join(project_folder, "data")
notebooks_folder = os.path.join(project_folder, "notebooks")
sys.path.append(notebooks_folder)       # so i can import the model, grid and pid files

from robot_model import wrap_angle, vw_to_wheels, limit_wheels
from grid_map import world_to_cell, cell_to_world
from pid_controller import PID, make_dense_path
from planner_astar import a_star

# ------------------------------------------------------------
# 1. robot, motors and lidar
# ------------------------------------------------------------
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0                  # time step in seconds

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))    # velocity control
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)

robot_node = robot.getSelf()


def get_pose():
    # position and heading of the robot in the arena (taken from the simulator)
    position = robot_node.getPosition()
    rotation = robot_node.getOrientation()
    heading = math.atan2(rotation[3], rotation[0])
    return position[0], position[1], heading


def stop_robot():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)


# ------------------------------------------------------------
# 2. load the map from the perception step and plan the path
# ------------------------------------------------------------
fused_grid = np.load(os.path.join(data_folder, "fused_grid.npy"))
goal_estimate = np.load(os.path.join(data_folder, "goal_estimate.npy"))

# inflate the obstacles by one cell like in the planning step
blocked = np.uint8(fused_grid > 0)
planning_grid = cv2.dilate(blocked, np.ones((3, 3), np.uint8))

# pid gains from the tuning step (first guess gains if the file is missing)
gains_file = os.path.join(data_folder, "pid_gains.npy")
if os.path.exists(gains_file):
    gains = np.load(gains_file)
    kp = float(gains[0])
    ki = float(gains[1])
    kd = float(gains[2])
else:
    kp, ki, kd = 2.0, 0.1, 0.2

robot.step(timestep)                    # one step so the pose is valid
x, y, theta = get_pose()
start_world = (x, y)
goal_world = (float(goal_estimate[0]), float(goal_estimate[1]))
start_cell = world_to_cell(start_world[0], start_world[1])
goal_cell = world_to_cell(goal_world[0], goal_world[1])

print("Start (m):", round(x, 3), round(y, 3), " heading:", round(math.degrees(theta), 1), "deg")
print("Goal from the camera (m):", round(goal_world[0], 3), round(goal_world[1], 3))
print("PID gains: Kp =", kp, " Ki =", ki, " Kd =", kd)

if planning_grid[start_cell[0], start_cell[1]] == 1 or planning_grid[goal_cell[0], goal_cell[1]] == 1:
    print("Start or goal cell is blocked, cannot plan")
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    sys.exit()

path_cells, cost, explored = a_star(planning_grid, start_cell, goal_cell, "euclidean")
if path_cells is None:
    print("No path found")
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    sys.exit()
print("A* path:", len(path_cells), "cells,", len(explored), "nodes expanded")

# waypoints: real start, centres of the path cells, goal from the camera
waypoints = [start_world]
for i in range(1, len(path_cells) - 1):
    r, c = path_cells[i]
    waypoints.append(cell_to_world(r, c))
waypoints.append(goal_world)
dense_path = make_dense_path(waypoints, 0.01)
planned_length = np.sum(np.hypot(np.diff(dense_path[:, 0]), np.diff(dense_path[:, 1])))
print("Planned path length:", round(planned_length, 3), "m")
np.save(os.path.join(data_folder, "webots_planned_path.npy"), dense_path)

# ------------------------------------------------------------
# 3. closed loop control
# ------------------------------------------------------------
pid = PID(kp, ki, kd, out_limit=2.0, int_limit=1.0)
v_nom = 0.08            # forward speed (m/s)
lookahead = 0.12        # how far ahead on the path the robot aims (m)
goal_tol = 0.03         # stop when this close to the goal (m)
safe_distance = 0.055   # emergency stop if the lidar sees something this close (m)
max_time = 90.0         # give up after this many seconds

log_t = []
log_x = []
log_y = []
log_theta = []
log_heading_error = []
log_cross_track = []
log_v = []
log_w = []
log_lidar_min = []

t = 0.0
closest = 0
reached = False
emergency_stop = False
goal_x, goal_y = dense_path[-1]

while robot.step(timestep) != -1:
    t = t + dt
    x, y, theta = get_pose()

    # nearest lidar reading (all directions)
    ranges = lidar.getRangeImage()
    lidar_min = min(ranges)

    # closest point on the path (only looks forward from the last one)
    best = closest
    best_dist = 1e9
    for i in range(closest, min(closest + 60, len(dense_path))):
        d = math.hypot(dense_path[i][0] - x, dense_path[i][1] - y)
        if d < best_dist:
            best_dist = d
            best = i
    closest = best

    # signed cross track error (positive = robot is left of the path)
    nxt = min(closest + 1, len(dense_path) - 1)
    tx = dense_path[nxt][0] - dense_path[closest][0]
    ty = dense_path[nxt][1] - dense_path[closest][1]
    rx = x - dense_path[closest][0]
    ry = y - dense_path[closest][1]
    length = math.hypot(tx, ty)
    if length > 0:
        cross_track = (tx * ry - ty * rx) / length
    else:
        cross_track = best_dist

    # target point a bit further along the path
    target = closest
    while target < len(dense_path) - 1:
        if math.hypot(dense_path[target][0] - x, dense_path[target][1] - y) >= lookahead:
            break
        target = target + 1
    target_x, target_y = dense_path[target]

    # heading error -> pid -> turn rate
    desired_heading = math.atan2(target_y - y, target_x - x)
    heading_error = wrap_angle(desired_heading - theta)
    w_cmd = pid.update(heading_error, dt)

    # forward speed: slower when pointing the wrong way and near the goal
    v_cmd = v_nom * max(0.2, math.cos(min(abs(heading_error), math.pi / 2)))
    dist_to_goal = math.hypot(goal_x - x, goal_y - y)
    v_cmd = min(v_cmd, max(0.02, dist_to_goal))

    # log this step
    log_t.append(t)
    log_x.append(x)
    log_y.append(y)
    log_theta.append(theta)
    log_heading_error.append(heading_error)
    log_cross_track.append(cross_track)
    log_v.append(v_cmd)
    log_w.append(w_cmd)
    log_lidar_min.append(lidar_min)

    # stop conditions (the first few lidar readings are skipped)
    if dist_to_goal < goal_tol:
        reached = True
        break
    if len(log_t) > 5 and lidar_min < safe_distance:
        emergency_stop = True
        break
    if t > max_time:
        break

    # send the speeds to the wheels
    wl, wr = vw_to_wheels(v_cmd, w_cmd)
    wl, wr = limit_wheels(wl, wr)
    left_motor.setVelocity(wl)
    right_motor.setVelocity(wr)

stop_robot()

# ------------------------------------------------------------
# 4. save the log and print the results
# ------------------------------------------------------------
np.savez(os.path.join(data_folder, "webots_run_log.npz"),
         t=np.array(log_t), x=np.array(log_x), y=np.array(log_y),
         theta=np.array(log_theta), heading_error=np.array(log_heading_error),
         cross_track=np.array(log_cross_track), v=np.array(log_v), w=np.array(log_w),
         lidar_min=np.array(log_lidar_min), gains=np.array([kp, ki, kd]),
         reached=reached)

xs = np.array(log_x)
ys = np.array(log_y)
cross = np.array(log_cross_track)
travelled = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
final_distance = math.hypot(xs[-1] - goal_x, ys[-1] - goal_y)

print("\nGoal reached:", reached)
if emergency_stop:
    print("Stopped by the lidar safety check, nearest object:", round(lidar_min, 3), "m")
print("Time to goal:", round(t, 1), "s")
print("Distance travelled:", round(travelled, 3), "m")
print("Final distance to the goal:", round(final_distance, 3), "m")
print("Cross track error: mean", round(np.mean(np.abs(cross)), 4), "m, max",
      round(np.max(np.abs(cross)), 4), "m, rms", round(math.sqrt(np.mean(cross ** 2)), 4), "m")
print("Smallest lidar distance during the run:", round(min(log_lidar_min), 3), "m")
print("Log saved in:", data_folder)

# pause so i can look at the final view
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)