import os
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# load everything from the earlier steps
scans = np.load(os.path.join(data_folder, "lidar_scans.npy"))
poses = np.load(os.path.join(data_folder, "robot_poses.npy"))
points = np.load(os.path.join(data_folder, "lidar_points_world.npy"))
detections = pd.read_csv(os.path.join(data_folder, "camera_detections.csv"))

print("LiDAR scans:", scans.shape)
print("LiDAR world points:", points.shape)
print("Camera detections:", len(detections))

# ------------------------------------------------------------
# grid settings
# ------------------------------------------------------------
arena_min = -1.0                        # arena goes from -1 to 1 in x and y
arena_size = 2.0
resolution = 0.05                       # each cell is 5 cm
n_cells = int(round(arena_size / resolution))     # 40 cells in each direction


def world_to_cell(x, y):
    # x, y in metres -> (row, col) in the grid
    # points a bit outside the arena are stray readings so they are dropped
    if x < -1.05 or x > 1.05 or y < -1.05 or y > 1.05:
        return None
    col = int((x - arena_min) / resolution)
    row = int((y - arena_min) / resolution)
    col = min(max(col, 0), n_cells - 1)
    row = min(max(row, 0), n_cells - 1)
    return row, col


def cell_to_world(row, col):
    # centre of a cell in metres
    x = arena_min + (col + 0.5) * resolution
    y = arena_min + (row + 0.5) * resolution
    return x, y


# ------------------------------------------------------------
# Part 1 - lidar occupancy grid
# ------------------------------------------------------------
# count how many lidar points fall in each cell
hit_count = np.zeros((n_cells, n_cells), dtype=int)
for p in points:
    cell = world_to_cell(p[0], p[1])
    if cell is not None:
        hit_count[cell[0], cell[1]] += 1

# a cell is occupied if it got at least 2 hits (removes the stray single points)
lidar_grid = np.uint8(hit_count >= 2)

# group the occupied cells into objects (8-connected neighbours)
num, labels, stats, centroids = cv2.connectedComponentsWithStats(lidar_grid, connectivity=8)

# drop tiny components (less than 3 cells), they are just noise
clean_grid = np.zeros_like(lidar_grid)
for k in range(1, num):
    if stats[k, cv2.CC_STAT_AREA] >= 3:
        clean_grid[labels == k] = 1

# run the grouping again on the cleaned grid
num, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_grid, connectivity=8)

# the wall ring is the component that spans most of the arena
wall_ids = []
obstacle_ids = []
for k in range(1, num):
    width = stats[k, cv2.CC_STAT_WIDTH]
    height = stats[k, cv2.CC_STAT_HEIGHT]
    if width >= 30 or height >= 30:
        wall_ids.append(k)
    else:
        obstacle_ids.append(k)

print("Occupied cells after cleaning:", int(clean_grid.sum()))
print("Wall components:", len(wall_ids), " Obstacle components:", len(obstacle_ids))

# the lidar only sees the outside edge of each obstacle, so fill the inside
filled_masks = {}
for k in obstacle_ids:
    mask = np.uint8(labels == k)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, -1)
    filled = np.maximum(filled, mask)
    filled_masks[k] = filled

wall_mask = np.zeros((n_cells, n_cells), dtype=bool)
for k in wall_ids:
    wall_mask[labels == k] = True

# ------------------------------------------------------------
# Part 2 - match the camera detections to the lidar objects
# ------------------------------------------------------------
lidar_fov = 6.2832
n_rays = scans.shape[1]
ray_step = lidar_fov / n_rays
max_range = 2.5


def bearing_to_ray(bearing):
    # camera bearing (positive = left) -> lidar ray number
    # same ray order as in a1_perception.py (rays go clockwise)
    index = int(round((lidar_fov / 2 - bearing) / ray_step))
    return index % n_rays


def find_component(row, col):
    # looks at the cell and the cells around it for a lidar component
    for dr in [0, -1, 1]:
        for dc in [0, -1, 1]:
            r2 = row + dr
            c2 = col + dc
            if 0 <= r2 < n_cells and 0 <= c2 < n_cells:
                if labels[r2, c2] > 0:
                    return labels[r2, c2]
    return 0


votes = {}                   # component number -> {colour: number of votes}
fused_x = []
fused_y = []
fused_colour = []
n_on_obstacle = 0
n_on_wall = 0
n_no_hit = 0
n_no_component = 0
n_edge_total = 0
n_edge_ok = 0

obstacle_detections = detections[detections["colour"] != "yellow"]

for index, row in obstacle_detections.iterrows():
    sample = int(row["sample"])
    touches_edge = bool(row["touches_edge"])
    if touches_edge:
        n_edge_total += 1

    # lidar distance in the direction of the detection (median of 5 rays)
    ray = bearing_to_ray(row["bearing"])
    window = []
    for k in range(-2, 3):
        window.append(scans[sample][(ray + k) % n_rays])
    r = np.median(window)

    if r >= max_range - 0.01:            # the ray did not hit anything
        n_no_hit += 1
        continue

    # position of the detected object in the arena
    angle = row["robot_yaw"] + row["bearing"]
    px = row["robot_x"] + r * math.cos(angle)
    py = row["robot_y"] + r * math.sin(angle)

    cell = world_to_cell(px, py)
    if cell is None:
        n_no_component += 1
        continue
    comp = find_component(cell[0], cell[1])
    if comp == 0:
        n_no_component += 1
        continue
    if comp in wall_ids:                 # ray went past the object and hit the wall
        n_on_wall += 1
        continue

    # camera colour matched to a lidar obstacle, add a vote
    n_on_obstacle += 1
    if touches_edge:
        n_edge_ok += 1
    colour = row["colour"]
    if comp not in votes:
        votes[comp] = {}
    votes[comp][colour] = votes[comp].get(colour, 0) + 1
    fused_x.append(px)
    fused_y.append(py)
    fused_colour.append(colour)

print("\nAssociation of camera detections with lidar objects:")
print("  matched to an obstacle:", n_on_obstacle)
print("  ray hit the wall instead:", n_on_wall)
print("  ray hit nothing:", n_no_hit)
print("  no lidar object nearby:", n_no_component)
print("  matched while touching the image edge:", n_edge_ok, "of", n_edge_total)

# each obstacle gets the colour with the most votes
class_ids = {"red": 2, "blue": 3, "orange": 4, "green": 5}
unlabelled_id = 6

fused_grid = np.zeros((n_cells, n_cells), dtype=np.uint8)    # 0 = free
fused_grid[wall_mask] = 1                                    # 1 = wall

true_centres = {"red": (0.4, 0.3), "blue": (-0.3, -0.2),
                "orange": (-0.4, 0.5), "green": (0.5, -0.5)}

print("\nLabelled obstacles:")
for k in obstacle_ids:
    if k in votes:
        best_colour = max(votes[k], key=votes[k].get)
        class_id = class_ids[best_colour]
    else:
        best_colour = "unlabelled"
        class_id = unlabelled_id
    fused_grid[filled_masks[k] == 1] = class_id

    # centre of the object in metres
    rows_k, cols_k = np.nonzero(filled_masks[k])
    cx, cy = cell_to_world(rows_k.mean(), cols_k.mean())
    message = "  component " + str(k) + ": " + best_colour + ", votes " + str(votes.get(k, {}))
    message = message + ", centre (" + str(round(cx, 2)) + ", " + str(round(cy, 2)) + ")"
    if best_colour in true_centres:
        tx, ty = true_centres[best_colour]
        error = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        message = message + ", position error " + str(round(error, 3)) + " m"
    print(message)

# ------------------------------------------------------------
# Part 3 - the goal (flat yellow disc, the lidar cannot see it)
# ------------------------------------------------------------
start_point = (-0.8, -0.8)
true_goal = (0.8, 0.8)             # the goal i defined in the world file
yellow = detections[detections["colour"] == "yellow"]
goal_estimate = None

# every yellow detection gives a line from the robot, the goal is where the lines meet
if len(yellow) >= 2:
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for index, row in yellow.iterrows():
        angle = row["robot_yaw"] + row["bearing"]
        d = np.array([math.cos(angle), math.sin(angle)])
        p = np.array([row["robot_x"], row["robot_y"]])
        M = np.eye(2) - np.outer(d, d)
        A = A + M
        b = b + np.dot(M, p)
    if abs(np.linalg.det(A)) > 0.1:        # lines are not almost parallel
        goal_estimate = np.linalg.solve(A, b)

print("\nGoal from the camera:")
if goal_estimate is not None:
    goal_error = math.sqrt((goal_estimate[0] - true_goal[0]) ** 2 +
                           (goal_estimate[1] - true_goal[1]) ** 2)
    print("  estimated goal:", round(goal_estimate[0], 3), round(goal_estimate[1], 3))
    print("  defined goal:  ", true_goal)
    print("  error:", round(goal_error, 3), "m")
else:
    print("  could not triangulate (need 2 yellow detections from different places)")
    print("  using the defined goal:", true_goal)

# ------------------------------------------------------------
# Part 4 - check the map against the real world
# ------------------------------------------------------------
# build the real grid from the world file
true_grid = np.zeros((n_cells, n_cells), dtype=np.uint8)
true_grid[0, :] = 1
true_grid[n_cells - 1, :] = 1
true_grid[:, 0] = 1
true_grid[:, n_cells - 1] = 1

# real obstacles: (centre x, centre y, width, height) and the cylinder
boxes = [(0.4, 0.3, 0.2, 0.2), (-0.3, -0.2, 0.25, 0.15), (-0.4, 0.5, 0.3, 0.1)]
for row_i in range(n_cells):
    for col_i in range(n_cells):
        x, y = cell_to_world(row_i, col_i)
        for (bx, by, bw, bh) in boxes:
            if abs(x - bx) <= bw / 2 and abs(y - by) <= bh / 2:
                true_grid[row_i, col_i] = 1
        if (x - 0.5) ** 2 + (y + 0.5) ** 2 <= 0.06 ** 2:
            true_grid[row_i, col_i] = 1

detected = np.uint8(fused_grid > 0)
kernel = np.ones((3, 3), np.uint8)
true_dilated = cv2.dilate(true_grid, kernel)
detected_dilated = cv2.dilate(detected, kernel)

# precision: detected cells that are within 1 cell of a real object
# recall: real cells that are within 1 cell of a detected cell
precision = np.sum((detected == 1) & (true_dilated == 1)) / np.sum(detected)
recall = np.sum((true_grid == 1) & (detected_dilated == 1)) / np.sum(true_grid)
print("\nOccupancy grid accuracy (1 cell tolerance):")
print("  precision:", round(precision, 3))
print("  recall:   ", round(recall, 3))

# save the fused map for the path planning step
np.save(os.path.join(data_folder, "fused_grid.npy"), fused_grid)
if goal_estimate is not None:
    np.save(os.path.join(data_folder, "goal_estimate.npy"), goal_estimate)

# ------------------------------------------------------------
# figure: lidar only, camera rays, fused map
# ------------------------------------------------------------
plot_colours = {"red": "red", "blue": "blue", "orange": "orange",
                "green": "green", "yellow": "gold"}
extent = [-1, 1, -1, 1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# left: lidar only
ax1.imshow(clean_grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
ax1.set_title("1. LiDAR occupancy grid (geometry only)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# middle: camera bearing lines and where they hit the lidar objects
for index, row in detections.iterrows():
    angle = row["robot_yaw"] + row["bearing"]
    x0 = row["robot_x"]
    y0 = row["robot_y"]
    style = "--" if bool(row["touches_edge"]) else "-"
    ax2.plot([x0, x0 + 2.0 * math.cos(angle)], [y0, y0 + 2.0 * math.sin(angle)],
             style, color=plot_colours[row["colour"]], alpha=0.35, linewidth=1)
for i in range(len(fused_x)):
    ax2.scatter(fused_x[i], fused_y[i], s=40, color=plot_colours[fused_colour[i]],
                edgecolors="black", zorder=3)
ax2.plot(poses[:, 0], poses[:, 1], "k^", markersize=6, label="Robot positions")
if goal_estimate is not None:
    ax2.plot(goal_estimate[0], goal_estimate[1], marker="*", color="gold",
             markersize=16, markeredgecolor="black", linestyle="none",
             label="Goal (triangulated)")
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_aspect("equal")
ax2.grid(True)
ax2.legend(loc="lower left", fontsize=8)
ax2.set_title("2. Camera bearings matched to LiDAR range")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")

# right: fused map
cmap = ListedColormap(["white", "black", "red", "blue", "orange", "green", "gray"])
ax3.imshow(fused_grid, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=6)
ax3.plot(start_point[0], start_point[1], "ks", markersize=9, label="Start")
ax3.plot(true_goal[0], true_goal[1], marker="*", color="gold", markersize=16,
         markeredgecolor="black", linestyle="none", label="Goal")
legend_items = [Patch(facecolor="black", label="Wall"),
                Patch(facecolor="red", label="Red obstacle"),
                Patch(facecolor="blue", label="Blue obstacle"),
                Patch(facecolor="orange", label="Orange obstacle"),
                Patch(facecolor="green", label="Green landmark")]
handles, names = ax3.get_legend_handles_labels()
ax3.legend(handles=handles + legend_items, loc="lower left", fontsize=7)
ax3.set_title("3. Fused map (LiDAR shape + camera labels)")
ax3.set_xlabel("x (m)")
ax3.set_ylabel("y (m)")

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A1_sensor_fusion.png"), dpi=150)
plt.show()

print("\nDone. Figure saved in:", figures_folder)
print("Fused grid saved in:", data_folder)