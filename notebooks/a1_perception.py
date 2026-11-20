import os
import glob
import math
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
data_folder = os.path.join(project_folder, "data")
camera_folder = os.path.join(data_folder, "camera")
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# ------------------------------------------------------------
# Part 1 - load the data from the webots controller
# ------------------------------------------------------------
scans = np.load(os.path.join(data_folder, "lidar_scans.npy"))   # 24 scans x 360 rays
poses = np.load(os.path.join(data_folder, "robot_poses.npy"))   # x, y, yaw for each scan
image_files = sorted(glob.glob(os.path.join(camera_folder, "img_*.png")))

print("LiDAR scans:", scans.shape)
print("Robot poses:", poses.shape)
print("Camera images found:", len(image_files))
if len(image_files) != len(scans):
    print("WARNING: number of images is not the same as number of scans")

# only use samples that have both a scan and an image
n_samples = min(len(image_files), len(scans))

# ------------------------------------------------------------
# Part 2 - LiDAR preprocessing
# ------------------------------------------------------------
# lidar settings (same as in the world file)
fov = 6.2832            # 360 degrees in radians
n_rays = scans.shape[1]
max_range = 2.5

# angle of each ray relative to the front of the robot
# webots numbers the rays clockwise (this setting was checked on the map)
LIDAR_CLOCKWISE = True
ray_numbers = np.arange(n_rays)
if LIDAR_CLOCKWISE:
    ray_angles = fov / 2 - ray_numbers * (fov / n_rays)
else:
    ray_angles = -fov / 2 + ray_numbers * (fov / n_rays)


def median_filter(scan, size=5):
    # takes the middle value of each ray and its neighbours
    # the scan is a circle so the ends wrap around
    half = size // 2
    n = len(scan)
    filtered = []
    for i in range(n):
        window = []
        for j in range(i - half, i + half + 1):
            window.append(scan[j % n])
        filtered.append(np.median(window))
    return np.array(filtered)


# the simulated lidar has no noise, so i add some fake noise to test the filter
np.random.seed(42)
demo_id = len(scans) - 1
clean_scan = scans[demo_id]
noisy_scan = clean_scan + np.random.normal(0, 0.01, n_rays)
spike_rays = np.random.choice(n_rays, 10, replace=False)
noisy_scan[spike_rays] = np.random.uniform(0.2, max_range, 10)   # 10 bad readings
filtered_scan = median_filter(noisy_scan, 5)

# compare the error before and after filtering
error_noisy = np.mean(np.abs(noisy_scan - clean_scan))
error_filtered = np.mean(np.abs(filtered_scan - clean_scan))
print("Mean error of noisy scan:   ", round(error_noisy, 4), "m")
print("Mean error after filtering: ", round(error_filtered, 4), "m")

# filter every scan
filtered_scans = []
for scan in scans:
    filtered_scans.append(median_filter(scan, 5))


def scan_to_world(scan, pose):
    # converts distance + angle into x, y points in the arena
    x_robot, y_robot, yaw = pose
    points = []
    for i in range(len(scan)):
        r = scan[i]
        if r < max_range - 0.01:        # skip rays that did not hit anything
            angle = yaw + ray_angles[i]
            px = x_robot + r * math.cos(angle)
            py = y_robot + r * math.sin(angle)
            points.append([px, py])
    return points


all_points = []
for i in range(len(filtered_scans)):
    all_points = all_points + scan_to_world(filtered_scans[i], poses[i])
all_points = np.array(all_points)
print("Total LiDAR hit points:", all_points.shape[0])

np.save(os.path.join(data_folder, "lidar_points_world.npy"), all_points)

# figure 1: filter result (left) and map of the lidar points (right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(clean_scan, label="Clean scan", color="green")
ax1.plot(noisy_scan, label="Noisy scan (noise + spikes)", color="red", alpha=0.5)
ax1.plot(filtered_scan, label="After median filter", color="blue")
ax1.set_title("LiDAR preprocessing - median filter (sample " + str(demo_id) + ")")
ax1.set_xlabel("Ray number")
ax1.set_ylabel("Distance (m)")
ax1.legend()
ax1.grid(True)

ax2.scatter(all_points[:, 0], all_points[:, 1], s=3, color="black", label="LiDAR hits")
ax2.plot(poses[:, 0], poses[:, 1], "r^", label="Robot positions")
# real obstacle outlines from the world file, just to check the points line up
ax2.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="green",
                        linestyle="--", label="Real obstacle outlines"))
ax2.add_patch(Rectangle((0.4 - 0.1, 0.3 - 0.1), 0.2, 0.2, fill=False,
                        edgecolor="green", linestyle="--"))
ax2.add_patch(Rectangle((-0.3 - 0.125, -0.2 - 0.075), 0.25, 0.15, fill=False,
                        edgecolor="green", linestyle="--"))
ax2.add_patch(Rectangle((-0.4 - 0.15, 0.5 - 0.05), 0.3, 0.1, fill=False,
                        edgecolor="green", linestyle="--"))
ax2.add_patch(Circle((0.5, -0.5), 0.06, fill=False, edgecolor="green",
                     linestyle="--"))
ax2.set_title("LiDAR points in the arena frame (sanity check)")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_aspect("equal")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A1_lidar_preprocessing.png"), dpi=150)
plt.show()

# ------------------------------------------------------------
# Part 3 - camera detection with OpenCV
# ------------------------------------------------------------
# camera settings
cam_width = 160
cam_height = 120
cam_fov = 0.84          # e-puck camera field of view in radians
focal = (cam_width / 2) / math.tan(cam_fov / 2)

# HSV ranges for each colour (hue goes 0 to 179 in opencv)
# red needs two ranges because the hue wraps around 0
# the shaded sides of the boxes are dark, so the value minimum for red and
# orange is low (40) but the saturation minimum is high (180) so the
# brown floor tiles are not picked up
colour_ranges = {
    "red":    [((0, 180, 40), (6, 255, 255)), ((172, 180, 40), (179, 255, 255))],
    "orange": [((8, 180, 40), (22, 255, 255))],
    "yellow": [((24, 150, 130), (34, 255, 255))],
    "green":  [((40, 120, 60), (80, 255, 255))],
    "blue":   [((100, 150, 60), (130, 255, 255))],
}

# what each colour stands for in my world
colour_roles = {"red": "obstacle", "blue": "obstacle", "orange": "obstacle",
                "green": "landmark", "yellow": "goal"}

# box colours for drawing (BGR order in opencv)
draw_colours = {"red": (0, 0, 255), "blue": (255, 0, 0), "orange": (0, 140, 255),
                "green": (0, 200, 0), "yellow": (0, 255, 255)}

min_area = 30           # ignore blobs smaller than this (in pixels)

# the yellow goal is a flat disc so it only looks like a thin strip
# opening would delete it, so it is skipped for yellow and the min area is smaller
thin_colours = ["yellow"]
min_areas = {"yellow": 8}


def detect_objects(image_bgr):
    # finds the coloured objects in one image
    # gives back the detections, the blurred image and the combined mask

    # blur first to reduce noise
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    # hsv separates colour from brightness
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = np.zeros(hsv.shape[:2], np.uint8)
    found = []

    for colour_name, ranges in colour_ranges.items():
        # mask for this colour
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for (low, high) in ranges:
            low_array = np.array(low, dtype=np.uint8)
            high_array = np.array(high, dtype=np.uint8)
            part = cv2.inRange(hsv, low_array, high_array)
            mask = cv2.bitwise_or(mask, part)

        # opening removes small specks, closing fills small holes
        if colour_name not in thin_colours:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        # outlines of the blobs
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[-2]       # [-2] works for old and new opencv versions

        smallest = min_areas.get(colour_name, min_area)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= smallest:
                x, y, w, h = cv2.boundingRect(contour)
                centre_x = x + w / 2
                # bearing = angle to the object, positive means to the left
                bearing = -math.atan((centre_x - cam_width / 2) / focal)
                # if the box touches the edge we only see part of the object
                touches_edge = (x <= 0) or (x + w >= cam_width)
                found.append({"colour": colour_name,
                              "role": colour_roles[colour_name],
                              "x": x, "y": y, "w": w, "h": h,
                              "area": area, "bearing": bearing,
                              "touches_edge": touches_edge})
    return found, blurred, combined_mask


def draw_detections(image_bgr, found):
    # draws a box and the colour name on each detection
    output = image_bgr.copy()
    for d in found:
        colour = draw_colours[d["colour"]]
        cv2.rectangle(output, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), colour, 1)
        cv2.putText(output, d["colour"], (d["x"], max(d["y"] - 3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return output


# run the detector on every image
all_detections = []
images = []
detections_per_sample = []

for i in range(n_samples):
    image = cv2.imread(image_files[i])
    images.append(image)
    found, blurred, mask = detect_objects(image)
    detections_per_sample.append(found)

    for d in found:
        row = dict(d)
        row["sample"] = i
        row["robot_x"] = poses[i][0]
        row["robot_y"] = poses[i][1]
        row["robot_yaw"] = poses[i][2]
        row["world_bearing"] = poses[i][2] + d["bearing"]    # bearing in the arena frame
        all_detections.append(row)

# save the detections (used later for the sensor fusion)
table = pd.DataFrame(all_detections)
table.to_csv(os.path.join(data_folder, "camera_detections.csv"), index=False)

print("Total detections:", len(table))
if len(table) > 0:
    print(table["colour"].value_counts())
    print("Detections touching the image edge:", int(table["touches_edge"].sum()))

# figure 2: detections on 8 example images
example_ids = np.linspace(0, n_samples - 1, 8, dtype=int)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, i in zip(axes.flatten(), example_ids):
    annotated = draw_detections(images[i], detections_per_sample[i])
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    heading = round(math.degrees(poses[i][2]))
    ax.set_title("Sample " + str(i) + " (heading " + str(heading) + " deg)", fontsize=9)
    ax.axis("off")
plt.suptitle("Camera object detection (OpenCV, HSV thresholding)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A1_camera_detections.png"), dpi=150)
plt.show()

# figure 3: the pipeline stages for the image with the most detections
best_id = 0
for i in range(n_samples):
    if len(detections_per_sample[i]) > len(detections_per_sample[best_id]):
        best_id = i

original = images[best_id]
found, blurred, mask = detect_objects(original)
annotated = draw_detections(original, found)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Original image")
axes[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. After Gaussian blur")
axes[2].imshow(mask, cmap="gray")
axes[2].set_title("3. Colour masks (cleaned)")
axes[3].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
axes[3].set_title("4. Detected objects")
for ax in axes:
    ax.axis("off")
plt.suptitle("Camera perception pipeline (sample " + str(best_id) + ")")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A1_camera_pipeline.png"), dpi=150)
plt.show()

print("Done. Figures saved in:", figures_folder)