import os
import math
import numpy as np
import cv2
from controller import Supervisor

# ---------- 1. Set up the robot and sensors ----------
robot = Supervisor()                         # Supervisor lets us move the robot
timestep = int(robot.getBasicTimeStep())
print("Controller started")

camera = robot.getDevice("camera")
camera.enable(timestep)
print("Camera enabled")

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
print("LiDAR enabled")

max_range = lidar.getMaxRange()              # 2.5 m (set in the world file)
cam_width = camera.getWidth()                # 160
cam_height = camera.getHeight()              # 120

# We need these two fields to place the robot at each sample position
robot_node = robot.getSelf()
translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# ---------- 2. Folders where the data will be saved ----------
# This file is in ASR700_Project/webots_A/controllers/data_collector/
# so we go up three folders to reach ASR700_Project
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.normpath(os.path.join(this_folder, "..", "..", ".."))
data_folder = os.path.join(project_folder, "data")
camera_folder = os.path.join(data_folder, "camera")
os.makedirs(camera_folder, exist_ok=True)

# ---------- 3. Sample positions and headings ----------
# Positions are in free space (not inside any obstacle)
positions = [(-0.8, -0.8), (0.7, 0.1), (-0.7, 0.1),
             (0.2, -0.7), (0.0, 0.7), (0.0, 0.0)]

# 4 headings (in radians) so the camera covers all directions
headings = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

all_scans = []        # one LiDAR scan per sample
all_poses = []        # (x, y, yaw) per sample
sample_id = 0

# ---------- 4. Collect the data ----------
for (x, y) in positions:
    for yaw in headings:
        # Move the robot to this pose
        translation_field.setSFVec3f([x, y, 0])
        rotation_field.setSFRotation([0, 0, 1, yaw])
        robot_node.resetPhysics()

        # Let the simulation run a few steps so the sensors update
        for i in range(10):
            robot.step(timestep)

        # Read the LiDAR scan (360 distances in metres)
        scan = lidar.getRangeImage()
        # Rays that hit nothing return infinity, so we replace them with max range
        scan = [max_range if math.isinf(r) else r for r in scan]
        all_scans.append(scan)

        # Read the camera image (Webots gives it as BGRA bytes)
        raw_image = camera.getImage()
        frame = np.frombuffer(raw_image, np.uint8).reshape((cam_height, cam_width, 4))
        bgr_image = frame[:, :, :3].copy()      # drop the alpha channel
        filename = "img_%03d.png" % sample_id
        cv2.imwrite(os.path.join(camera_folder, filename), bgr_image)

        # Save the pose so we know where each scan was taken
        all_poses.append([x, y, yaw])

        print("Sample", sample_id, "saved: x =", x, "y =", y,
              "heading =", round(math.degrees(yaw)), "deg")
        sample_id = sample_id + 1

# ---------- 5. Save the LiDAR scans and poses ----------
np.save(os.path.join(data_folder, "lidar_scans.npy"), np.array(all_scans))
np.save(os.path.join(data_folder, "robot_poses.npy"), np.array(all_poses))

print("Done!", sample_id, "samples saved in", data_folder)

# Pause the simulation so we can look at the final view
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)