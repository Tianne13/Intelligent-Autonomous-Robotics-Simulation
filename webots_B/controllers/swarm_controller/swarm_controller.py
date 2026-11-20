import math
import numpy as np
from controller import Robot


def main():
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    robot_name = robot.getName()

    # Initialize wheel motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Initialize LiDAR sensor for obstacle and neighbor detection
    lidar = robot.getDevice("lidar")
    lidar.enable(time_step)

    print(f"Swarm Agent [{robot_name}] initialized successfully.")

    while robot.step(time_step) != -1:
        ranges = lidar.getRangeImage()
        min_obstacle = min(ranges)

        # Simple baseline cooperative behavior: move forward, turn if obstacle close
        if min_obstacle < 0.2:
            # Turn away from obstacle
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(-1.0)
        else:
            # Cruise speed
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)


if __name__ == "__main__":
    main()