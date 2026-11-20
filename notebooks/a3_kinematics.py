import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from robot_model import (wheel_radius, axle_length, max_wheel_speed, max_v, dt,
                         motor_tau, wheels_to_vw, vw_to_wheels,
                         kinematic_step, dynamic_step)

# folders
this_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(this_folder)
figures_folder = os.path.join(project_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# ------------------------------------------------------------
# 1. robot parameters
# ------------------------------------------------------------
max_w = 2 * wheel_radius * max_wheel_speed / axle_length    # spinning on the spot
print("Wheel radius:", wheel_radius, "m")
print("Axle length:", axle_length, "m")
print("Max wheel speed:", max_wheel_speed, "rad/s")
print("Max forward speed:", round(max_v, 4), "m/s")
print("Max turn rate (spinning on the spot):", round(max_w, 3), "rad/s")
print("Time step:", dt, "s, motor time constant (assumed):", motor_tau, "s")

# when driving at speed v the wheels have less room left for turning
v_drive = 0.1
w_room = (max_v - v_drive) / (axle_length / 2)
print("At v =", v_drive, "m/s the largest turn rate before a wheel saturates is",
      round(w_room, 2), "rad/s")

# ------------------------------------------------------------
# 2. check the wheel conversion (there and back again)
# ------------------------------------------------------------
v_test = 0.08
w_test = 1.2
wl, wr = vw_to_wheels(v_test, w_test)
v_back, w_back = wheels_to_vw(wl, wr)
print("\nConversion check: v, w =", v_test, w_test)
print("  wheel speeds (left, right):", round(wl, 3), round(wr, 3), "rad/s")
print("  back to v, w:", round(v_back, 4), round(w_back, 4))

# ------------------------------------------------------------
# 3. open loop tests with the kinematic model
# ------------------------------------------------------------
def run_kinematic(v_list, w_list):
    # runs the kinematic model for a list of commands, returns the path and final heading
    x = 0.0
    y = 0.0
    theta = 0.0
    xs = [x]
    ys = [y]
    for i in range(len(v_list)):
        x, y, theta = kinematic_step(x, y, theta, v_list[i], w_list[i])
        xs.append(x)
        ys.append(y)
    return xs, ys, theta


# straight line: v = 0.1 m/s for 5 s
n = int(round(5.0 / dt))
straight_x, straight_y, straight_theta = run_kinematic([0.1] * n, [0.0] * n)
print("\nStraight line test:")
print("  distance travelled:", round(straight_x[-1], 4), "m, expected", round(0.1 * n * dt, 4), "m")

# circle: v = 0.1, w = 0.5 so the radius should be v / w = 0.2 m
w_circle = 0.5
n = int(round(2 * math.pi / w_circle / dt))
circle_x, circle_y, circle_theta = run_kinematic([0.1] * n, [w_circle] * n)
closing_error = math.sqrt(circle_x[-1] ** 2 + circle_y[-1] ** 2)
print("Circle test (expected radius", 0.1 / w_circle, "m):")
print("  distance from the start after one full circle:", round(closing_error, 4), "m")

# spin on the spot: v = 0, w = 1 rad/s for about pi seconds
n = int(round(math.pi / dt))
spin_x, spin_y, spin_theta = run_kinematic([0.0] * n, [1.0] * n)
print("Spin test:")
print("  final heading:", round(math.degrees(spin_theta), 1), "deg (expected about 180)")

# S-curve: turn left for 4 s then right for 4 s
n = int(round(4.0 / dt))
s_x, s_y, s_theta = run_kinematic([0.1] * (2 * n), [0.8] * n + [-0.8] * n)

# ------------------------------------------------------------
# 4. dynamic model: motor lag step response
# ------------------------------------------------------------
v_step = 0.1
wl_command = vw_to_wheels(v_step, 0.0)[0]
state = (0.0, 0.0, 0.0, 0.0, 0.0)
step_time = []
wl_actual = []
n_resp = int(round(1.0 / dt))
for i in range(n_resp + 1):
    step_time.append(i * dt)
    wl_actual.append(state[3])
    state = dynamic_step(state, v_step, 0.0)

# time to reach 63% of the command (should be close to the time constant)
t63 = None
for i in range(len(wl_actual)):
    if wl_actual[i] >= 0.632 * wl_command:
        t63 = step_time[i]
        break
print("\nMotor lag step response:")
print("  commanded wheel speed:", round(wl_command, 3), "rad/s")
print("  time to reach 63% of the command:", round(t63, 3), "s (time constant", motor_tau, "s)")

# ------------------------------------------------------------
# 5. kinematic model vs model with motor lag (same commands)
# ------------------------------------------------------------
n = int(round(5.0 / dt))
kin_x, kin_y, kin_theta = run_kinematic([0.1] * n, [1.0] * n)

state = (0.0, 0.0, 0.0, 0.0, 0.0)
dyn_x = [0.0]
dyn_y = [0.0]
for i in range(n):
    state = dynamic_step(state, 0.1, 1.0)
    dyn_x.append(state[0])
    dyn_y.append(state[1])

gap = math.sqrt((kin_x[-1] - dyn_x[-1]) ** 2 + (kin_y[-1] - dyn_y[-1]) ** 2)
print("\nKinematic vs motor lag model (v = 0.1, w = 1.0 for 5 s):")
print("  difference in the final position:", round(gap, 4), "m")

# ------------------------------------------------------------
# 6. figure
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
ax1, ax2, ax3, ax4 = axes.flatten()

# panel 1: drawing of the robot (top view)
ax1.add_patch(Circle((0, 0), 0.037, fill=False, edgecolor="black", linewidth=2))
ax1.add_patch(Rectangle((-wheel_radius, axle_length / 2 - 0.004), 2 * wheel_radius, 0.008,
                        facecolor="gray", edgecolor="black"))
ax1.add_patch(Rectangle((-wheel_radius, -axle_length / 2 - 0.004), 2 * wheel_radius, 0.008,
                        facecolor="gray", edgecolor="black"))
ax1.annotate("", xy=(0.075, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
ax1.text(0.04, 0.006, "v (heading, angle = theta)", color="red", fontsize=9)
ax1.annotate("", xy=(0.048, axle_length / 2), xytext=(0.048, -axle_length / 2),
             arrowprops=dict(arrowstyle="<->", color="blue"))
ax1.text(0.052, -0.004, "L = 0.052 m", color="blue", fontsize=9)
ax1.text(-0.07, axle_length / 2 + 0.006, "left wheel (w_L)", fontsize=9)
ax1.text(-0.07, -axle_length / 2 - 0.014, "right wheel (w_R), radius r = 0.0205 m", fontsize=9)
ax1.text(-0.075, -0.085, r"$\dot{x} = v\cos\theta$", fontsize=11)
ax1.text(-0.075, -0.1, r"$\dot{y} = v\sin\theta$", fontsize=11)
ax1.text(-0.075, -0.115, r"$\dot{\theta} = \omega$", fontsize=11)
ax1.set_xlim(-0.085, 0.12)
ax1.set_ylim(-0.125, 0.075)
ax1.set_aspect("equal")
ax1.set_title("Differential-drive model (top view)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# panel 2: open loop trajectories
ax2.plot(straight_x, straight_y, label="Straight (v = 0.1)")
ax2.plot(circle_x, circle_y, label="Circle (v = 0.1, w = 0.5)")
ax2.plot(s_x, s_y, label="S-curve (w = +0.8 then -0.8)")
ax2.plot(0, 0, "ko", label="Start")
ax2.set_aspect("equal")
ax2.grid(True)
ax2.legend(fontsize=8)
ax2.set_title("Kinematic model: open loop tests")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")

# panel 3: motor lag step response
ax3.plot(step_time, wl_actual, "b-", label="Actual wheel speed")
ax3.plot([0, 1.0], [wl_command, wl_command], "r--", label="Commanded wheel speed")
ax3.plot([t63], [0.632 * wl_command], "ko", label="63% point")
ax3.grid(True)
ax3.legend()
ax3.set_title("Dynamic model: motor lag (assumed time constant 0.1 s)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Wheel speed (rad/s)")

# panel 4: kinematic vs motor lag trajectory
ax4.plot(kin_x, kin_y, "b-", label="Kinematic only")
ax4.plot(dyn_x, dyn_y, "r--", label="With motor lag")
ax4.plot(0, 0, "ko", label="Start")
ax4.set_aspect("equal")
ax4.grid(True)
ax4.legend(fontsize=8)
ax4.set_title("Same commands (v = 0.1, w = 1.0, 5 s)")
ax4.set_xlabel("x (m)")
ax4.set_ylabel("y (m)")

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "A3_kinematic_model.png"), dpi=150)
plt.show()

print("\nDone. Figure saved in:", figures_folder)