import math

# e-puck parameters
wheel_radius = 0.0205           # m
axle_length = 0.052             # m, distance between the two wheels
max_wheel_speed = 6.28          # rad/s, motor limit
dt = 0.032                      # s, same as the webots time step (32 ms)
max_v = wheel_radius * max_wheel_speed      # about 0.129 m/s

# assumed motor lag (time constant), not measured
motor_tau = 0.1                 # s


def wrap_angle(angle):
    # keeps an angle between -pi and pi
    return math.atan2(math.sin(angle), math.cos(angle))


def wheels_to_vw(w_left, w_right):
    # wheel speeds (rad/s) -> forward speed v (m/s) and turn rate w (rad/s)
    v = wheel_radius * (w_right + w_left) / 2
    w = wheel_radius * (w_right - w_left) / axle_length
    return v, w


def vw_to_wheels(v, w):
    # forward speed and turn rate -> wheel speeds (rad/s)
    w_right = (v + w * axle_length / 2) / wheel_radius
    w_left = (v - w * axle_length / 2) / wheel_radius
    return w_left, w_right


def limit_wheels(w_left, w_right):
    # keeps the wheel speeds inside the motor limit
    # if one wheel is too fast both are scaled down so the turning stays the same
    biggest = max(abs(w_left), abs(w_right))
    if biggest > max_wheel_speed:
        scale = max_wheel_speed / biggest
        w_left = w_left * scale
        w_right = w_right * scale
    return w_left, w_right


def kinematic_step(x, y, theta, v, w, step=dt):
    # one euler step of the kinematic model
    x = x + v * math.cos(theta) * step
    y = y + v * math.sin(theta) * step
    theta = wrap_angle(theta + w * step)
    return x, y, theta


def motor_step(w_actual, w_command, step=dt, tau=motor_tau):
    # first order lag: the wheel speed moves towards the command
    return w_actual + (w_command - w_actual) * step / tau


def dynamic_step(state, v_cmd, w_cmd, step=dt):
    # one step of the model with the motor lag and speed limit
    # state = (x, y, theta, left wheel speed, right wheel speed)
    x, y, theta, wl, wr = state
    wl_cmd, wr_cmd = vw_to_wheels(v_cmd, w_cmd)
    wl_cmd, wr_cmd = limit_wheels(wl_cmd, wr_cmd)
    wl = motor_step(wl, wl_cmd, step)
    wr = motor_step(wr, wr_cmd, step)
    v, w = wheels_to_vw(wl, wr)
    x, y, theta = kinematic_step(x, y, theta, v, w, step)
    return (x, y, theta, wl, wr)