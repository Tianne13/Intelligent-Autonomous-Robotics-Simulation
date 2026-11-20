import math
import numpy as np

from robot_model import dt, wrap_angle, dynamic_step, wheels_to_vw


class PID:
    def __init__(self, kp, ki, kd, out_limit=None, int_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit          # largest output allowed
        self.int_limit = int_limit          # largest value of the integral (anti-windup)
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, step=dt):
        # integral term, clamped so it cannot wind up
        self.integral = self.integral + error * step
        if self.int_limit is not None:
            self.integral = max(-self.int_limit, min(self.int_limit, self.integral))

        # derivative term (0 on the first call)
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / step
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_limit is not None:
            output = max(-self.out_limit, min(self.out_limit, output))
        return output


def make_dense_path(points, spacing=0.01):
    # adds a point every 1 cm along the straight lines between the waypoints
    dense = [points[0]]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(length / spacing))
        for k in range(1, n + 1):
            dense.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    return np.array(dense)


def follow_path(dense_path, gains, start_pose, v_nom=0.08, lookahead=0.12,
                goal_tol=0.03, max_time=60.0, w_limit=2.0):
    # drives the model along the path with a PID on the heading error
    # returns a dictionary with everything that was logged
    kp, ki, kd = gains
    pid = PID(kp, ki, kd, out_limit=w_limit, int_limit=1.0)

    x, y, theta = start_pose
    state = (x, y, theta, 0.0, 0.0)
    goal_x, goal_y = dense_path[-1]
    closest = 0

    log = {"t": [], "x": [], "y": [], "theta": [], "heading_error": [],
           "cross_track": [], "v": [], "w": []}
    t = 0.0
    reached = False

    while t < max_time:
        x, y, theta = state[0], state[1], state[2]

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

        # target point a bit further along the path (lookahead)
        target = closest
        while target < len(dense_path) - 1:
            if math.hypot(dense_path[target][0] - x, dense_path[target][1] - y) >= lookahead:
                break
            target = target + 1
        target_x, target_y = dense_path[target]

        # heading error and the PID gives the turn rate
        desired_heading = math.atan2(target_y - y, target_x - x)
        heading_error = wrap_angle(desired_heading - theta)
        w_cmd = pid.update(heading_error)

        # forward speed: slower when the robot points the wrong way and near the goal
        v_cmd = v_nom * max(0.2, math.cos(min(abs(heading_error), math.pi / 2)))
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        v_cmd = min(v_cmd, max(0.02, dist_to_goal))

        # log
        v_actual, w_actual = wheels_to_vw(state[3], state[4])
        log["t"].append(t)
        log["x"].append(x)
        log["y"].append(y)
        log["theta"].append(theta)
        log["heading_error"].append(heading_error)
        log["cross_track"].append(cross_track)
        log["v"].append(v_actual)
        log["w"].append(w_actual)

        if dist_to_goal < goal_tol:
            reached = True
            break

        state = dynamic_step(state, v_cmd, w_cmd)
        t = t + dt

    log["reached"] = reached
    log["time"] = t
    return log