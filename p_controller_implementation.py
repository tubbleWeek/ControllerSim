"""
Modified by William Chastek to model a f1tenth robotic system. Trying to implement the p control use in https://docs.google.com/presentation/d/1jpnlQ7ysygTPCi8dmyZjooqzxNXWqMgO31ZhcOlKVOE/edit#slide=id.g63d5f5680f_0_0

Based on:
Path tracking simulation with pure pursuit steering and PID speed control witten by Atsushi Sakai (@Atsushi_twi) and Guillaume Jacquenot (@Gjacquenot)

usage:
python pure_pursuit_shepherd.py

Notes:
the shepherd_lab_traj.csv must be in the same directory otherwise you will have to modify the code

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# Parameters
k = 0.5  # look forward gain
Lfc = 1.0  # [m] look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 0.33  # [m] wheel base of vehicle

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[(ind + 1)%len(self.cx)],
                                                          self.cy[(ind + 1)%len(self.cy)])
                if distance_this_index < distance_next_index:
                    break
                # ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                ind = (ind + 1) % len(self.cx)
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)
    # Based on https://mecharithm.com/learning/lesson/homogenous-transformation-matrices-configurations-in-robotics-12
    theta = state.yaw
    gloabl_tx = trajectory.cx[ind] # This is the target waypoints x position
    global_ty = trajectory.cy[ind] # This is the target waypoints y position

    transformation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), state.x],
        [np.sin(theta), np.cos(theta), state.y],
        [0, 0, 1],
    ])

    trans_inv = np.linalg.inv(transformation_matrix)
    look_ahead_point_global = np.array([gloabl_tx, global_ty, 1])
    
    look_ahead_point_robot = trans_inv @ look_ahead_point_global

    # print(look_ahead_point_robot)

    r = pow(Lf, 2) / (2*abs(look_ahead_point_robot[1]))

    delta = k * 2 * look_ahead_point_robot[1] / pow(r, 2) # Angle calculated using CL2 Waterloo p-controllerequation https://github.com/CL2-UWaterloo/f1tenth_ws/blob/main/src/pure_pursuit/src/pure_pursuit.cpp
    return delta, ind

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    #  target course
    data = np.loadtxt("./shepherd_lab_traj.csv", delimiter=",")
    cx = data[:, 0]
    cy = data[:, 1]
    cv = data[:, 2]

    # cx = np.arange(0, 50, 0.5)
    # cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 20.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state ,
    state = State(x=-2.5, y=-12.5, yaw=1.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time:
        if lastIndex == target_ind:
            target_ind = 0
        target_speed = cv[target_ind]
        # Calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(
            state, target_course, target_ind)

        state.update(ai, di)  # Control vehicle

        time += dt
        states.append(time, state)
        

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    # assert lastIndex >= target_ind, "Cannot goal"
    

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    try:
        print("Pure pursuit path tracking simulation start")
        main()
    except KeyboardInterrupt:
        sys.exit()
    # Code to handle the KeyboardInterrupt
    
