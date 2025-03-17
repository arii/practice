# robot_plot.py
# visualize simple dynamics
from __future__ import annotations
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use("TkAgg")


@dataclass
class Configuration:
    q1: float
    q2: float

    def as_array(self):
        return np.array([self.q1, self.q2])

    def __str__(self):
        return f"({self.q1:.2f},{self.q2:.2f})"


@dataclass
class State:
    x: float
    y: float

    def __str__(self):
        return f"({self.x:.2f},{self.y:.2f})"

    def as_array(self):
        return np.array([self.x, self.y])


@dataclass
class LinkLimit:
    min_angle: float
    max_angle: float

    def in_range(self, angle: float) -> bool:
        return self.min_angle < angle and angle < self.max_angle


class TwoLinkRevoluteManipulator:
    def __init__(self, l1: float, l2: float):
        self.l1 = l1
        self.l2 = l2
        self.l1_limit = LinkLimit(0, np.pi)
        self.l2_limit = LinkLimit(-np.pi, np.pi)
        self._axes = None

    def validate_link_limits(self, q: Configuration) -> bool:
        return self.l1_limit.in_range(
            q.q1) and self.l2_limit.in_range(q.q2)

    def validate_taskspace(self, q: Configuration) -> bool:
        state = self.forward_kinematics(q)
        return state.y > 0

    def validate_configuration(self, q: Configuration) -> bool:
        return self.validate_link_limits(q) and self.validate_taskspace(q)

    def length(self):
        return self.l1 + self.l2

    def forward_kinematics(self, q: Configuration) -> State:
        x = self.l1*np.cos(q.q1) + self.l2*np.cos(q.q1 + q.q2)
        y = self.l1*np.sin(q.q1) + self.l2*np.sin(q.q1 + q.q2)
        return State(x, y)

    def inverse_kinematics(self, state: State) -> list[Configuration]:
        results = []
        r = np.linalg.norm(state.as_array())
        if r > self.length():
            return results
        alpha = np.arccos((self.l1**2 + self.l2**2 - r**2)/(2*self.l1*self.l2))
        beta = np.arccos((r**2 + self.l1**2 - self.l2**2)/(2*self.l1*r))
        for sign in [1, -1]:
            th1 = np.arctan2(state.y, state.x) + sign*beta
            th2 = np.pi + sign*alpha

            th1 = np.arctan2(np.sin(th1), np.cos(th1))
            th2 = np.arctan2(np.sin(th2), np.cos(th2))
            results.append(Configuration(th1, th2))
        return results

    def axes(self):
        if self._axes is None:
            self.init_plot()
        return self._axes

    def init_plot(self):
        if self._axes is None:
            print("new axes!")
            _, self._axes = plt.subplots()
        lim = (-1.2*self.length(), 1.2*self.length())
        self._axes.set_xlim(lim)
        self._axes.set_ylim(lim)

    def plot_configuration(self, q):
        is_valid = self.validate_configuration(q)

        def plot_link(s1, s2):
            x = [s1.x, s2.x]
            y = [s1.y, s2.y]
            color = "grey" if is_valid else "red"
            alpha = 0.5 if is_valid else 0.05
            self.axes().plot(x, y, marker='o', color=color,
                             linewidth=2, alpha=alpha, markersize=3)

        origin = State(0, 0)
        midpoint = State(self.l1*np.cos(q.q1), self.l1*np.sin(q.q1))
        end_effector = self.forward_kinematics(q)
        plot_link(origin, midpoint)
        plot_link(midpoint, end_effector)
        self.axes().scatter(origin.x, origin.y, color='black')
        if is_valid:
            self.axes().scatter(end_effector.x, end_effector.y, color='grey',
                                s=4)
        self.axes().set_title(
            f"Q({q}) -> State({origin}, {midpoint}, {end_effector})")


def plot_solutions_along_x(r: TwoLinkRevoluteManipulator):
    x = r.l1 + r.l2 * np.random.rand()
    for y in np.linspace(-r.length(), r.length(), num=30):
        # r.axes().scatter(x, y)
        for q in r.inverse_kinematics(State(x, y)):
            r.plot_configuration(q)

def plot_random_configurations(r: TwoLinkRevoluteManipulator):
    for q1 in np.arange(0, np.pi*2, step=np.pi/4):
        for q2 in np.arange(-np.pi, np.pi, step=np.pi/10):
            q = Configuration(q1, q2)
            r.plot_configuration(q)
            # input("next")


def main():
    plt.ion()
    r = TwoLinkRevoluteManipulator(0.35, 0.25)
    #plot_random_configurations(r)
    plot_solutions_along_x(r)

main()