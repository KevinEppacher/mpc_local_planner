#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import torch

Tensor = torch.Tensor

@dataclass
class Pose2D:
    """Robot configuration/state in the plane."""
    x: Tensor   # shape: []
    y: Tensor   # shape: []
    theta: Tensor  # shape: []

    @staticmethod
    def from_xytheta(x: float, y: float, theta: float) -> "Pose2D":
        return Pose2D(torch.tensor(x, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32),
                      torch.tensor(theta, dtype=torch.float32))

    def as_vector(self) -> Tensor:
        return torch.stack([self.x, self.y, self.theta])

    @staticmethod
    def from_vector(v: Tensor) -> "Pose2D":
        return Pose2D(v[0], v[1], v[2])


@dataclass
class Twist2D:
    """Control input for a unicycle/diff-drive robot."""
    v: Tensor       # linear velocity
    omega: Tensor   # angular velocity

    def as_vector(self) -> Tensor:
        return torch.stack([self.v, self.omega])

    @staticmethod
    def from_vector(u: Tensor) -> "Twist2D":
        return Twist2D(u[0], u[1])


class Robot:
    """
    Thin robot wrapper around the motion model and footprint params.
    Keeps dynamics in one place so MPC/optimizer stay clean.
    """
    def __init__(self, dt: float, v_limits: tuple[float, float],
                 w_limits: tuple[float, float],
                 footprint_radius: float, safety_margin: float):
        self.dt = float(dt)
        self.v_min, self.v_max = v_limits
        self.w_min, self.w_max = w_limits
        self.footprint_radius = float(footprint_radius)
        self.safety_margin = float(safety_margin)

    @staticmethod
    def _wrap_angle(a: Tensor) -> Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    def clamp(self, u: Twist2D, v_min=None, v_max=None, w_min=None, w_max=None) -> Twist2D:
        vmin = self.v_min if v_min is None else float(v_min)
        vmax = self.v_max if v_max is None else float(v_max)
        wmin = self.w_min if w_min is None else float(w_min)
        wmax = self.w_max if w_max is None else float(w_max)
        v = torch.clamp(u.v, min=vmin, max=vmax)
        w = torch.clamp(u.omega, min=wmin, max=wmax)
        return Twist2D(v, w)

    def step(self, x: Pose2D, u: Twist2D) -> Pose2D:
        """Unicycle forward Euler step."""
        v, w = u.v, u.omega
        T = self.dt
        xn = x.x + T * v * torch.cos(x.theta)
        yn = x.y + T * v * torch.sin(x.theta)
        thn = self._wrap_angle(x.theta + T * w)
        return Pose2D(xn, yn, thn)
