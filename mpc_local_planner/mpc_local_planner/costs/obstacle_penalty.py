#!/usr/bin/env python3
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
from mpc_local_planner.core.state import Pose2D
from mpc_local_planner.core.costmap import CostmapGrid

Tensor = torch.Tensor

@dataclass
class ObstacleParams:
    weight: float
    power: float
    thresh: float
    unknown_is_lethal: bool
    ahead_dists: List[float]
    samples_per_circle: int
    footprint_radius: float
    safety_margin: float

class ObstaclePenalty:
    """
    Readable, stepwise obstacle penalty:
      evaluate() -> sample -> aggregate -> soften -> scale -> debug
    """
    def __init__(self, costmap: CostmapGrid, p: ObstacleParams):
        self.cm = costmap
        self.p = p
        self.debug: Dict[str, Any] | None = None

    @staticmethod
    def _rot(c: float, s: float, ox: float, oy: float) -> tuple[float, float]:
        # Rotate (ox,oy) by heading with cos=c, sin=s
        return c*ox - s*oy, s*ox + c*oy

    def _ring_centers(self, x: Pose2D) -> List[tuple[Tensor, Tensor]]:
        """1) Define centers along the robot's look-ahead direction."""
        c, s = torch.cos(x.theta), torch.sin(x.theta)
        centers = []
        for d in (self.p.ahead_dists or [0.0]):
            cx = x.x + c * float(d)
            cy = x.y + s * float(d)
            centers.append((cx, cy))
        return centers

    def _sample_circle(self, cx: Tensor, cy: Tensor, c: Tensor, s: Tensor
                      ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Return list of (wx, wy, cval) as TENSORS."""
        r = float(self.p.footprint_radius + self.p.safety_margin)
        M = max(1, int(self.p.samples_per_circle))
        out: List[Tuple[Tensor, Tensor, Tensor]] = []

        if r <= 1e-6 or M == 1:
            cval = self.cm.sample(cx, cy)              # Tensor
            out.append((cx, cy, cval))
            return out

        for j in range(M):
            phi = 2.0 * math.pi * j / M
            ox, oy = r * math.cos(phi), r * math.sin(phi)
            rx = c * ox - s * oy                      # Tensor
            ry = s * ox + c * oy
            wx = cx + rx
            wy = cy + ry
            cval = self.cm.sample(wx, wy)             # Tensor
            out.append((wx, wy, cval))
        return out

    def _gather_samples(self, x: Pose2D) -> List[Tuple[Tensor, Tensor, Tensor]]:
        samples: List[Tuple[Tensor, Tensor, Tensor]] = []
        c, s = torch.cos(x.theta), torch.sin(x.theta)
        for cx, cy in self._ring_centers(x):
            samples.extend(self._sample_circle(cx, cy, c, s))
        return samples

    @staticmethod
    def _smooth_max(values: Tensor, beta: float = 8.0) -> Tensor:
        """4) Smooth maximum (log-sum-exp) in [0,1]."""
        m = torch.max(values)
        out = m + torch.log(torch.clamp(torch.exp((values - m) * beta).mean(), min=1e-9)) / beta
        return torch.clamp(out, 0.0, 1.0)

    def _soft_threshold(self, v: Tensor) -> Tensor:
        """5) Map [0,1] to [0,1] using threshold p.thresh; unknown lethal if set."""
        if self.p.unknown_is_lethal:
            v = torch.where(v >= 0.992, torch.tensor(1.0, dtype=torch.float32), v)
        t = float(self.p.thresh)
        return torch.clamp((v - t) / (1.0 - t + 1e-6), 0.0, 1.0)

    def _scale(self, s: Tensor) -> Tensor:
        """6) Raise and scale."""
        return (s ** float(self.p.power)) * float(self.p.weight)

    def evaluate(self, x: Pose2D) -> Tensor:
        if not self.cm or not self.cm.ready:
            self.debug = None
            return torch.tensor(0.0, dtype=torch.float32)

        samples = self._gather_samples(x)                           # tensors
        if not samples:
            self.debug = None
            return torch.tensor(0.0, dtype=torch.float32)

        vals = torch.stack([c for (_, _, c) in samples], dim=0)     # keep graph
        v = self._smooth_max(vals)                                  # tensors
        s = self._soft_threshold(v)
        penalty = self._scale(s)

        # Debug nur als floats bauen
        dbg_samples = [(float(wx.item()), float(wy.item()), float(c.item()))
                       for (wx, wy, c) in samples]
        self.debug = {
            "state": (float(x.x.item()), float(x.y.item()), float(x.theta.item())),
            "radius": float(self.p.footprint_radius + self.p.safety_margin),
            "lookahead": list(map(float, self.p.ahead_dists or [0.0])),
            "samples": dbg_samples,
            "smooth_max": float(v.item()),
            "soft": float(s.item()),
            "penalty": float(penalty.item()),
        }
        return penalty

