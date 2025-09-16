#!/usr/bin/env python3
import numpy as np
import torch
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

class CostmapGrid:
    """
    Stores OccupancyGrid and exposes bilinear sampling in world coords.
    Input data 0..255 -> normalized to [0,1]. Unknown/lethal >=253 -> 1.0.
    """
    def __init__(self, node: Node):
        self.node = node
        self.ready = False
        self.res = None; self.ox = None; self.oy = None
        self.W = None; self.H = None
        self.frame_id = None
        self.grid = None  # torch.FloatTensor [H, W] in [0,1]

    def cb(self, msg: OccupancyGrid):
        info = msg.info
        self.res = float(info.resolution)
        self.W = int(info.width); self.H = int(info.height)
        self.ox = float(info.origin.position.x); self.oy = float(info.origin.position.y)
        self.frame_id = msg.header.frame_id

        data = np.asarray(msg.data, dtype=np.int16).reshape(self.H, self.W)
        grid = np.clip(data, 0, 255).astype(np.float32)
        grid[grid >= 253] = 255.0
        self.grid = torch.from_numpy(grid / 255.0)
        self.ready = True

    def sample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear sample at world (x,y). Returns scalar torch tensor in [0,1]."""
        if not self.ready or self.grid is None:
            return torch.tensor(0.0, dtype=torch.float32)

        gx = (x - self.ox) / (self.res + 1e-9)
        gy = (y - self.oy) / (self.res + 1e-9)

        x0 = torch.floor(gx); y0 = torch.floor(gy)
        x1 = x0 + 1.0;        y1 = y0 + 1.0

        x0c = torch.clamp(x0, 0.0, self.W - 1.0); y0c = torch.clamp(y0, 0.0, self.H - 1.0)
        x1c = torch.clamp(x1, 0.0, self.W - 1.0); y1c = torch.clamp(y1, 0.0, self.H - 1.0)

        wx = torch.clamp(gx - x0, 0.0, 1.0)
        wy = torch.clamp(gy - y0, 0.0, 1.0)

        G = self.grid  # [H, W]
        def at(ix, iy): return G[iy.long(), ix.long()]

        c00 = at(x0c, y0c); c10 = at(x1c, y0c)
        c01 = at(x0c, y1c); c11 = at(x1c, y1c)
        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        return c0 * (1.0 - wy) + c1 * wy
