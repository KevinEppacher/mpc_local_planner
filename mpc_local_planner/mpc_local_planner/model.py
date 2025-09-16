#!/usr/bin/env python3
import torch

class Model:
    """Holds the motion/kinematics model used by the MPC."""
    def __init__(self, controller):
        self.c = controller
        controller.node.get_logger().info("Model initialized (Differential Drive).")

    def kinematics_step(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Unicycle model:
          x_{k+1} = x_k + T * v * cos(theta)
          y_{k+1} = y_k + T * v * sin(theta)
          th_{k+1}= th_k + T * w
        """
        x, y, th = state
        v, w = u
        T = self.c.T
        return torch.stack([
            x + T * v * torch.cos(th),
            y + T * v * torch.sin(th),
            th + T * w
        ])
