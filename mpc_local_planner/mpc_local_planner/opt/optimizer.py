#!/usr/bin/env python3
import time
import torch
from mpc_local_planner.core.state import Pose2D, Twist2D
from mpc_local_planner.costs.obstacle_penalty import ObstaclePenalty

class Optimizer:
    def __init__(self, N: int, robot, Q, R, S, T: float,
                 penalty: ObstaclePenalty,
                 iters: int = 8, time_frac: float = 0.5):
        self.N = N
        self.robot = robot
        self.Q, self.R, self.S = Q, R, S
        self.T = float(T)
        self.penalty = penalty
        self.max_iters = int(iters)
        self.time_frac = float(time_frac)
        self.U = None  # [N, 2]
        torch.set_num_threads(1)

    @staticmethod
    def wrap_angle(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    def init_warm_start(self):
        self.U = torch.zeros(self.N, 2, dtype=torch.float32, requires_grad=True)

    def rollout_cost(self, x0: Pose2D, ref_traj: torch.Tensor):
        """Returns (scalar cost, stacked states [N+1,3])."""
        x = x0
        J = torch.tensor(0.0, dtype=torch.float32)
        Xs = [x.as_vector().clone()]

        for k in range(self.N):
            u = Twist2D.from_vector(self.U[k])
            u = self.robot.clamp(u)

            x = self.robot.step(x, u)
            Xs.append(x.as_vector().clone())

            # tracking error in R^3 with yaw wrapping
            r = ref_traj[k]
            lin = x.as_vector()[:2] - r[:2]
            yaw_err = self.wrap_angle(x.theta - r[2])
            diff = torch.stack([lin[0], lin[1], yaw_err])

            J = J + diff @ self.Q @ diff + u.as_vector() @ self.R @ u.as_vector() \
                  + self.penalty.evaluate(x)

        # terminal cost
        rT = ref_traj[-1]
        linT = x.as_vector()[:2] - rT[:2]
        yawT = self.wrap_angle(x.theta - rT[2])
        diffT = torch.stack([linT[0], linT[1], yawT])
        J = J + diffT @ self.S @ diffT

        return J, torch.stack(Xs, dim=0)

    def solve(self, x0_vec: torch.Tensor, ref_traj: torch.Tensor):
        if self.U is None or self.U.shape[0] != self.N:
            self.init_warm_start()

        x0 = Pose2D.from_vector(x0_vec)
        optim = torch.optim.Adam([self.U], lr=0.2)
        time_budget = max(1e-3, self.time_frac * self.T)

        t0 = time.perf_counter()
        for _ in range(self.max_iters):
            optim.zero_grad(set_to_none=True)
            J, _ = self.rollout_cost(x0, ref_traj)
            J.backward()
            torch.nn.utils.clip_grad_norm_([self.U], max_norm=1.0)
            optim.step()
            if (time.perf_counter() - t0) > time_budget:
                break

        with torch.no_grad():
            _, X_pred = self.rollout_cost(x0, ref_traj)
            # ensure output respects limits
            u0_raw = Twist2D.from_vector(self.U[0].detach())
            u0 = self.robot.clamp(u0_raw)            # <— clamp here

            # warm start shift
            src = self.U[1:].detach().clone()
            self.U[:-1].copy_(src)
            self.U[-1].zero_()
        return u0, X_pred