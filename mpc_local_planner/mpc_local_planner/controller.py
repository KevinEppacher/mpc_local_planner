#!/usr/bin/env python3
import math
import torch
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped, PoseArray

from mpc_local_planner.utils import yaw_from_quat
from mpc_local_planner.core.state import Pose2D, Robot
from mpc_local_planner.core.costmap import CostmapGrid
from mpc_local_planner.costs.obstacle_penalty import ObstaclePenalty, ObstacleParams
from mpc_local_planner.opt.optimizer import Optimizer


class MPCController:
    """
    MPC wiring:
      - owns Robot (limits, dt)
      - owns ObstaclePenalty (costmap + params)
      - owns Optimizer (Q,R,S, horizon)
    Keeps all ROS parameters in one place and updates them via a callback.
    """

    def __init__(self, node: Node, costmap: CostmapGrid):
        self.node = node
        p = self.node

        # --- declare parameters (defaults) ---
        def decl(name, default):
            if not p.has_parameter(name):
                p.declare_parameter(name, default)

        # controller core
        decl("controller.N", 12)
        decl("controller.T", 0.2)
        decl("controller.Q", [10.0, 10.0, 0.25])
        decl("controller.R", [0.20, 0.05])
        decl("controller.S", [2.0, 2.0, 0.50])
        decl("controller.v_min", -0.10)
        decl("controller.v_max",  0.40)
        decl("controller.omega_min", -math.pi/4)
        decl("controller.omega_max",  math.pi/4)

        # goal behavior
        decl("goal.stop_pos_tol", 0.12)     # [m]
        decl("goal.stop_yaw_deg", 15.0)     # [deg]
        decl("goal.slow_radius", 1.0)       # [m]
        decl("goal.slow_exp", 1.0)
        decl("goal.terminal_boost", 5.0)

        # obstacle/costmap
        decl("robot.footprint_radius", 0.18)
        decl("obstacle.safety_margin", 0.05)
        decl("costmap.weight", 300.0)
        decl("costmap.power",  2.0)
        decl("costmap.thresh", 0.20)
        decl("costmap.unknown_is_lethal", True)
        decl("costmap.ahead_distances", [0.0, 0.20, 0.40])
        decl("costmap.samples_per_circle", 8)

        # solver
        decl("solver.max_iters", 8)
        decl("solver.time_frac", 0.5)

        # --- read parameters into attributes ---
        self.N  = int(p.get_parameter("controller.N").value)
        self.T  = float(p.get_parameter("controller.T").value)
        self.Qv = list(p.get_parameter("controller.Q").value)
        self.Rv = list(p.get_parameter("controller.R").value)
        self.Sv = list(p.get_parameter("controller.S").value)

        self.v_min = float(p.get_parameter("controller.v_min").value)
        self.v_max = float(p.get_parameter("controller.v_max").value)
        self.w_min = float(p.get_parameter("controller.omega_min").value)
        self.w_max = float(p.get_parameter("controller.omega_max").value)

        self.stop_pos_tol   = float(p.get_parameter("goal.stop_pos_tol").value)
        self.stop_yaw_tol   = math.radians(float(p.get_parameter("goal.stop_yaw_deg").value))
        self.slow_radius    = float(p.get_parameter("goal.slow_radius").value)
        self.slow_exp       = float(p.get_parameter("goal.slow_exp").value)
        self.terminal_boost = float(p.get_parameter("goal.terminal_boost").value)

        self.fp_radius   = float(p.get_parameter("robot.footprint_radius").value)
        self.safe_margin = float(p.get_parameter("obstacle.safety_margin").value)

        self.cm_weight = float(p.get_parameter("costmap.weight").value)
        self.cm_power  = float(p.get_parameter("costmap.power").value)
        self.cm_thresh = float(p.get_parameter("costmap.thresh").value)
        self.cm_unknown_is_lethal = bool(p.get_parameter("costmap.unknown_is_lethal").value)
        self.cm_ahead = list(p.get_parameter("costmap.ahead_distances").value)
        self.cm_samples = int(p.get_parameter("costmap.samples_per_circle").value)

        self.solver_iters = int(p.get_parameter("solver.max_iters").value)
        self.solver_tfrac = float(p.get_parameter("solver.time_frac").value)

        # --- matrices ---
        self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
        self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
        self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))

        # --- submodules ---
        self.robot = Robot(
            dt=self.T,
            v_limits=(self.v_min, self.v_max),
            w_limits=(self.w_min, self.w_max),
            footprint_radius=self.fp_radius,
            safety_margin=self.safe_margin,
        )

        self.penalty = ObstaclePenalty(
            costmap,
            ObstacleParams(
                weight=self.cm_weight,
                power=self.cm_power,
                thresh=self.cm_thresh,
                unknown_is_lethal=self.cm_unknown_is_lethal,
                ahead_dists=self.cm_ahead,
                samples_per_circle=self.cm_samples,
                footprint_radius=self.fp_radius,
                safety_margin=self.safe_margin,
            ),
        )

        self.optimizer = Optimizer(
            N=self.N, robot=self.robot, Q=self.Q, R=self.R, S=self.S, T=self.T,
            penalty=self.penalty, iters=self.solver_iters, time_frac=self.solver_tfrac,
        )

        self.last_obst_dbg = None
        p.add_on_set_parameters_callback(self._on_param_update)

    # ---- main solve ----
    def solve_mpc(self, current: PoseStamped, ref_traj: PoseArray, T: float):
        # current state
        x = float(current.pose.position.x)
        y = float(current.pose.position.y)
        yaw = yaw_from_quat(current.pose.orientation)
        x0_vec = torch.tensor([x, y, yaw], dtype=torch.float32)

        # reference tensor [N,3]
        N = self.N
        refs = []
        for i in range(min(N, len(ref_traj.poses))):
            pr = ref_traj.poses[i]
            refs.append(torch.tensor(
                [float(pr.position.x), float(pr.position.y), yaw_from_quat(pr.orientation)],
                dtype=torch.float32
            ))
        if not refs:
            refs.append(torch.tensor([x, y, yaw], dtype=torch.float32))
        while len(refs) < N:
            refs.append(refs[-1])
        ref_t = torch.stack(refs, dim=0)

        # goal metrics
        gx, gy, gyaw = float(ref_t[-1, 0]), float(ref_t[-1, 1]), float(ref_t[-1, 2])
        d_goal = math.hypot(gx - x, gy - y)
        yaw_err = math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))

        # hard stop
        if (d_goal <= self.stop_pos_tol) and (abs(yaw_err) <= self.stop_yaw_tol):
            X_pred = torch.zeros(N + 1, 3, dtype=torch.float32)
            X_pred[0, :] = x0_vec
            for k in range(1, N + 1):
                X_pred[k, :] = X_pred[k - 1, :]
            return 0.0, 0.0, X_pred

        # dynamic limits near goal + terminal boost
        if self.slow_radius > 1e-6:
            scale = max(0.0, min(1.0, d_goal / self.slow_radius))
            scale = pow(scale, self.slow_exp)
            self.robot.v_min = self.v_min
            self.robot.v_max = self.v_max * scale
            self.robot.w_min = self.w_min
            self.robot.w_max = self.w_max * scale
            self.optimizer.S = self.S * (1.0 + self.terminal_boost * (1.0 - scale))
        else:
            self.robot.v_min = self.v_min
            self.robot.v_max = self.v_max
            self.robot.w_min = self.w_min
            self.robot.w_max = self.w_max
            self.optimizer.S = self.S

        # solve
        u0, X_pred = self.optimizer.solve(x0_vec, ref_t)
        u0 = self.robot.clamp(u0)  # final clamp
        self.last_obst_dbg = self.penalty.debug
        return float(u0.v.item()), float(u0.omega.item()), X_pred

    # ---- param callback ----
    def _on_param_update(self, params):
        for param in params:
            n = param.name
            v = param.value

            # controller core
            if n == "controller.N":
                self.N = int(v)
                self.optimizer.N = self.N
                self.optimizer.init_warm_start()
            elif n == "controller.T":
                self.T = float(v)
                self.robot.dt = self.T
            elif n == "controller.Q":
                self.Qv = list(v); self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
                self.optimizer.Q = self.Q
            elif n == "controller.R":
                self.Rv = list(v); self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
                self.optimizer.R = self.R
            elif n == "controller.S":
                self.Sv = list(v); self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))
                self.optimizer.S = self.S

            elif n == "controller.v_min":
                self.v_min = float(v); self.robot.v_min = self.v_min
            elif n == "controller.v_max":
                self.v_max = float(v); self.robot.v_max = self.v_max
            elif n == "controller.omega_min":
                self.w_min = float(v); self.robot.w_min = self.w_min
            elif n == "controller.omega_max":
                self.w_max = float(v); self.robot.w_max = self.w_max

            # goal behavior
            elif n == "goal.stop_pos_tol":
                self.stop_pos_tol = float(v)
            elif n == "goal.stop_yaw_deg":
                self.stop_yaw_tol = math.radians(float(v))
            elif n == "goal.slow_radius":
                self.slow_radius = float(v)
            elif n == "goal.slow_exp":
                self.slow_exp = float(v)
            elif n == "goal.terminal_boost":
                self.terminal_boost = float(v)

            # costmap/obstacle
            elif n == "robot.footprint_radius":
                self.fp_radius = float(v)
                self.robot.footprint_radius = self.fp_radius
                self.penalty.p.footprint_radius = self.fp_radius
            elif n == "obstacle.safety_margin":
                self.safe_margin = float(v)
                self.robot.safety_margin = self.safe_margin
                self.penalty.p.safety_margin = self.safe_margin
            elif n == "costmap.weight":
                self.cm_weight = float(v); self.penalty.p.weight = self.cm_weight
            elif n == "costmap.power":
                self.cm_power = float(v); self.penalty.p.power = self.cm_power
            elif n == "costmap.thresh":
                self.cm_thresh = float(v); self.penalty.p.thresh = self.cm_thresh
            elif n == "costmap.unknown_is_lethal":
                self.cm_unknown_is_lethal = bool(v); self.penalty.p.unknown_is_lethal = self.cm_unknown_is_lethal
            elif n == "costmap.ahead_distances":
                self.cm_ahead = list(v); self.penalty.p.ahead_dists = self.cm_ahead
            elif n == "costmap.samples_per_circle":
                self.cm_samples = int(v); self.penalty.p.samples_per_circle = self.cm_samples

            # solver
            elif n == "solver.max_iters":
                self.solver_iters = int(v); self.optimizer.max_iters = self.solver_iters
            elif n == "solver.time_frac":
                self.solver_tfrac = float(v); self.optimizer.time_frac = self.solver_tfrac

        return SetParametersResult(successful=True)
