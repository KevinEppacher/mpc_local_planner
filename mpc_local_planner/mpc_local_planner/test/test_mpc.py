#!/usr/bin/env python3
import math
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TwistStamped, PoseStamped, PoseArray, Pose

# TF2
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
import tf_transformations


# -----------------------------
# Small utilities
# -----------------------------
def yaw_from_quat(q) -> float:
    """Return yaw [rad] from geometry_msgs/Quaternion."""
    return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

def quat_from_yaw(yaw: float):
    """Return (x,y,z,w) quaternion for given yaw [rad]."""
    return tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)

def make_pose(x: float, y: float, yaw: float) -> Pose:
    """Create geometry_msgs/Pose from x,y,yaw."""
    qx, qy, qz, qw = quat_from_yaw(yaw)
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.orientation.x = float(qx)
    p.orientation.y = float(qy)
    p.orientation.z = float(qz)
    p.orientation.w = float(qw)
    return p

# =========================
# Costmap holder with bilinear sampling
# =========================
class CostmapGrid:
    """
    Holds an OccupancyGrid and provides bilinear sampling in world coords.
    Input is 0..255; mapped to [0,1]. Lethal/unknown (>=253) → ~1.0.
    """
    def __init__(self, node: Node):
        self.node = node
        self.ready = False
        self.res = None
        self.ox = None
        self.oy = None
        self.W = None
        self.H = None
        self.frame_id = None
        self.grid = None  # torch.FloatTensor [H, W] in [0,1]

    def cb(self, msg: OccupancyGrid):
        info = msg.info
        self.res = float(info.resolution)
        self.W = int(info.width)
        self.H = int(info.height)
        self.ox = float(info.origin.position.x)
        self.oy = float(info.origin.position.y)
        self.frame_id = msg.header.frame_id

        data = np.asarray(msg.data, dtype=np.int16).reshape(self.H, self.W)
        grid = np.clip(data, 0, 255).astype(np.float32)
        grid[grid >= 253] = 255.0  # lethal / unknown
        grid = grid / 255.0
        self.grid = torch.from_numpy(grid)  # CPU tensor
        self.ready = True

    def sample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear sample at world (x,y). Returns scalar torch tensor in [0,1]."""
        if not self.ready or self.grid is None:
            return torch.tensor(0.0, dtype=torch.float32)

        gx = (x - self.ox) / (self.res + 1e-9)
        gy = (y - self.oy) / (self.res + 1e-9)

        x0 = torch.floor(gx); y0 = torch.floor(gy)
        x1 = x0 + 1.0;       y1 = y0 + 1.0

        # clamp indices into the map
        x0c = torch.clamp(x0, 0.0, self.W - 1.0); y0c = torch.clamp(y0, 0.0, self.H - 1.0)
        x1c = torch.clamp(x1, 0.0, self.W - 1.0); y1c = torch.clamp(y1, 0.0, self.H - 1.0)

        wx = torch.clamp(gx - x0, 0.0, 1.0)
        wy = torch.clamp(gy - y0, 0.0, 1.0)

        G = self.grid  # [H, W]

        def at(ix, iy):
            return G[iy.long(), ix.long()]

        c00 = at(x0c, y0c); c10 = at(x1c, y0c)
        c01 = at(x0c, y1c); c11 = at(x1c, y1c)

        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        return c0 * (1.0 - wy) + c1 * wy


# =========================
# Local Planner Node
# =========================
class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner_node")
        self.get_logger().info(f"LocalPlannerNode initialized: {self.get_name()}")

        # Parameters (kept minimal)
        if not self.has_parameter("robot.parent_frame"):
            self.declare_parameter("robot.parent_frame", "map")
        if not self.has_parameter("robot.base_frame"):
            self.declare_parameter("robot.base_frame", "base_link")
        if not self.has_parameter("costmap.topic"):
            self.declare_parameter("costmap.topic", "/local_costmap/costmap")

        self.parent_frame = self.get_parameter("robot.parent_frame").value
        self.base_frame   = self.get_parameter("robot.base_frame").value
        self.costmap_topic = self.get_parameter("costmap.topic").value

        # Submodules
        self.controller = MPCController(self)
        self.planner    = TrajectoryPlanner(self)

        # TF instead of plain /odom
        self.tf_buffer   = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Internal pose cache (PoseStamped)
        self.current_pose = None
        self.T_map_odom = None

        # Publishers
        self.cmd_vel_pub  = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.horizon_pub  = self.create_publisher(PoseArray, "/mpc/horizon", 10)
        self.ref_pub      = self.create_publisher(PoseArray, "/mpc/reference", 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.costmap = CostmapGrid(self)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.costmap.cb, 1
        )

        # Timers
        self.timer    = self.create_timer(self.controller.T, self.controller_timer_callback)
        self.tf_timer = self.create_timer(0.2, self.tf_timer_callback)

        self.get_logger().info(f"Using TF for pose: {self.parent_frame} -> {self.base_frame}")
        self.get_logger().info(f"Subscribed costmap: {self.costmap_topic}")

        # Timing diagnostics
        self._last_cb_wall = time.perf_counter()
        self._ema_ms = 0.0

    # -------- TF helpers --------
    def _odom_to_map_pose(self, ps_odom: PoseStamped) -> PoseStamped:
        """Transform PoseStamped from odom to map using cached T(map<-odom)."""
        # odom->base
        q = [ps_odom.pose.orientation.x, ps_odom.pose.orientation.y,
             ps_odom.pose.orientation.z, ps_odom.pose.orientation.w]
        T_odom_base = tf_transformations.quaternion_matrix(q)
        T_odom_base[0, 3] = ps_odom.pose.position.x
        T_odom_base[1, 3] = ps_odom.pose.position.y
        T_odom_base[2, 3] = ps_odom.pose.position.z

        # map->odom
        t = self.T_map_odom.translation
        q = [self.T_map_odom.rotation.x, self.T_map_odom.rotation.y,
             self.T_map_odom.rotation.z, self.T_map_odom.rotation.w]
        T_map_odom = tf_transformations.quaternion_matrix(q)
        T_map_odom[0, 3] = t.x
        T_map_odom[1, 3] = t.y
        T_map_odom[2, 3] = t.z

        # map->base
        T_map_base = T_map_odom @ T_odom_base

        # back to PoseStamped (map)
        out = PoseStamped()
        out.header.frame_id = "map"
        out.header.stamp = ps_odom.header.stamp
        out.pose.position.x = T_map_base[0, 3]
        out.pose.position.y = T_map_base[1, 3]
        out.pose.position.z = T_map_base[2, 3]
        q_map_base = tf_transformations.quaternion_from_matrix(T_map_base)
        out.pose.orientation.x = q_map_base[0]
        out.pose.orientation.y = q_map_base[1]
        out.pose.orientation.z = q_map_base[2]
        out.pose.orientation.w = q_map_base[3]
        return out

    # -------- Callbacks --------
    def _odom_cb(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        if self.T_map_odom is not None:
            self.current_pose = self._odom_to_map_pose(ps)
        else:
            self.current_pose = ps  # fallback

    def tf_timer_callback(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.parent_frame, "odom", rclpy.time.Time())
            self.T_map_odom = tr.transform
        except Exception as e:
            self.get_logger().debug(f"Transform not available: {e}")

    def controller_timer_callback(self):
        # ---- timing
        start_wall = time.perf_counter()
        period_ms = (start_wall - self._last_cb_wall) * 1000.0
        self._last_cb_wall = start_wall

        # ---- idle (no pose or no goal)
        if self.current_pose is None or self.planner.goal_pose is None:
            self._publish_cmd(0.0, 0.0)
            self._log_timing(start_wall, period_ms, idle=True)
            return

        # ---- reference & MPC
        ref_traj = self.planner.build_reference(self.current_pose, self.controller.N, self.controller.T)
        self.ref_pub.publish(ref_traj)

        v, w, X_pred = self.controller.solve_mpc(self.current_pose, ref_traj, self.controller.T)
        self._publish_cmd(v, w)
        self._publish_horizon(X_pred)

        # ---- timing log
        self._log_timing(start_wall, period_ms)

    # -------- small helpers for publishing & timing --------
    def _publish_cmd(self, v: float, w: float):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.parent_frame
        cmd.twist.linear.x  = float(v)
        cmd.twist.angular.z = float(w)
        self.cmd_vel_pub.publish(cmd)

    def _publish_horizon(self, X_pred: torch.Tensor):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.parent_frame
        for i in range(X_pred.shape[0]):  # N+1 states
            x, y, th = float(X_pred[i, 0]), float(X_pred[i, 1]), float(X_pred[i, 2])
            pa.poses.append(make_pose(x, y, th))
        self.horizon_pub.publish(pa)

    def _log_timing(self, start_wall: float, period_ms: float, idle: bool = False):
        comp_ms   = (time.perf_counter() - start_wall) * 1000.0
        budget_ms = self.controller.T * 1000.0
        alpha = 0.2
        self._ema_ms = (1.0 - alpha) * self._ema_ms + alpha * comp_ms
        tag = "IDLE" if idle else ("WARN" if comp_ms > budget_ms else "OK")
        self.get_logger().info(
            f"[{tag}] loop: compute={comp_ms:.1f} ms, period={period_ms:.1f} ms, "
            f"budget={budget_ms:.0f} ms, ema={self._ema_ms:.1f} ms"
        )


# =========================
# Controller
# =========================
class MPCController:
    def __init__(self, node: Node):
        self.node = node
        p = self.node

        # --- SAFE DECLARE helper ---
        def decl(name, default):
            if not p.has_parameter(name):
                p.declare_parameter(name, default)

        # ---- Core MPC params ----
        decl("controller.N", 12)
        decl("controller.T", 0.2)
        decl("controller.Q", [10.0, 10.0, 0.25])
        decl("controller.R", [0.2, 0.05])
        decl("controller.S", [2.0, 2.0, 0.5])
        decl("controller.v_min", -0.10)
        decl("controller.v_max",  0.40)
        decl("controller.omega_min", -math.pi/4)
        decl("controller.omega_max",  math.pi/4)

        # ---- Goal / stop & slow-down ----
        decl("goal.stop_pos_tol", 0.12)   # [m]
        decl("goal.stop_yaw_deg", 15.0)   # [deg]
        decl("goal.slow_radius", 1.0)     # [m]
        decl("goal.slow_exp", 1.0)
        decl("goal.terminal_boost", 5.0)

        # ---- Costmap penalty ----
        # WICHTIG: 'costmap.topic' NICHT nochmal deklarieren (LocalPlannerNode macht das bereits)!
        decl("costmap.weight", 300.0)
        decl("costmap.power",  2.0)
        decl("costmap.thresh", 0.20)
        decl("costmap.unknown_is_lethal", True)
        decl("robot.footprint_radius", 0.18)
        decl("obstacle.safety_margin", 0.05)
        decl("costmap.ahead_distances", [0.0, 0.20, 0.40])
        decl("costmap.samples_per_circle", 8)

        # ---- Solver (Adam) ----
        decl("solver.max_iters", 8)
        decl("solver.time_frac", 0.5)

        # ---- Read params ----
        self.N  = p.get_parameter("controller.N").value
        self.T  = p.get_parameter("controller.T").value
        self.Qv = list(p.get_parameter("controller.Q").value)
        self.Rv = list(p.get_parameter("controller.R").value)
        self.Sv = list(p.get_parameter("controller.S").value)
        self.v_min = p.get_parameter("controller.v_min").value
        self.v_max = p.get_parameter("controller.v_max").value
        self.w_min = p.get_parameter("controller.omega_min").value
        self.w_max = p.get_parameter("controller.omega_max").value

        self.stop_pos_tol = float(p.get_parameter("goal.stop_pos_tol").value)
        self.stop_yaw_tol = math.radians(float(p.get_parameter("goal.stop_yaw_deg").value))
        self.slow_radius  = float(p.get_parameter("goal.slow_radius").value)
        self.slow_exp     = float(p.get_parameter("goal.slow_exp").value)
        self.terminal_boost = float(p.get_parameter("goal.terminal_boost").value)

        # 'costmap.topic' wurde im LocalPlannerNode deklariert:
        self.costmap_topic = p.get_parameter("costmap.topic").value
        self.costmap_weight = p.get_parameter("costmap.weight").value
        self.costmap_power  = p.get_parameter("costmap.power").value
        self.costmap_thresh = p.get_parameter("costmap.thresh").value
        self.costmap_unknown_lethal = p.get_parameter("costmap.unknown_is_lethal").value
        self.fp_r = p.get_parameter("robot.footprint_radius").value
        self.safe_m = p.get_parameter("obstacle.safety_margin").value
        self.ahead_dists = list(p.get_parameter("costmap.ahead_distances").value)
        self.samples_per_circle = int(p.get_parameter("costmap.samples_per_circle").value)

        self.solver_iters = p.get_parameter("solver.max_iters").get_parameter_value().integer_value
        self.solver_tfrac = p.get_parameter("solver.time_frac").get_parameter_value().double_value

        # Torch matrices
        self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
        self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
        self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))

        # Submodules
        self.model = Model(self)
        self.optimizer = Optimizer(self)

        # Dynamic reconfigure & warm start
        p.add_on_set_parameters_callback(self.parameter_update_callback)
        self.optimizer.init_warm_start(self.N)

    def solve_mpc(self, current_state: PoseStamped, ref_traj: PoseArray, T: float):
        # current pose -> tensor
        x = current_state.pose.position.x
        y = current_state.pose.position.y
        yaw = yaw_from_quat(current_state.pose.orientation)
        x0 = torch.tensor([x, y, yaw], dtype=torch.float32)

        # PoseArray -> [N,3] tensor
        refs = []
        for i in range(min(self.N, len(ref_traj.poses))):
            pr = ref_traj.poses[i]
            refs.append(torch.tensor([pr.position.x, pr.position.y, yaw_from_quat(pr.orientation)], dtype=torch.float32))
        while len(refs) < self.N:
            refs.append(refs[-1] if refs else torch.tensor([x, y, yaw], dtype=torch.float32))
        ref_t = torch.stack(refs, dim=0)

        # goal metrics (use last ref)
        gx, gy, gyaw = ref_t[-1, 0].item(), ref_t[-1, 1].item(), ref_t[-1, 2].item()
        d_goal = math.hypot(gx - x, gy - y)
        yaw_err = math.atan2(math.sin(yaw - gyaw), math.cos(yaw - gyaw))

        # hard stop if within tight tolerance
        if (d_goal <= self.stop_pos_tol) and (abs(yaw_err) <= self.stop_yaw_tol):
            u0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
            X_pred = torch.zeros(self.N + 1, 3, dtype=torch.float32)
            X_pred[0, :] = x0
            for k in range(1, self.N + 1):
                X_pred[k, :] = X_pred[k - 1, :]
            return float(u0[0]), float(u0[1]), X_pred

        # slow down near goal & boost terminal costs
        if self.slow_radius > 1e-6:
            scale = min(1.0, max(0.0, (d_goal / self.slow_radius)))
            scale = pow(scale, self.slow_exp)
            self._v_max_eff = float(self.v_max * scale)
            self._w_max_eff = float(self.w_max * scale)
            self.S_dyn = self.S * (1.0 + self.terminal_boost * (1.0 - scale))
        else:
            self._v_max_eff = self.v_max
            self._w_max_eff = self.w_max
            self.S_dyn = self.S

        # Solve
        u0, X_pred = self.optimizer.solve(x0, ref_t)
        return float(u0[0].item()), float(u0[1].item()), X_pred

    # dynamic params update (only those we actually use)
    def parameter_update_callback(self, params):
        for param in params:
            n = param.name
            if n == "controller.N":
                self.N = int(param.value); self.optimizer.init_warm_start(self.N)
            elif n == "controller.T":
                self.T = float(param.value)
                try: self.node.destroy_timer(self.node.timer)
                except Exception: pass
                self.node.timer = self.node.create_timer(self.T, self.node.controller_timer_callback)
            elif n == "controller.Q":
                self.Qv = list(param.value); self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
            elif n == "controller.R":
                self.Rv = list(param.value); self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
            elif n == "controller.S":
                self.Sv = list(param.value); self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))
            elif n == "controller.v_min":      self.v_min = float(param.value)
            elif n == "controller.v_max":      self.v_max = float(param.value)
            elif n == "controller.omega_min":  self.w_min = float(param.value)
            elif n == "controller.omega_max":  self.w_max = float(param.value)
            elif n == "costmap.weight":        self.costmap_weight = float(param.value)
            elif n == "costmap.power":         self.costmap_power  = float(param.value)
            elif n == "costmap.thresh":        self.costmap_thresh = float(param.value)
            elif n == "costmap.unknown_is_lethal": self.costmap_unknown_lethal = bool(param.value)
            elif n == "robot.footprint_radius":   self.fp_r = float(param.value)
            elif n == "obstacle.safety_margin":   self.safe_m = float(param.value)
            elif n == "costmap.ahead_distances":  self.ahead_dists = list(param.value)
            elif n == "costmap.samples_per_circle": self.samples_per_circle = int(param.value)
            elif n == "solver.max_iters":  self.solver_iters = int(param.value)
            elif n == "solver.time_frac":  self.solver_tfrac = float(param.value)
            elif n == "goal.stop_pos_tol":   self.stop_pos_tol = float(param.value)
            elif n == "goal.stop_yaw_deg":   self.stop_yaw_tol = math.radians(float(param.value))
            elif n == "goal.slow_radius":    self.slow_radius  = float(param.value)
            elif n == "goal.slow_exp":       self.slow_exp     = float(param.value)
            elif n == "goal.terminal_boost": self.terminal_boost = float(param.value)
        return SetParametersResult(successful=True)


# =========================
# Trajectory Planner
# =========================
class TrajectoryPlanner:
    def __init__(self, node: Node):
        self.node = node
        self.goal_pose: PoseStamped | None = None

        # Lock radius: inside this, reference is fully "locked" to the goal pose
        if not self.node.has_parameter("reference.lock_radius"):
            self.node.declare_parameter("reference.lock_radius", 0.5)
        self.lock_radius = float(self.node.get_parameter("reference.lock_radius").value)

        self.goal_pose_sub = self.node.create_subscription(
            PoseStamped, "/goal_pose", self.goal_pose_callback, 10
        )

    def goal_pose_callback(self, msg: PoseStamped):
        self.goal_pose = msg

    def _to_frame(self, ps: PoseStamped, target: str) -> PoseStamped:
        if ps.header.frame_id == target or ps.header.frame_id == "":
            return ps
        # target <- source
        tr = self.node.tf_buffer.lookup_transform(
            target, ps.header.frame_id, rclpy.time.Time()
        ).transform
        # build 4x4 from transform
        q = [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
        T_ts = tf_transformations.quaternion_matrix(q)
        T_ts[0,3] = tr.translation.x
        T_ts[1,3] = tr.translation.y
        T_ts[2,3] = tr.translation.z
        # pose -> matrix
        qg = [ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w]
        T_sp = tf_transformations.quaternion_matrix(qg)
        T_sp[0,3] = ps.pose.position.x
        T_sp[1,3] = ps.pose.position.y
        T_sp[2,3] = ps.pose.position.z
        # target pose
        T_tp = T_ts @ T_sp
        out = PoseStamped()
        out.header.stamp = ps.header.stamp
        out.header.frame_id = target
        out.pose.position.x = T_tp[0,3]
        out.pose.position.y = T_tp[1,3]
        out.pose.position.z = T_tp[2,3]
        qo = tf_transformations.quaternion_from_matrix(T_tp)
        out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = qo
        return out

    def build_reference(self, current: PoseStamped, N: int, T: float) -> PoseArray:
        pa = PoseArray()
        pa.header.stamp = current.header.stamp
        pa.header.frame_id = self.node.parent_frame

        # make sure goal is in parent_frame
        if self.goal_pose is None:
            gx, gy, yaw_g = current.pose.position.x, current.pose.position.y, 0.0
        else:
            goal_pf = self._to_frame(self.goal_pose, self.node.parent_frame)
            gx, gy = goal_pf.pose.position.x, goal_pf.pose.position.y
            gq = goal_pf.pose.orientation
            _,_, yaw_g = tf_transformations.euler_from_quaternion([gq.x, gq.y, gq.z, gq.w])

        x0, y0 = current.pose.position.x, current.pose.position.y
        d = math.hypot(gx - x0, gy - y0)

        # lock behavior (falls vorhanden)
        lock_radius = float(self.node.get_parameter("reference.lock_radius").value) if self.node.has_parameter("reference.lock_radius") else 0.0
        if d <= lock_radius:
            for _ in range(N):
                pose = Pose()
                pose.position.x = gx; pose.position.y = gy
                pose.orientation.z = math.sin(yaw_g/2.0); pose.orientation.w = math.cos(yaw_g/2.0)
                pa.poses.append(pose)
            return pa

        th_ref = math.atan2(gy - y0, gx - x0)
        for k in range(N):
            a = (k+1)/float(N)
            pose = Pose()
            pose.position.x = x0 + a*(gx-x0)
            pose.position.y = y0 + a*(gy-y0)
            pose.orientation.z = math.sin(th_ref/2.0)
            pose.orientation.w = math.cos(th_ref/2.0)
            pa.poses.append(pose)
        return pa

# =========================
# Optimizer (PyTorch MPC)
# =========================
class Optimizer:
    def __init__(self, controller: MPCController):
        self.c = controller
        self.U = None  # Warm-start [N,2]
        torch.set_num_threads(1)

    @staticmethod
    def wrap_angle(a: torch.Tensor) -> torch.Tensor:
        """Map any angle to (-pi, pi]."""
        return torch.atan2(torch.sin(a), torch.cos(a))

    def init_warm_start(self, N: int):
        self.U = torch.zeros(N, 2, dtype=torch.float32, requires_grad=True)

    def clamp_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        """Clamp control with dynamic limits if available."""
        v_max_eff = float(min(self.c.v_max, getattr(self.c, "_v_max_eff", self.c.v_max)))
        w_max_eff = float(min(self.c.w_max, getattr(self.c, "_w_max_eff", self.c.w_max)))
        v_min_eff = max(self.c.v_min, -v_max_eff)
        w_min_eff = max(self.c.w_min, -w_max_eff)
        v = torch.clamp(u_vec[0], v_min_eff, v_max_eff)
        w = torch.clamp(u_vec[1], w_min_eff, w_max_eff)
        return torch.stack([v, w])

    # ---- Costs ----
    def obstacle_penalty(self, state: torch.Tensor) -> torch.Tensor:
        """
        Look-ahead ring sampling around centerline points in front of the robot.
        Aggregated with smooth-max (fixed beta) + soft ramp above threshold.
        """
        cm = getattr(self.c.node, "costmap", None)
        if cm is None or not cm.ready:
            return torch.tensor(0.0, dtype=torch.float32)

        x, y, th = state[0], state[1], state[2]
        cos_th = torch.cos(th); sin_th = torch.sin(th)

        r = float(self.c.fp_r + self.c.safe_m)
        M = max(1, int(self.c.samples_per_circle))
        d_list = self.c.ahead_dists if len(self.c.ahead_dists) > 0 else [0.0]

        vals = []
        for d in d_list:
            cx = x + cos_th * float(d)
            cy = y + sin_th * float(d)
            if r <= 1e-6 or M == 1:
                vals.append(cm.sample(cx, cy))
            else:
                for j in range(M):
                    phi = 2.0 * math.pi * j / M
                    ox = r * math.cos(phi)
                    oy = r * math.sin(phi)
                    wx = cx + cos_th * ox - sin_th * oy
                    wy = cy + sin_th * ox + cos_th * oy
                    vals.append(cm.sample(wx, wy))

        V = torch.stack(vals) if len(vals) > 1 else vals[0].unsqueeze(0)

        # smooth max with fixed beta (no parameter)
        beta = 8.0
        m = torch.max(V)
        val = m + torch.log(torch.clamp(torch.exp((V - m) * beta).mean(), min=1e-9)) / beta
        val = torch.clamp(val, 0.0, 1.0)

        if self.c.costmap_unknown_lethal:
            val = torch.where(val >= 0.992, torch.tensor(1.0, dtype=torch.float32), val)

        # soft ramp above threshold
        t = float(self.c.costmap_thresh)
        soft = torch.clamp((val - t) / (1.0 - t + 1e-6), 0.0, 1.0)
        return (soft ** float(self.c.costmap_power)) * float(self.c.costmap_weight)

    # ---- Solve ----
    def solve(self, x0: torch.Tensor, ref_traj: torch.Tensor):
        if self.U is None or self.U.shape[0] != self.c.N:
            self.init_warm_start(self.c.N)

        Q, R = self.c.Q, self.c.R
        S = getattr(self.c, "S_dyn", self.c.S)
        N = self.c.N

        def rollout_cost_and_traj(U_local):
            state = x0.clone().detach().float()
            J = torch.tensor(0.0, dtype=torch.float32)
            Xs = [state.clone()]

            for k in range(N):
                u = self.clamp_u(U_local[k])
                state = self.c.model.kinematics_step(state, u)
                Xs.append(state.clone())

                lin_err = state[:2] - ref_traj[k][:2]
                yaw_err = self.wrap_angle(state[2] - ref_traj[k][2])
                diff = torch.stack([lin_err[0], lin_err[1], yaw_err])

                J = J + diff @ Q @ diff + u @ R @ u + self.obstacle_penalty(state)

            lin_err_T = state[:2] - ref_traj[-1][:2]
            yaw_err_T = self.wrap_angle(state[2] - ref_traj[-1][2])
            diff_T = torch.stack([lin_err_T[0], lin_err_T[1], yaw_err_T])
            J = J + diff_T @ S @ diff_T

            return J, torch.stack(Xs, dim=0)

        # Adam with time budget
        optimizer = torch.optim.Adam([self.U], lr=0.2)
        max_iters = int(getattr(self.c, "solver_iters", 8))
        time_frac = float(getattr(self.c, "solver_tfrac", 0.5))
        time_budget = max(1e-3, time_frac * self.c.T)  # seconds

        start = time.perf_counter()
        for _ in range(max_iters):
            optimizer.zero_grad(set_to_none=True)
            J, _ = rollout_cost_and_traj(self.U)
            J.backward()
            torch.nn.utils.clip_grad_norm_([self.U], max_norm=1.0)
            optimizer.step()
            if (time.perf_counter() - start) > time_budget:
                break

        with torch.no_grad():
            _, X_pred = rollout_cost_and_traj(self.U)
            u0 = self.clamp_u(self.U[0].detach())
            # warm start shift
            nextU = self.U[1:].detach().clone()
            self.U[:-1].copy_(nextU)
            self.U[-1].zero_()
        return u0, X_pred


# =========================
# Model (Kinematics)
# =========================
class Model:
    """Holds the motion/kinematics model used by the MPC."""
    def __init__(self, controller: MPCController):
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

# =========================
# Entry point
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
