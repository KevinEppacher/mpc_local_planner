#!/usr/bin/env python3
import math
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, PoseArray, Pose
from builtin_interfaces.msg import Time as RosTime

# TF2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from rclpy.duration import Duration
import tf_transformations
from nav_msgs.msg import OccupancyGrid, Path

class CostmapGrid:
    """
    Minimal costmap holder with bilinear sampling.
    Converts OccupancyGrid (0..255) to [0,1], treats 253..255 as 1.0 (lethal/unknown).
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

        data = np.array(msg.data, dtype=np.int16).reshape(self.H, self.W)
        # Map 0..255 -> 0..1, lethal/unknown => 1.0
        grid = np.clip(data, 0, 255).astype(np.float32)
        grid[grid >= 253] = 255.0
        grid = grid / 255.0  # [0,1]
        self.grid = torch.from_numpy(grid)  # CPU tensor
        self.ready = True

    def sample(self, x: torch.Tensor, y: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Bilinear sample at world (x,y). Returns scalar torch tensor.
        If out of bounds or not ready -> 0.
        """
        if not self.ready or self.grid is None:
            return torch.tensor(0.0, dtype=torch.float32)

        # world -> grid coords
        gx = (x - self.ox) / self.res
        gy = (y - self.oy) / self.res

        # Nachbarn
        x0 = torch.floor(gx); y0 = torch.floor(gy)
        x1 = x0 + 1.0;       y1 = y0 + 1.0

        # clamp in bounds
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
        c  = c0 * (1.0 - wy) + c1 * wy  # in [0,1]
        return c * float(weight)



# =========================
# Local Planner Node
# =========================
class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner_node")
        self.get_logger().info(f"LocalPlannerNode initialized: {self.get_name()}")

        # Submodules
        self.controller = MPCController(self)
        self.planner = TrajectoryPlanner(self)

        # Parameters
        if not self.has_parameter("robot.parent_frame"):
            self.declare_parameter("robot.parent_frame", "map")
        if not self.has_parameter("robot.base_frame"):
            self.declare_parameter("robot.base_frame", "base_link")

        self.parent_frame = self.get_parameter("robot.parent_frame").get_parameter_value().string_value
        self.base_frame   = self.get_parameter("robot.base_frame").get_parameter_value().string_value

        # >>> Use TF instead of /odom
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Internal pose cache (PoseStamped)
        self.current_pose = None
        self.T_map_odom = None

        # Publishers
        self.cmd_vel_pub  = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.horizon_pub  = self.create_publisher(PoseArray, "/mpc/horizon", 10)  # publish planned horizon

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.costmap = CostmapGrid(self)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            "/costmap",
            self.costmap.cb,
            1
        )

        # Timer (controller cycle)
        self.timer = self.create_timer(self.controller.T, self.controller_timer_callback)

        # Timer for TF updates
        self.tf_timer = self.create_timer(0.2, self.tf_timer_callback)

        self.get_logger().info(f"Using TF for pose: {self.parent_frame} -> {self.base_frame}")

    # Odometry → PoseStamped (x,y,yaw)
    def _odom_cb(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose

        if self.T_map_odom is not None:
            # Odom pose as matrix
            q = [ps.pose.orientation.x, ps.pose.orientation.y,
                ps.pose.orientation.z, ps.pose.orientation.w]
            T_odom_base = tf_transformations.quaternion_matrix(q)
            T_odom_base[0,3] = ps.pose.position.x
            T_odom_base[1,3] = ps.pose.position.y
            T_odom_base[2,3] = ps.pose.position.z

            # map->odom as matrix
            t = self.T_map_odom.translation
            q = [self.T_map_odom.rotation.x, self.T_map_odom.rotation.y,
                self.T_map_odom.rotation.z, self.T_map_odom.rotation.w]
            T_map_odom = tf_transformations.quaternion_matrix(q)
            T_map_odom[0,3] = t.x
            T_map_odom[1,3] = t.y
            T_map_odom[2,3] = t.z

            # Multiplication: map->base = map->odom * odom->base
            T_map_base = T_map_odom @ T_odom_base

            # Back to PoseStamped
            self.current_pose = PoseStamped()
            self.current_pose.header.frame_id = "map"
            self.current_pose.header.stamp = msg.header.stamp
            self.current_pose.pose.position.x = T_map_base[0,3]
            self.current_pose.pose.position.y = T_map_base[1,3]
            self.current_pose.pose.position.z = T_map_base[2,3]
            q_map_base = tf_transformations.quaternion_from_matrix(T_map_base)
            self.current_pose.pose.orientation.x = q_map_base[0]
            self.current_pose.pose.orientation.y = q_map_base[1]
            self.current_pose.pose.orientation.z = q_map_base[2]
            self.current_pose.pose.orientation.w = q_map_base[3]

        else:
            # Fallback: odom frame only
            self.current_pose = ps

    def tf_timer_callback(self):
        try:
            self.transform = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
            self.T_map_odom = self.transform.transform
        except Exception as e:
            self.get_logger().debug(f"Transform not available: {e}")
            return None

    def controller_timer_callback(self):
        if self.current_pose is None or self.planner.goal_pose is None:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = self.parent_frame
            self.cmd_vel_pub.publish(cmd)
            return

        ref_traj = self.planner.build_reference(self.current_pose, self.controller.N, self.controller.T)

        # now: v, w **and** X_pred
        v, w, X_pred = self.controller.solve_mpc(self.current_pose, ref_traj, self.controller.T)

        # Publish Twist
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.parent_frame
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)
        self.cmd_vel_pub.publish(cmd)

        # Publish planned horizon as PoseArray
        pa = PoseArray()
        pa.header.stamp = cmd.header.stamp
        pa.header.frame_id = self.parent_frame
        for i in range(X_pred.shape[0]):  # N+1 states
            px, py, pth = float(X_pred[i,0]), float(X_pred[i,1]), float(X_pred[i,2])
            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.orientation.z = math.sin(pth/2.0)
            pose.orientation.w = math.cos(pth/2.0)
            pa.poses.append(pose)
        self.horizon_pub.publish(pa)


# =========================
# Controller
# =========================
class MPCController:
    def __init__(self, node: Node):
        self.node = node

        # -------- Declare parameters --------
        p = self.node
        _decl = p.declare_parameter
        if not p.has_parameter("controller.N"):             _decl("controller.N", 12)
        if not p.has_parameter("controller.T"):             _decl("controller.T", 0.2)
        if not p.has_parameter("controller.Q"):             _decl("controller.Q", [1.0, 1.0, 0.2])
        if not p.has_parameter("controller.R"):             _decl("controller.R", [0.2, 0.05])
        if not p.has_parameter("controller.S"):             _decl("controller.S", [2.0, 2.0, 0.4])  # Terminal weight
        if not p.has_parameter("controller.v_min"):         _decl("controller.v_min", -0.4)
        if not p.has_parameter("controller.v_max"):         _decl("controller.v_max",  0.4)
        if not p.has_parameter("controller.omega_min"):     _decl("controller.omega_min", -math.pi/4)
        if not p.has_parameter("controller.omega_max"):     _decl("controller.omega_max",  math.pi/4)
        if not p.has_parameter("controller.ref_speed"):     _decl("controller.ref_speed", 0.5)

        # Obstacle penalty
        if not p.has_parameter("obstacle.center_x"):        _decl("obstacle.center_x", 2.0)
        if not p.has_parameter("obstacle.center_y"):        _decl("obstacle.center_y", 0.5)
        if not p.has_parameter("obstacle.diameter"):        _decl("obstacle.diameter", 1.0)
        if not p.has_parameter("robot.diameter"):           _decl("robot.diameter", 0.3)
        if not p.has_parameter("obstacle.penalty_weight"):  _decl("obstacle.penalty_weight", 300.0)
        if not p.has_parameter("obstacle.safety_margin"):   _decl("obstacle.safety_margin", 0.0)
        if not p.has_parameter("costmap.topic"):            _decl("costmap.topic", "/costmap")
        if not p.has_parameter("costmap.weight"):           _decl("costmap.weight", 50.0)
        if not p.has_parameter("costmap.unknown_is_lethal"):_decl("costmap.unknown_is_lethal", True)

        # -------- Read parameters --------
        self.N  = p.get_parameter("controller.N").get_parameter_value().integer_value
        self.T  = p.get_parameter("controller.T").get_parameter_value().double_value
        self.Qv = list(p.get_parameter("controller.Q").get_parameter_value().double_array_value)
        self.Rv = list(p.get_parameter("controller.R").get_parameter_value().double_array_value)
        self.Sv = list(p.get_parameter("controller.S").get_parameter_value().double_array_value)
        self.v_min = p.get_parameter("controller.v_min").get_parameter_value().double_value
        self.v_max = p.get_parameter("controller.v_max").get_parameter_value().double_value
        self.w_min = p.get_parameter("controller.omega_min").get_parameter_value().double_value
        self.w_max = p.get_parameter("controller.omega_max").get_parameter_value().double_value
        self.ref_speed = p.get_parameter("controller.ref_speed").get_parameter_value().double_value
        self.costmap_topic = p.get_parameter("costmap.topic").get_parameter_value().string_value
        self.costmap_weight = p.get_parameter("costmap.weight").get_parameter_value().double_value
        self.costmap_unknown_lethal = p.get_parameter("costmap.unknown_is_lethal").get_parameter_value().bool_value


        self.obs_cx = p.get_parameter("obstacle.center_x").get_parameter_value().double_value
        self.obs_cy = p.get_parameter("obstacle.center_y").get_parameter_value().double_value
        self.obs_d  = p.get_parameter("obstacle.diameter").get_parameter_value().double_value
        self.rob_d  = p.get_parameter("robot.diameter").get_parameter_value().double_value
        self.obs_w  = p.get_parameter("obstacle.penalty_weight").get_parameter_value().double_value
        self.obs_margin = p.get_parameter("obstacle.safety_margin").get_parameter_value().double_value

        # Torch matrices
        self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
        self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
        self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))

        # Submodules
        self.model = Model(self)
        self.optimizer = Optimizer(self)

        # Dynamic reconfigure
        p.add_on_set_parameters_callback(self.parameter_update_callback)

        # Warm-start U (persisted in Optimizer)
        self.optimizer.init_warm_start(self.N)

    def solve_mpc(self, current_state: PoseStamped, ref_traj: PoseArray, T: float):
        # Pose → torch
        x = current_state.pose.position.x
        y = current_state.pose.position.y
        q = current_state.pose.orientation
        _,_, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])  # Convert quaternion to yaw
        x0 = torch.tensor([x, y, yaw], dtype=torch.float32)

        # PoseArray → refs [N,3]
        refs = []
        for i in range(min(self.N, len(ref_traj.poses))):
            pr = ref_traj.poses[i]
            xr, yr = pr.position.x, pr.position.y
            qr = pr.orientation
            _,_, yaw_r = tf_transformations.euler_from_quaternion([qr.x, qr.y, qr.z, qr.w])
            refs.append(torch.tensor([xr, yr, yaw_r], dtype=torch.float32))
        while len(refs) < self.N:
            refs.append(refs[-1] if refs else torch.tensor([x, y, yaw], dtype=torch.float32))
        ref_t = torch.stack(refs, dim=0)

        # Solve → u0 & planned trajectory
        u0, X_pred = self.optimizer.solve(x0, ref_t)
        v = float(u0[0].item()); w = float(u0[1].item())
        return v, w, X_pred

    # Dynamic parameter update
    def parameter_update_callback(self, params):
        for param in params:
            n = param.name
            if n == "controller.N":
                self.N = int(param.value); self.optimizer.init_warm_start(self.N)
            elif n == "controller.T":
                self.T = float(param.value)
                self.node.destroy_timer(self.node.timer)
                self.node.timer = self.node.create_timer(self.T, self.node.controller_timer_callback)
            elif n == "controller.Q":
                self.Qv = list(param.value); self.Q = torch.diag(torch.tensor(self.Qv, dtype=torch.float32))
            elif n == "controller.R":
                self.Rv = list(param.value); self.R = torch.diag(torch.tensor(self.Rv, dtype=torch.float32))
            elif n == "controller.S":
                self.Sv = list(param.value); self.S = torch.diag(torch.tensor(self.Sv, dtype=torch.float32))
            elif n == "controller.v_min":   self.v_min = float(param.value)
            elif n == "controller.v_max":   self.v_max = float(param.value)
            elif n == "controller.omega_min": self.w_min = float(param.value)
            elif n == "controller.omega_max": self.w_max = float(param.value)
            elif n == "controller.ref_speed": self.ref_speed = float(param.value)
            elif n == "obstacle.center_x":    self.obs_cx = float(param.value)
            elif n == "obstacle.center_y":    self.obs_cy = float(param.value)
            elif n == "obstacle.diameter":    self.obs_d  = float(param.value)
            elif n == "robot.diameter":       self.rob_d  = float(param.value)
            elif n == "obstacle.penalty_weight": self.obs_w = float(param.value)
            elif n == "obstacle.safety_margin":  self.obs_margin = float(param.value)
        return SetParametersResult(successful=True)


# =========================
# Trajectory Planner (aus /plan)
# =========================
class TrajectoryPlanner:
    """
    Erzeugt eine MPC-Referenz aus Nav2 /plan:
    - liest /plan (nav_msgs/Path)
    - berechnet Yaw als Tangente des Pfades (statt der orientierung aus /plan)
    - projiziert Start auf den Pfad (nächster Punkt zur Roboterpose)
    - resampled auf genau N Punkte innerhalb Lookahead-Distanz
    Optional: setzt controller.T automatisch aus Schrittweite und v_max.
    """
    def __init__(self, node: Node):
        self.node = node

        # Parameter
        p = self.node
        if not p.has_parameter("planner.plan_topic"):
            p.declare_parameter("planner.plan_topic", "/plan")
        if not p.has_parameter("planner.lookahead_dist"):
            p.declare_parameter("planner.lookahead_dist", 2.0)
        if not p.has_parameter("planner.project_from_robot"):
            p.declare_parameter("planner.project_from_robot", True)
        if not p.has_parameter("planner.feed_forward"):
            p.declare_parameter("planner.feed_forward", 0.8)
        if not p.has_parameter("planner.autoset_dt"):
            p.declare_parameter("planner.autoset_dt", False)

        self.plan_topic        = p.get_parameter("planner.plan_topic").get_parameter_value().string_value
        self.lookahead         = p.get_parameter("planner.lookahead_dist").get_parameter_value().double_value
        self.project_from_robot= p.get_parameter("planner.project_from_robot").get_parameter_value().bool_value
        self.feed_forward      = p.get_parameter("planner.feed_forward").get_parameter_value().double_value
        self.autoset_dt        = p.get_parameter("planner.autoset_dt").get_parameter_value().bool_value

        # Eingänge
        self.goal_pose: PoseStamped | None = None   # Fallback, falls kein /plan
        self.plan: Path | None = None

        self.goal_pose_sub = self.node.create_subscription(PoseStamped, "/goal_pose", self.goal_pose_callback, 10)
        self.plan_sub      = self.node.create_subscription(Path,        self.plan_topic,       self.plan_callback, 10)

        self.node.get_logger().info(f"TrajectoryPlanner: listening to plan '{self.plan_topic}', lookahead={self.lookahead} m")

    def goal_pose_callback(self, msg: PoseStamped):
        self.goal_pose = msg

    def plan_callback(self, msg: Path):
        self.plan = msg

    @staticmethod
    def _unwrap(yaws: np.ndarray) -> np.ndarray:
        out = yaws.copy()
        for i in range(1, len(out)):
            d = out[i] - out[i-1]
            while d > math.pi:
                out[i] -= 2*math.pi; d -= 2*math.pi
            while d < -math.pi:
                out[i] += 2*math.pi; d += 2*math.pi
        return out

    @staticmethod
    def _quat_from_yaw(yaw: float):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _build_from_plan(self, current: PoseStamped, N: int) -> PoseArray | None:
        if self.plan is None or len(self.plan.poses) < 2:
            return None

        # XY aus Plan
        pts = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in self.plan.poses], dtype=float)
        if pts.shape[0] < 2:
            return None

        # Tangenten-Yaw
        diffs = np.diff(pts, axis=0)
        yaws = np.arctan2(diffs[:, 1], diffs[:, 0])
        yaws = np.append(yaws, yaws[-1])   # letzter yaw = vor letzter
        yaws = self._unwrap(yaws)

        # Startindex: nächster Pfadpunkt zur Roboterpose
        if self.project_from_robot and current is not None:
            x0 = current.pose.position.x
            y0 = current.pose.position.y
            dists = np.linalg.norm(pts - np.array([x0, y0]), axis=1)
            start_idx = int(np.argmin(dists))
        else:
            start_idx = 0

        pts  = pts[start_idx:]
        yaws = yaws[start_idx:]
        if pts.shape[0] < 2:
            return None

        # Bogenlänge s
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        max_s = min(float(s[-1]), float(self.lookahead))

        # Resample auf genau N Punkte 0..max_s
        N = max(2, int(N))
        s_new = np.linspace(0.0, max_s, N)
        x_new = np.interp(s_new, s, pts[:, 0])
        y_new = np.interp(s_new, s, pts[:, 1])
        yaw_new = np.interp(s_new, s, yaws)
        # wrap zurück
        yaw_new = (yaw_new + math.pi) % (2 * math.pi) - math.pi

        # PoseArray bauen
        out = PoseArray()
        out.header = self.plan.header
        out.header.frame_id = out.header.frame_id or self.node.parent_frame
        for x, y, yaw in zip(x_new, y_new, yaw_new):
            qx, qy, qz, qw = self._quat_from_yaw(float(yaw))
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.x = qx
            p.orientation.y = qy
            p.orientation.z = qz
            p.orientation.w = qw
            out.poses.append(p)
        return out

    def _fallback_linear(self, current: PoseStamped, N: int) -> PoseArray:
        # Wie bisher: Linear von current → goal_pose
        pa = PoseArray()
        pa.header.stamp = current.header.stamp
        pa.header.frame_id = self.node.parent_frame
        x0 = current.pose.position.x
        y0 = current.pose.position.y
        gx = self.goal_pose.pose.position.x if self.goal_pose else x0
        gy = self.goal_pose.pose.position.y if self.goal_pose else y0
        th_ref = math.atan2(gy - y0, gx - x0)

        for k in range(N):
            a = float(k + 1) / float(N)
            xr = x0 + a * (gx - x0)
            yr = y0 + a * (gy - y0)
            pose = Pose()
            pose.position.x = xr
            pose.position.y = yr
            pose.orientation.z = math.sin(th_ref / 2.0)
            pose.orientation.w = math.cos(th_ref / 2.0)
            pa.poses.append(pose)
        return pa

    def build_reference(self, current: PoseStamped, N: int, T: float) -> PoseArray:
        """
        Liefert PoseArray (N Posen) mit tangentialem Yaw. Nutzt /plan wenn vorhanden,
        sonst fällt auf lineare Interpolation zu /goal_pose zurück.
        Optional: controller.T automatisch anpassen.
        """
        ref = self._build_from_plan(current, N)
        if ref is None:
            # Fallback
            return self._fallback_linear(current, N)

        # Optional: T aus Schrittweite & v_max ableiten und Controller-T setzen
        if self.autoset_dt and N >= 2:
            # Schrittweite ~ Lookahead/(N-1), dt = step/v_max * feed_forward
            step = self.lookahead / float(N - 1)
            v_max = getattr(self.node.controller, "v_max", 0.6)
            dt = (step / max(v_max, 1e-6)) * float(self.feed_forward)
            try:
                if abs(dt - self.node.controller.T) > 1e-3:
                    # ruft deinen Parameter-Callback auf und setzt Timer neu
                    self.node.set_parameters([rclpy.parameter.Parameter(
                        name="controller.T",
                        value=rclpy.Parameter.Type.DOUBLE,
                        double_value=float(dt)
                    )])
            except Exception as e:
                self.node.get_logger().warn(f"autoset_dt failed: {e}")

        return ref


# =========================
# Optimizer (PyTorch MPC)
# =========================
class Optimizer:
    def __init__(self, controller: MPCController):
        self.c = controller
        self.U = None  # Warm-start buffer [N,2]
        torch.set_num_threads(1)

    def init_warm_start(self, N: int):
        self.U = torch.zeros(N, 2, dtype=torch.float32, requires_grad=True)

    def clamp_u(self, u_vec: torch.Tensor) -> torch.Tensor:
        v = torch.clamp(u_vec[0], self.c.v_min, self.c.v_max)
        w = torch.clamp(u_vec[1], self.c.w_min, self.c.w_max)
        return torch.stack([v, w])

    def kinematics_step(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # state=[x,y,theta], u=[v,w]
        x, y, th = state
        v, w = u
        T = self.c.T
        return torch.stack([
            x + T * v * torch.cos(th),
            y + T * v * torch.sin(th),
            th + T * w
        ])

    def obstacle_penalty(self, state: torch.Tensor) -> torch.Tensor:
        # Prefer costmap if available
        cm = getattr(self.c.node, "costmap", None)
        if cm is not None and cm.ready:
            return cm.sample(state[0], state[1], self.c.costmap_weight)

        # Fallback: single circular obstacle (your previous behavior)
        dx = state[0] - self.c.obs_cx
        dy = state[1] - self.c.obs_cy
        dist = torch.sqrt(dx*dx + dy*dy + 1e-9)
        min_dist = 0.5*self.c.rob_d + 0.5*self.c.obs_d + self.c.obs_margin
        w = torch.tensor(self.c.obs_w, dtype=torch.float32)
        return torch.clamp(min_dist - dist, min=0.0)**2 * w

    def solve(self, x0: torch.Tensor, ref_traj: torch.Tensor):
        """
        x0: [3], ref_traj: [N,3]
        returns:
          u0: torch[2]
          X_pred: torch[N+1, 3]  (incl. x0)
        """
        if self.U is None or self.U.shape[0] != self.c.N:
            self.init_warm_start(self.c.N)

        optimizer = torch.optim.LBFGS([self.U], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
        # optimizer = torch.optim.Adam([self.U], lr=0.1)
        # optimizer = torch.optim.RMSprop([self.U], lr=0.01)
        Q, R, S = self.c.Q, self.c.R, self.c.S
        N = self.c.N

        def rollout_cost_and_traj(U_local):
            state = x0.clone().detach().float()
            J = torch.tensor(0.0, dtype=torch.float32)
            Xs = [state.clone()]  # include start state
            for k in range(N):
                u = self.clamp_u(U_local[k])
                state = self.kinematics_step(state, u)
                Xs.append(state.clone())
                diff = state - ref_traj[k]
                J = J + diff @ Q @ diff + u @ R @ u + self.obstacle_penalty(state)
            J = J + (state - ref_traj[-1]) @ S @ (state - ref_traj[-1])
            Xs = torch.stack(Xs, dim=0)  # [N+1, 3]
            return J, Xs

        def closure():
            optimizer.zero_grad(set_to_none=True)
            J, _ = rollout_cost_and_traj(self.U)
            J.backward()
            return J

        for _ in range(2):
            optimizer.step(closure)

        # After optimization: compute trajectory (before shift)
        with torch.no_grad():
            _, X_pred = rollout_cost_and_traj(self.U)
            u0 = self.clamp_u(self.U[0].detach())

            # Warm-start: shift safely (clone to avoid overlap)
            nextU = self.U[1:].detach().clone()
            self.U[:-1].copy_(nextU)
            self.U[-1].zero_()

        return u0, X_pred


# =========================
# Model (Kinematics)
# =========================
class Model:
    def __init__(self, controller: MPCController):
        self.c = controller
        controller.node.get_logger().info("Model initialized (Differential Drive).")


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
