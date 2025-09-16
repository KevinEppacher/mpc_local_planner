#!/usr/bin/env python3
import time, math, torch, rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TwistStamped, PoseArray, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf_transformations

from mpc_local_planner.controller import MPCController
from mpc_local_planner.trajectory_planner import TrajectoryPlanner
from mpc_local_planner.core.costmap import CostmapGrid
from mpc_local_planner.utils import make_pose, _inferno_rgb01

class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner_node")
        # basic params
        if not self.has_parameter("robot.parent_frame"):
            self.declare_parameter("robot.parent_frame", "map")
        if not self.has_parameter("robot.base_frame"):
            self.declare_parameter("robot.base_frame", "base_link")
        if not self.has_parameter("costmap.topic"):
            self.declare_parameter("costmap.topic", "/local_costmap/costmap")

        self.parent_frame = self.get_parameter("robot.parent_frame").value
        self.base_frame   = self.get_parameter("robot.base_frame").value
        self.costmap_topic = self.get_parameter("costmap.topic").value

        # submodules
        self.costmap = CostmapGrid(self)
        self.controller = MPCController(self, self.costmap)
        self.planner    = TrajectoryPlanner(self)

        # TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.T_map_odom = None
        self.current_pose = None

        # io
        self.cmd_vel_pub  = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.horizon_pub  = self.create_publisher(PoseArray, "horizon", 10)
        self.ref_pub      = self.create_publisher(PoseArray, "reference", 10)
        self.viz_pub      = self.create_publisher(MarkerArray, "/mpc/obstacle_viz", 10)

        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, self.costmap_topic, self.costmap.cb, 1)

        self.timer    = self.create_timer(self.controller.T, self.controller_timer_callback)
        self.tf_timer = self.create_timer(0.2, self.tf_timer_callback)

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

        self._publish_obstacle_viz()

        # ---- timing log
        self._log_timing(start_wall, period_ms)

    # -------- small helpers for publishing & timing --------
    def _publish_cmd(self, v: float, w: float):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.parent_frame
        v = max(self.controller.robot.v_min, min(self.controller.robot.v_max, v))
        w = max(self.controller.robot.w_min, min(self.controller.robot.w_max, w))
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

    def _publish_obstacle_viz(self):
        dbg = getattr(self.controller, "last_obst_dbg", None)
        if not dbg:
            return
        ma = MarkerArray()
        ns = "mpc_obst"
        now = self.get_clock().now().to_msg()

        # Common: frame + lifetime
        def base_marker(mid, mtype):
            m = Marker()
            m.header.frame_id = self.parent_frame
            m.header.stamp = now
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            return m

        # 4.1 Look-ahead circles (LINE_STRIP)
        mid = 0
        x, y, th = dbg["state"]
        r = dbg["radius"]
        for i, d in enumerate(dbg["lookahead"]):
            cx = x + math.cos(th)*d
            cy = y + math.sin(th)*d
            circ = base_marker(mid, Marker.LINE_STRIP); mid += 1
            circ.scale.x = 0.02  # line width
            circ.color.r, circ.color.g, circ.color.b, circ.color.a = (0.1, 0.8, 0.1, 0.8)
            # close the circle with ~40 segments
            K = 40
            for k in range(K+1):
                phi = 2.0*math.pi*k/ K
                ox, oy = r*math.cos(phi), r*math.sin(phi)
                # rotate by th to keep aligned with robot (optional — circle is invariant)
                px = cx + math.cos(th)*ox - math.sin(th)*oy
                py = cy + math.sin(th)*ox + math.cos(th)*oy
                pt = Point()
                pt.x, pt.y, pt.z = px, py, 0.02
                circ.points.append(pt)
            ma.markers.append(circ)

        # 4.2 Sampled points as a single POINTS marker (per-point colors, inferno)
        if dbg["samples"]:
            pts = base_marker(mid, Marker.SPHERE_LIST); mid += 1
            pts.scale.x = pts.scale.y = pts.scale.z = 0.06
            pts.color.a = 1.0

            pts.points = []
            pts.colors = []

            # Some pipelines provide tuples of (sx, sy, raw) and others (sx, sy, raw, ring).
            # Be tolerant: slice the first three fields and ignore extras.
            for s in dbg["samples"]:
                if not isinstance(s, (list, tuple)) or len(s) < 3:
                    continue
                sx, sy, raw = s[0], s[1], s[2]

                x = 1 - float(raw)
                lo, hi = 0.3, 0.8
                x = (x - lo) / (hi - lo)
                r, g, b = _inferno_rgb01(x)
                c = ColorRGBA()
                c.r, c.g, c.b, c.a = float(r), float(g), float(b), 0.95

                p = Point()
                p.x, p.y, p.z = float(sx), float(sy), 0.04

                pts.points.append(p)
                pts.colors.append(c)

            ma.markers.append(pts)

        # 4.3 Text with values (smooth_max / soft / penalty)
        txt = base_marker(mid, Marker.TEXT_VIEW_FACING); mid += 1
        txt.pose.position.x = x; txt.pose.position.y = y; txt.pose.position.z = 0.3
        txt.scale.z = 0.18  # font height
        txt.color.r, txt.color.g, txt.color.b, txt.color.a = (0.9, 0.9, 0.9, 0.9)
        txt.text = f"max~: {dbg['smooth_max']:.2f}\nsoft: {dbg['soft']:.2f}\npen: {dbg['penalty']:.1f}"
        ma.markers.append(txt)

        self.viz_pub.publish(ma)
