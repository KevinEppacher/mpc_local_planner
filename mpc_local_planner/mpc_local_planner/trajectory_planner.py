#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import tf_transformations

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
