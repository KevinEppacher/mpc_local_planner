#!/usr/bin/env python3
"""
Goal→Plan Bridge for Nav2

Listens to an RViz "2D Goal Pose" topic (PoseStamped) and triggers Nav2's
`/compute_path_to_pose` action on `planner_server`. Optionally republishes the
resulting `nav_msgs/Path` on a topic (e.g., /global_plan) with transient local
QoS so you can visualize it in RViz even if it arrives before RViz subscribes.

Parameters (ROS 2 params):
- goal_topic   (string, default "/goal_pose")
- action_name  (string, default "/compute_path_to_pose")
- output_topic (string, default "/global_plan")
- planner_id   (string, default "")  # e.g., "GridBased" for NavFn/Smac depending on your setup
- use_start    (bool,   default False) # False = use TF robot pose as start
- goal_frame   (string, default "")   # If non-empty, override goal.header.frame_id

Usage:
  ros2 run <your_pkg> goal_to_plan_bridge

Or directly:
  python3 goal_to_plan_bridge.py --ros-args \
    -p goal_topic:=/goal_pose -p output_topic:=/global_plan

Make sure `planner_server` and your global costmap are ACTIVE and TF provides
`map -> base_link`. If `use_sim_time` is enabled, ensure `/clock` is publishing.
"""

from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus


def _status_to_str(status: int) -> str:
    try:
        mapping = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        return mapping.get(status, str(status))
    except Exception:
        return str(status)

class GoalToPlanBridge(Node):
    """Bridge node: PoseStamped goal → Nav2 action → Path publisher."""

    def __init__(self):
        super().__init__('goal_to_plan_bridge')

        # Declare params
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('action_name', '/compute_path_to_pose')
        self.declare_parameter('output_topic', '/global_plan')
        self.declare_parameter('planner_id', '')
        self.declare_parameter('use_start', False)
        self.declare_parameter('goal_frame', '')

        goal_topic   = self.get_parameter('goal_topic').get_parameter_value().string_value
        action_name  = self.get_parameter('action_name').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self._planner_id = self.get_parameter('planner_id').get_parameter_value().string_value
        self._use_start  = self.get_parameter('use_start').get_parameter_value().bool_value
        self._goal_frame = self.get_parameter('goal_frame').get_parameter_value().string_value

        # QoS: result path publisher with TRANSIENT_LOCAL (latched-like)
        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._path_pub = self.create_publisher(Path, output_topic, path_qos)

        # Subscribe to goal poses from RViz
        self._goal_sub = self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)

        # Action client
        self._ac = ActionClient(self, ComputePathToPose, action_name)
        self._current_goal_handle = None

        self.get_logger().info(
            f"Bridge ready. Subscribing to '{goal_topic}', action '{action_name}',"
            f" publishing Path on '{output_topic}'."
        )

    def _on_goal(self, msg: PoseStamped):
        """Callback when a new goal pose is received from RViz."""
        # Optionally override frame
        if self._goal_frame:
            msg.header.frame_id = self._goal_frame

        # Make sure action server is available
        if not self._ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Planner action server not available (timeout 1.0s).')
            return

        # Preempt: cancel any in-flight goal
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().debug(f'Cancel request failed: {e}')

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = msg
        goal_msg.use_start = self._use_start  # False: use TF for start pose
        goal_msg.planner_id = self._planner_id

        self.get_logger().info(
            f"Requesting plan to x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
            f" in frame='{msg.header.frame_id or 'UNKNOWN'}' (use_start={self._use_start},"
            f" planner_id='{self._planner_id or '<default>'}')."
        )

        send_future = self._ac.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Plan goal was rejected by the action server.')
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info('Plan goal accepted. Waiting for result...')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        try:
            result = future.result().result
            status = future.result().status
        except Exception as e:
            self.get_logger().error(f'Failed to get result: {e}')
            return

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f'Planning finished with non-success status: {_status_to_str(status)}')
        else:
            self.get_logger().info('Planning finished: SUCCEEDED')

        path: Path = result.path
        n = len(path.poses)
        if n == 0:
            self.get_logger().warn(f'Planner returned empty path (error_code={result.error_code}).')
            return

        # Publish path (republished copy preserves header from planner)
        self._path_pub.publish(path)
        self.get_logger().info(f'Published path with {n} poses to {self._path_pub.topic_name}.')


def main():
    rclpy.init()
    node = GoalToPlanBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
