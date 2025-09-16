import rclpy
from mpc_local_planner.mpc_local_planner import LocalPlannerNode

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

