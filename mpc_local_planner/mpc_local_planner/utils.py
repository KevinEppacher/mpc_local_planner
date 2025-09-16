#!/usr/bin/env python3
from geometry_msgs.msg import Pose
import tf_transformations

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

def _lerp(a, b, t): 
    return a + (b - a) * t

def _inferno_rgb01(x: float):
    """
    x in [0,1] -> (r,g,b) in [0,1], kleine LUT + linearer Blend.
    Werte grob aus matplotlib.inferno gesampelt.
    """
    x = max(0.0, min(1.0, float(x)))
    # (x, r, g, b)
    LUT = [
        (0.00, 0.001, 0.000, 0.014),
        (0.20, 0.071, 0.044, 0.240),
        (0.40, 0.239, 0.111, 0.345),
        (0.60, 0.500, 0.177, 0.327),
        (0.80, 0.798, 0.279, 0.200),
        (1.00, 0.988, 0.998, 0.645),
    ]
    for i in range(len(LUT)-1):
        x0,r0,g0,b0 = LUT[i]
        x1,r1,g1,b1 = LUT[i+1]
        if x <= x1:
            t = 0.0 if x1==x0 else (x - x0) / (x1 - x0)
            r = _lerp(r0, r1, t)
            g = _lerp(g0, g1, t)
            b = _lerp(b0, b1, t)
            return r, g, b
    return LUT[-1][1], LUT[-1][2], LUT[-1][3]
