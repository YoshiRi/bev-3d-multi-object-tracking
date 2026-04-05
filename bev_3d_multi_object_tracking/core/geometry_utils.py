import math
from typing import Tuple


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """yaw角 (rad) をクォータニオン (x, y, z, w) に変換する。roll/pitch はゼロ前提。"""
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(quaternion: Tuple[float, float, float, float]) -> float:
    """クォータニオン (x, y, z, w) から yaw角 (rad) を取り出す。"""
    x, y, z, w = quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
