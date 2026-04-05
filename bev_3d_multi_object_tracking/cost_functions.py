"""
よく使われるコスト関数の実装。

HungarianCostMatrixAssociation / GNNCostMatrixAssociation の
cost_func 引数に渡して使う。

    association = HungarianCostMatrixAssociation(
        cost_func=euclidean_distance,
        max_distance=5.0,
    )
"""

import math
from typing import Optional

import numpy as np

from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


# ------------------------------------------------------------------
# ユークリッド距離
# ------------------------------------------------------------------

def euclidean_distance(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    """BEV 平面上のユークリッド距離 (x-y のみ)。"""
    tp = tracklet.get_current_kinematic().position
    dp = detection.get_position()
    return math.sqrt((tp[0] - dp[0]) ** 2 + (tp[1] - dp[1]) ** 2)


# ------------------------------------------------------------------
# Mahalanobis 距離
# ------------------------------------------------------------------

def mahalanobis_distance(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    """
    x-y 位置の Mahalanobis 距離。

    KinematicState.position_covariance (2x2 の flat tuple) を使って
    予測位置の不確かさを考慮した距離を計算する。

    position_covariance が None の場合はユークリッド距離にフォールバックする。
    """
    kin = tracklet.get_current_kinematic()
    tp = np.array(kin.position[:2])
    dp = np.array(detection.get_position()[:2])
    diff = dp - tp

    cov = kin.position_covariance
    if cov is None:
        return float(np.linalg.norm(diff))

    cov_mat = np.array(cov).reshape(2, 2)
    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(diff))

    dist_sq = float(diff @ cov_inv @ diff)
    return math.sqrt(max(dist_sq, 0.0))


# ------------------------------------------------------------------
# BEV IoU コスト (1 - IoU)
# ------------------------------------------------------------------

def bev_iou_cost(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    """
    BEV 平面上の軸平行バウンディングボックス IoU から計算するコスト (1 - IoU)。

    yaw を無視した軸平行近似を使うため高速だが、
    大きな yaw 差がある物体には誤差が生じる。
    完全に重ならない場合は 1.0 (最大コスト) を返す。

    Args:
        tracklet  : トラック (position, geometric_info を使用)
        detection : 検出 (position, dimensions を使用)

    Returns:
        cost = 1.0 - IoU  (範囲: [0.0, 1.0])
    """
    iou = _axis_aligned_bev_iou(tracklet, detection)
    return 1.0 - iou


def _axis_aligned_bev_iou(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    """軸平行 BEV IoU の計算。"""
    tx, ty, _ = tracklet.get_current_kinematic().position
    tl, tw, _ = tracklet.get_geometric_info().dimensions

    dx, dy, _ = detection.get_position()
    dl, dw, _ = detection.get_dimensions()

    # 各ボックスの半辺 (length=前後, width=左右)
    t_x1, t_x2 = tx - tl / 2, tx + tl / 2
    t_y1, t_y2 = ty - tw / 2, ty + tw / 2
    d_x1, d_x2 = dx - dl / 2, dx + dl / 2
    d_y1, d_y2 = dy - dw / 2, dy + dw / 2

    inter_x = max(0.0, min(t_x2, d_x2) - max(t_x1, d_x1))
    inter_y = max(0.0, min(t_y2, d_y2) - max(t_y1, d_y1))
    inter_area = inter_x * inter_y

    area_t = tl * tw
    area_d = dl * dw
    union_area = area_t + area_d - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area
