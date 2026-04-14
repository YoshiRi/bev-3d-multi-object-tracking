import math

import numpy as np
import pytest

from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.geometry_utils import yaw_to_quaternion
from bev_3d_multi_object_tracking.core.tracklet import Tracklet
from bev_3d_multi_object_tracking.cost_functions import (
    bev_iou_cost,
    euclidean_distance,
    mahalanobis_distance,
)
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
from bev_3d_multi_object_tracking.io.dict_parser import DictParser


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def make_detection(x=0.0, y=0.0, length=4.0, width=2.0):
    return DetectedObject3D(
        header=Header(frame_id="map", timestamp=0.0),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(0.0),
        ),
        geometric_info=GeometricInfo(dimensions=(length, width, 1.5)),
        object_info=ObjectInfo(),
    )


def make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0) -> Tracklet:
    det = make_detection(x=x, y=y, length=length, width=width)
    return Tracklet(filter_instance=VehicleTracker(), initial_detection=det)


# ==================================================================
# euclidean_distance
# ==================================================================

class TestEuclideanDistance:

    def test_same_position_is_zero(self):
        t = make_tracklet(x=3.0, y=4.0)
        d = make_detection(x=3.0, y=4.0)
        assert euclidean_distance(t, d) == pytest.approx(0.0, abs=1e-4)

    def test_known_distance(self):
        """(0,0) と (3,4) の距離は 5.0。"""
        t = make_tracklet(x=0.0, y=0.0)
        d = make_detection(x=3.0, y=4.0)
        assert euclidean_distance(t, d) == pytest.approx(5.0, abs=1e-4)

    def test_symmetry(self):
        """距離は対称。"""
        t = make_tracklet(x=1.0, y=2.0)
        d = make_detection(x=5.0, y=6.0)
        dist = euclidean_distance(t, d)
        # 逆方向でも同じになるはず (Tracklet の x/y を手動で確認)
        assert dist == pytest.approx(math.sqrt(32.0), abs=1e-3)

    def test_non_negative(self):
        t = make_tracklet(x=-10.0, y=5.0)
        d = make_detection(x=3.0, y=-2.0)
        assert euclidean_distance(t, d) >= 0.0


# ==================================================================
# mahalanobis_distance
# ==================================================================

class TestMahalanobisDistance:

    def test_falls_back_to_euclidean_when_no_covariance(self):
        """position_covariance が None のとき ユークリッド距離と一致する。"""
        t = make_tracklet(x=0.0, y=0.0)
        d = make_detection(x=3.0, y=4.0)
        # KinematicState の position_covariance を None に上書き
        t._kinematic_state.position_covariance = None
        mah = mahalanobis_distance(t, d)
        euc = euclidean_distance(t, d)
        assert mah == pytest.approx(euc, abs=1e-4)

    def test_same_position_is_zero(self):
        t = make_tracklet(x=2.0, y=3.0)
        d = make_detection(x=2.0, y=3.0)
        assert mahalanobis_distance(t, d) == pytest.approx(0.0, abs=1e-4)

    def test_isotropic_covariance_equals_scaled_euclidean(self):
        """等方的な共分散行列 σ²I のとき、Mahalanobis = euclidean / σ。"""
        t = make_tracklet(x=0.0, y=0.0)
        d = make_detection(x=3.0, y=0.0)
        sigma_sq = 4.0  # σ² = 4, σ = 2
        # 2x2 単位行列 × σ²
        cov = (sigma_sq, 0.0, 0.0, sigma_sq)
        t._kinematic_state.position_covariance = cov
        mah = mahalanobis_distance(t, d)
        # Mahalanobis = sqrt( diff^T * (σ²I)^-1 * diff ) = sqrt(3²/4) = 1.5
        assert mah == pytest.approx(1.5, abs=1e-4)

    def test_non_negative(self):
        t = make_tracklet(x=1.0, y=1.0)
        d = make_detection(x=5.0, y=5.0)
        assert mahalanobis_distance(t, d) >= 0.0

    def test_with_vt_covariance(self):
        """VehicleTracker の covariance が付いた Tracklet でも動作する。"""
        t = make_tracklet(x=0.0, y=0.0)
        d = make_detection(x=1.0, y=0.0)
        # VehicleTracker は position_covariance を設定しているので fallback しない
        assert t.get_current_kinematic().position_covariance is not None
        dist = mahalanobis_distance(t, d)
        assert dist >= 0.0
        assert math.isfinite(dist)


# ==================================================================
# bev_iou_cost
# ==================================================================

class TestBevIouCost:

    def test_identical_box_cost_is_zero(self):
        """完全一致 → IoU = 1.0 → cost = 0.0。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        d = make_detection(x=0.0, y=0.0, length=4.0, width=2.0)
        assert bev_iou_cost(t, d) == pytest.approx(0.0, abs=1e-6)

    def test_no_overlap_cost_is_one(self):
        """完全に離れている → IoU = 0.0 → cost = 1.0。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        d = make_detection(x=100.0, y=100.0, length=4.0, width=2.0)
        assert bev_iou_cost(t, d) == pytest.approx(1.0, abs=1e-6)

    def test_partial_overlap_between_zero_and_one(self):
        """部分重なり → 0.0 < cost < 1.0。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        d = make_detection(x=2.0, y=0.0, length=4.0, width=2.0)  # 半分重なる
        cost = bev_iou_cost(t, d)
        assert 0.0 < cost < 1.0

    def test_cost_range(self):
        """cost は常に [0, 1] の範囲に収まる。"""
        positions = [(0, 0), (1, 0), (3, 0), (10, 0)]
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        for dx, dy in positions:
            d = make_detection(x=dx, y=dy, length=4.0, width=2.0)
            cost = bev_iou_cost(t, d)
            assert 0.0 <= cost <= 1.0

    def test_symmetric(self):
        """IoU は対称なので cost も対称。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        d = make_detection(x=1.0, y=0.0, length=4.0, width=2.0)
        cost_fwd = bev_iou_cost(t, d)

        # 逆向き (別の Tracklet で x=1 のものと detection x=0 を比較)
        t2 = make_tracklet(x=1.0, y=0.0, length=4.0, width=2.0)
        d2 = make_detection(x=0.0, y=0.0, length=4.0, width=2.0)
        cost_bwd = bev_iou_cost(t2, d2)
        assert cost_fwd == pytest.approx(cost_bwd, abs=1e-6)

    def test_different_sizes(self):
        """サイズが異なる場合も IoU は計算できる。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        d = make_detection(x=0.0, y=0.0, length=2.0, width=1.0)  # 小さいボックス
        cost = bev_iou_cost(t, d)
        # 小さい方が大きい方に完全に包含される
        # inter = 2*1 = 2, union = 4*2 + 2*1 - 2 = 8, IoU = 2/8 = 0.25
        assert cost == pytest.approx(0.75, abs=1e-6)

    def test_adjacent_boxes_no_overlap(self):
        """ぴったり隣接 (接触のみ) → 重複面積ゼロ → cost = 1.0。"""
        t = make_tracklet(x=0.0, y=0.0, length=4.0, width=2.0)
        # t の右端は x=2.0, d の左端も x=2.0 → inter_x = 0
        d = make_detection(x=4.0, y=0.0, length=4.0, width=2.0)
        assert bev_iou_cost(t, d) == pytest.approx(1.0, abs=1e-6)
