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
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker


def make_detection(x=0.0, y=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5, timestamp=0.0):
    return DetectedObject3D(
        header=Header(frame_id="map", timestamp=timestamp),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(yaw),
        ),
        geometric_info=GeometricInfo(dimensions=(length, width, height)),
        object_info=ObjectInfo(),
    )


def test_initialize_state():
    tracker = VehicleTracker()
    det = make_detection(x=3.0, y=5.0, yaw=0.5, length=4.5, width=2.1, height=1.6)
    state = tracker.initialize_state(det)

    (pos_state, pos_cov), (shape_state, shape_cov) = state

    assert pos_state[0] == pytest.approx(3.0)
    assert pos_state[1] == pytest.approx(5.0)
    assert pos_state[2] == pytest.approx(0.5, abs=1e-5)
    assert pos_state[3] == pytest.approx(0.0)  # v=0 で初期化
    assert pos_state[4] == pytest.approx(0.0)  # yaw_rate=0 で初期化
    assert pos_cov.shape == (5, 5)

    assert shape_state[0] == pytest.approx(4.5)
    assert shape_state[1] == pytest.approx(2.1)
    assert shape_state[2] == pytest.approx(1.6)
    assert shape_cov.shape == (3, 3)


def test_predict_straight_line():
    """直進 (yaw_rate=0, v=5) での予測: x が v*dt だけ増加する。"""
    tracker = VehicleTracker()
    det = make_detection(x=0.0, y=0.0, yaw=0.0)
    state = tracker.initialize_state(det)

    # v=5 を手動で設定
    (pos_state, pos_cov), shape = state
    pos_state[3] = 5.0
    state = (pos_state, pos_cov), shape

    dt = 0.1
    new_state = tracker.predict(state, dt)
    (pos_pred, _), _ = new_state

    assert pos_pred[0] == pytest.approx(0.5, abs=1e-5)  # x += v*dt
    assert pos_pred[1] == pytest.approx(0.0, abs=1e-5)  # y 変化なし


def test_predict_curved():
    """旋回 (yaw_rate != 0) での予測: CTRV モデルの弧状移動。"""
    tracker = VehicleTracker()
    det = make_detection(x=0.0, y=0.0, yaw=0.0)
    state = tracker.initialize_state(det)

    (pos_state, pos_cov), shape = state
    pos_state[3] = 10.0   # v
    pos_state[4] = 0.1    # yaw_rate
    state = (pos_state, pos_cov), shape

    dt = 0.5
    new_state = tracker.predict(state, dt)
    (pos_pred, _), _ = new_state

    yaw_new = pos_pred[2]
    assert yaw_new == pytest.approx(0.05, abs=1e-5)  # yaw = yaw_rate * dt
    # x, y は直線移動より大きい/小さいはず (弧を描く)
    assert pos_pred[0] > 0.0
    assert pos_pred[1] > 0.0  # 左旋回で y 方向にずれる


def test_update_corrects_position():
    """更新後の状態は予測値と観測値の中間に収まるはず (Kalman 補正)。"""
    tracker = VehicleTracker()
    det_init = make_detection(x=0.0, y=0.0, yaw=0.0)
    state = tracker.initialize_state(det_init)

    # x=0 から x=1 方向に観測
    det_update = make_detection(x=1.0, y=0.0, yaw=0.0)
    updated_state = tracker.update(state, det_update)
    (pos_upd, _), _ = updated_state

    assert 0.0 < pos_upd[0] < 1.0


def test_to_kinematic_state_position():
    """to_kinematic_state が正しい位置・姿勢を返す。"""
    tracker = VehicleTracker()
    det = make_detection(x=2.0, y=3.0, yaw=math.pi / 4)
    state = tracker.initialize_state(det)

    kin = tracker.to_kinematic_state(state)

    assert kin.position[0] == pytest.approx(2.0)
    assert kin.position[1] == pytest.approx(3.0)
    assert kin.get_yaw() == pytest.approx(math.pi / 4, abs=1e-5)


def test_to_kinematic_state_velocity_direction():
    """v=10, yaw=0 なら velocity が (10, 0, 0) になるはず。"""
    tracker = VehicleTracker()
    det = make_detection(x=0.0, y=0.0, yaw=0.0)
    state = tracker.initialize_state(det)

    (pos_state, pos_cov), shape = state
    pos_state[3] = 10.0  # v
    state = (pos_state, pos_cov), shape

    kin = tracker.to_kinematic_state(state)
    assert kin.velocity[0] == pytest.approx(10.0, abs=1e-5)
    assert kin.velocity[1] == pytest.approx(0.0, abs=1e-5)


def test_to_kinematic_state_has_position_covariance():
    """to_kinematic_state が position_covariance を返す (Mahalanobis コスト用)。"""
    tracker = VehicleTracker()
    state = tracker.initialize_state(make_detection())
    kin = tracker.to_kinematic_state(state)

    assert kin.position_covariance is not None
    assert len(kin.position_covariance) == 4  # 2x2 の flat tuple
