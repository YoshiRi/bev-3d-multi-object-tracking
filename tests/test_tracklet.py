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
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker


def make_detection(x=0.0, y=0.0, yaw=0.0, timestamp=0.0, existence_prob=1.0):
    return DetectedObject3D(
        header=Header(frame_id="map", timestamp=timestamp),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(yaw),
        ),
        geometric_info=GeometricInfo(dimensions=(4.0, 2.0, 1.5)),
        object_info=ObjectInfo(existence_probability=existence_prob),
    )


def make_tracklet(x=0.0, y=0.0) -> Tracklet:
    return Tracklet(filter_instance=VehicleTracker(), initial_detection=make_detection(x, y))


# ------------------------------------------------------------------
# 初期状態
# ------------------------------------------------------------------

def test_initial_counters():
    t = make_tracklet()
    assert t.age == 1
    assert t.missed_count == 0
    assert t.is_confirmed is False
    assert t.is_lost is False


def test_initial_kinematic_position():
    t = make_tracklet(x=3.0, y=5.0)
    pos = t.get_current_kinematic().position
    assert pos[0] == pytest.approx(3.0)
    assert pos[1] == pytest.approx(5.0)


# ------------------------------------------------------------------
# predict
# ------------------------------------------------------------------

def test_predict_increments_missed_count():
    t = make_tracklet()
    t.predict(0.1)
    t.predict(0.1)
    t.predict(0.1)
    assert t.missed_count == 3


def test_predict_does_not_change_age():
    t = make_tracklet()
    t.predict(0.1)
    assert t.age == 1  # predict は age を変えない


def test_predict_does_not_set_lost():
    """lost の判断は Tracklet 外 (トラッカー) が行う。predict だけでは lost にならない。"""
    t = make_tracklet()
    for _ in range(100):
        t.predict(0.1)
    assert t.is_lost is False


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------

def test_update_increments_age():
    t = make_tracklet()
    t.update(make_detection(x=0.1))
    assert t.age == 2


def test_update_resets_missed_count():
    t = make_tracklet()
    t.predict(0.1)
    t.predict(0.1)
    assert t.missed_count == 2
    t.update(make_detection())
    assert t.missed_count == 0


def test_update_does_not_auto_confirm():
    """confirmed の判断は Tracklet 外 (トラッカー) が行う。"""
    t = make_tracklet()
    for _ in range(10):
        t.update(make_detection())
    assert t.is_confirmed is False


# ------------------------------------------------------------------
# filter_state の分離確認
# ------------------------------------------------------------------

def test_filter_state_is_not_kinematic_state():
    """VehicleTracker 使用時、_filter_state は KinematicState ではない (opaque tuple)。"""
    t = make_tracklet()
    assert not isinstance(t._filter_state, KinematicState)


def test_kinematic_state_accessible_after_predict():
    t = make_tracklet(x=0.0, y=0.0)
    t.predict(0.1)
    kin = t.get_current_kinematic()
    assert kin is not None
    assert isinstance(kin, KinematicState)


# ------------------------------------------------------------------
# mark_confirmed / mark_lost
# ------------------------------------------------------------------

def test_mark_confirmed():
    t = make_tracklet()
    assert t.is_confirmed is False
    t.mark_confirmed()
    assert t.is_confirmed is True


def test_mark_lost():
    t = make_tracklet()
    assert t.is_lost is False
    t.mark_lost()
    assert t.is_lost is True
