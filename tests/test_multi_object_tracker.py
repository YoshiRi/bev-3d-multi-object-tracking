import pytest

from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.geometry_utils import yaw_to_quaternion
from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
    HungarianCostMatrixAssociation,
)
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
from bev_3d_multi_object_tracking.multi_object_tracker import MultiObjectTracker, TrackingConfig


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def make_detection(x=0.0, y=0.0, timestamp=0.0):
    return DetectedObject3D(
        header=Header(frame_id="map", timestamp=timestamp),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(0.0),
        ),
        geometric_info=GeometricInfo(dimensions=(4.0, 2.0, 1.5)),
        object_info=ObjectInfo(),
    )


def euclidean_cost(tracklet, detection):
    tp = tracklet.get_current_kinematic().position
    dp = detection.get_position()
    return ((tp[0] - dp[0]) ** 2 + (tp[1] - dp[1]) ** 2) ** 0.5


def make_tracker(min_hits=3, max_misses=3):
    cfg = TrackingConfig(min_hits_to_confirm=min_hits, max_misses_to_lose=max_misses)
    association = HungarianCostMatrixAssociation(
        cost_func=euclidean_cost, max_distance=5.0
    )
    return MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=association,
        config=cfg,
    )


# ------------------------------------------------------------------
# 基本動作
# ------------------------------------------------------------------

def test_spawn_track_on_first_detection():
    tracker = make_tracker()
    tracks = tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    assert len(tracks) == 1


def test_track_persists_across_frames():
    """同じ位置付近の検出が続けばトラックは1本のまま。"""
    tracker = make_tracker()
    tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    tracker.update([make_detection(0.05, 0.0)], dt=0.1)
    tracks = tracker.update([make_detection(0.1, 0.0)], dt=0.1)
    assert len(tracks) == 1


def test_two_separate_objects_tracked():
    """離れた2物体は別々のトラックに割り当てられる。"""
    tracker = make_tracker()
    dets = [make_detection(0.0, 0.0), make_detection(50.0, 0.0)]
    tracker.update(dets, dt=0.1)
    tracks = tracker.update(dets, dt=0.1)
    assert len(tracks) == 2
    ids = {t.get_track_id() for t in tracks}
    assert len(ids) == 2


# ------------------------------------------------------------------
# ライフサイクル (TrackingConfig 経由で MultiObjectTracker が制御)
# ------------------------------------------------------------------

def test_track_confirmed_after_min_hits():
    tracker = make_tracker(min_hits=3)
    det = make_detection(0.0, 0.0)
    for _ in range(3):
        tracker.update([det], dt=0.1)
    confirmed = tracker.get_confirmed_tracks()
    assert len(confirmed) == 1


def test_track_not_confirmed_before_min_hits():
    tracker = make_tracker(min_hits=5)
    det = make_detection(0.0, 0.0)
    for _ in range(4):
        tracker.update([det], dt=0.1)
    assert len(tracker.get_confirmed_tracks()) == 0


def test_track_lost_after_max_misses():
    tracker = make_tracker(max_misses=3)
    # トラックを生成
    tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    # 検出なしのフレームを繰り返す
    for _ in range(3):
        tracker.update([], dt=0.1)
    # lost として除去されているはず
    assert len(tracker.get_all_tracks()) == 0


def test_track_not_lost_before_max_misses():
    tracker = make_tracker(max_misses=5)
    tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    for _ in range(4):
        tracker.update([], dt=0.1)
    assert len(tracker.get_all_tracks()) == 1


def test_new_track_after_disappearance():
    """消滅後に同じ位置に再検出 → 新規トラックが生成される (track_id が異なる)。"""
    tracker = make_tracker(max_misses=2)
    tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    old_id = tracker.get_all_tracks()[0].get_track_id()

    # 消滅させる
    for _ in range(3):
        tracker.update([], dt=0.1)
    assert len(tracker.get_all_tracks()) == 0

    # 再登場
    tracker.update([make_detection(0.0, 0.0)], dt=0.1)
    new_tracks = tracker.get_all_tracks()
    assert len(new_tracks) == 1
    assert new_tracks[0].get_track_id() != old_id


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset_clears_all_tracks():
    tracker = make_tracker()
    tracker.update([make_detection(0.0, 0.0), make_detection(10.0, 0.0)], dt=0.1)
    assert len(tracker.get_all_tracks()) == 2
    tracker.reset()
    assert len(tracker.get_all_tracks()) == 0
