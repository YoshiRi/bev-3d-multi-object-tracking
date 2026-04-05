import math
from pathlib import Path

import pytest

from bev_3d_multi_object_tracking.io.kitti_parser import KittiTrackingParser

FIXTURE = Path(__file__).parent / "fixtures" / "kitti_sample.txt"


# ------------------------------------------------------------------
# 基本読み込み
# ------------------------------------------------------------------

def test_n_frames():
    """フィクスチャは frame 0,1,2 の 3フレーム。"""
    parser = KittiTrackingParser(FIXTURE)
    assert parser.n_frames == 3


def test_frame0_has_two_objects():
    parser = KittiTrackingParser(FIXTURE)
    dets = parser.parse(0)
    assert len(dets) == 2


def test_frame1_has_three_objects():
    """frame 1 には Car x2 + Pedestrian x1。"""
    parser = KittiTrackingParser(FIXTURE)
    dets = parser.parse(1)
    assert len(dets) == 3


def test_frame2_car1_absent():
    """frame 2 では track_id=1 (Car) が消えている → Car x1 + Pedestrian x1。"""
    parser = KittiTrackingParser(FIXTURE)
    dets = parser.parse(2)
    class_names = {d.get_class_name() for d in dets}
    assert "Car" in class_names
    assert "Pedestrian" in class_names
    assert len(dets) == 2


def test_empty_frame_returns_empty_list():
    parser = KittiTrackingParser(FIXTURE)
    dets = parser.parse(99)
    assert dets == []


# ------------------------------------------------------------------
# 座標変換の検証
# ------------------------------------------------------------------
# フィクスチャ frame 0, track 0:
#   x_cam=-6.21, y_cam=1.67, z_cam=20.00, rotation_y=-1.57, h=1.48, w=1.69, l=4.02
#
# 期待値:
#   x_bev = z_cam = 20.00
#   y_bev = -x_cam = 6.21
#   z_bev = -y_cam + h/2 = -1.67 + 0.74 = -0.93
#   yaw_bev = -rotation_y = 1.57  (≈ pi/2)
#   length = l = 4.02
#   width  = w = 1.69
#   height = h = 1.48

def test_position_conversion():
    parser = KittiTrackingParser(FIXTURE)
    det = parser.parse(0)[0]  # frame 0, track 0 (z_cam最大 = 遠い方)

    # x_bev = z_cam = 20.0
    pos = det.get_position()
    assert pos[0] == pytest.approx(20.0, abs=1e-3)
    # y_bev = -x_cam = 6.21
    assert pos[1] == pytest.approx(6.21, abs=1e-3)
    # z_bev = -y_cam + h/2 = -1.67 + 0.74 = -0.93
    assert pos[2] == pytest.approx(-0.93, abs=1e-2)


def test_yaw_conversion():
    """rotation_y = -1.57 → yaw_bev = 1.57 ≈ pi/2。"""
    parser = KittiTrackingParser(FIXTURE)
    det = parser.parse(0)[0]
    yaw = det.get_yaw()
    assert yaw == pytest.approx(math.pi / 2, abs=1e-2)


def test_dimensions_conversion():
    """KITTI (h=1.48, w=1.69, l=4.02) → (length=4.02, width=1.69, height=1.48)。"""
    parser = KittiTrackingParser(FIXTURE)
    det = parser.parse(0)[0]
    dims = det.get_dimensions()
    assert dims[0] == pytest.approx(4.02)   # length = l
    assert dims[1] == pytest.approx(1.69)   # width  = w
    assert dims[2] == pytest.approx(1.48)   # height = h


def test_timestamp_from_frame_number():
    """timestamp = frame_idx * dt。"""
    parser = KittiTrackingParser(FIXTURE, dt=0.1)
    dets_0 = parser.parse(0)
    dets_2 = parser.parse(2)
    assert dets_0[0].get_timestamp() == pytest.approx(0.0)
    assert dets_2[0].get_timestamp() == pytest.approx(0.2)


def test_class_name_preserved():
    parser = KittiTrackingParser(FIXTURE)
    dets_1 = parser.parse(1)
    class_names = {d.get_class_name() for d in dets_1}
    assert "Car" in class_names
    assert "Pedestrian" in class_names


# ------------------------------------------------------------------
# valid_types フィルタリング
# ------------------------------------------------------------------

def test_filter_pedestrian_only():
    """valid_types={"Pedestrian"} にすると Pedestrian だけ返る。"""
    parser = KittiTrackingParser(FIXTURE, valid_types={"Pedestrian"})
    dets_1 = parser.parse(1)
    assert len(dets_1) == 1
    assert dets_1[0].get_class_name() == "Pedestrian"


def test_filter_car_only():
    parser = KittiTrackingParser(FIXTURE, valid_types={"Car"})
    dets_1 = parser.parse(1)
    assert len(dets_1) == 2
    for d in dets_1:
        assert d.get_class_name() == "Car"


# ------------------------------------------------------------------
# iter_frames
# ------------------------------------------------------------------

def test_iter_frames_yields_all():
    parser = KittiTrackingParser(FIXTURE)
    frames = list(parser.iter_frames())
    assert len(frames) == 3
    assert frames[0][0] == 0
    assert frames[2][0] == 2


# ------------------------------------------------------------------
# tracker との統合確認
# ------------------------------------------------------------------

def test_tracker_runs_with_kitti_data():
    """KITTI パーサーの出力をそのままトラッカーに流せることを確認。"""
    from bev_3d_multi_object_tracking.cost_functions import euclidean_distance
    from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
        HungarianCostMatrixAssociation,
    )
    from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
    from bev_3d_multi_object_tracking.multi_object_tracker import (
        MultiObjectTracker,
        TrackingConfig,
    )

    parser = KittiTrackingParser(FIXTURE, valid_types={"Car"})
    tracker = MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=HungarianCostMatrixAssociation(euclidean_distance, max_distance=5.0),
        config=TrackingConfig(min_hits_to_confirm=2, max_misses_to_lose=3),
    )

    for frame_idx, detections in parser.iter_frames():
        tracker.update(detections, dt=0.1)

    # 3フレームしかないので confirmed は少ないが、エラーなく動くことを確認
    all_tracks = tracker.get_all_tracks()
    assert len(all_tracks) >= 0  # クラッシュしないこと
