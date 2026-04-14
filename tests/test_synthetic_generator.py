import math

import pytest

from scenarios.synthetic_generator import (
    ObjectSpec,
    SyntheticScenario,
    create_highway_scenario,
    create_intersection_scenario,
    create_occlusion_scenario,
)


# ------------------------------------------------------------------
# ObjectSpec のデフォルト値
# ------------------------------------------------------------------

def test_object_spec_defaults():
    spec = ObjectSpec(object_id="test")
    assert spec.v == 0.0
    assert spec.yaw_rate == 0.0
    assert spec.start_frame == 0
    assert spec.end_frame is None
    assert spec.detection_rate == 1.0


# ------------------------------------------------------------------
# トラジェクトリ計算
# ------------------------------------------------------------------

def test_straight_line_trajectory():
    """v=10, yaw=0 で直進 → x が v*dt*frame 増加する。"""
    scenario = SyntheticScenario(
        n_frames=10, dt=0.1,
        objects=[ObjectSpec(object_id="car", x0=0.0, y0=0.0, yaw0=0.0, v=10.0)]
    )
    gt_frame0 = scenario.get_ground_truth(0)
    gt_frame5 = scenario.get_ground_truth(5)

    assert gt_frame0[0]["x"] == pytest.approx(0.0)
    assert gt_frame5[0]["x"] == pytest.approx(5.0, abs=1e-5)  # 10 * 0.1 * 5
    assert gt_frame5[0]["y"] == pytest.approx(0.0, abs=1e-5)


def test_ctrv_trajectory_yaw_advances():
    """yaw_rate != 0 のとき yaw が進む。"""
    scenario = SyntheticScenario(
        n_frames=5, dt=0.1,
        objects=[ObjectSpec(object_id="car", yaw0=0.0, v=10.0, yaw_rate=0.5)]
    )
    gt = scenario.get_ground_truth(4)
    assert gt[0]["yaw"] == pytest.approx(0.5 * 0.1 * 4, abs=1e-4)


def test_object_absent_before_start_frame():
    scenario = SyntheticScenario(
        n_frames=10, dt=0.1,
        objects=[ObjectSpec(object_id="car", start_frame=5)]
    )
    assert scenario.get_ground_truth(0) == []
    assert scenario.get_ground_truth(4) == []
    assert len(scenario.get_ground_truth(5)) == 1


def test_object_absent_after_end_frame():
    scenario = SyntheticScenario(
        n_frames=10, dt=0.1,
        objects=[ObjectSpec(object_id="car", end_frame=3)]
    )
    assert len(scenario.get_ground_truth(3)) == 1
    assert scenario.get_ground_truth(4) == []


# ------------------------------------------------------------------
# 検出ノイズ
# ------------------------------------------------------------------

def test_detections_have_noise(seed=0):
    """検出値は真値と一致しない (ノイズが乗る)。"""
    scenario = SyntheticScenario(
        n_frames=20, dt=0.1, seed=0,
        objects=[ObjectSpec(object_id="car", x0=0.0, y0=0.0, pos_noise_std=1.0)]
    )
    # 複数フレームの検出値が真値と完全一致することはほぼない
    diffs = []
    for f in range(20):
        gt = scenario.get_ground_truth(f)
        det = scenario.get_detections(f)
        if gt and det:
            diffs.append(abs(gt[0]["x"] - det[0]["x"]))
    assert max(diffs) > 0.0


def test_detection_rate_zero_produces_no_detections():
    scenario = SyntheticScenario(
        n_frames=10, dt=0.1, seed=0,
        objects=[ObjectSpec(object_id="car", detection_rate=0.0)]
    )
    for f in range(10):
        assert scenario.get_detections(f) == []


def test_detection_rate_one_always_detects():
    scenario = SyntheticScenario(
        n_frames=10, dt=0.1, seed=0,
        objects=[ObjectSpec(object_id="car", detection_rate=1.0)]
    )
    for f in range(10):
        assert len(scenario.get_detections(f)) == 1


# ------------------------------------------------------------------
# 複数物体・iter_frames
# ------------------------------------------------------------------

def test_multiple_objects():
    scenario = SyntheticScenario(
        n_frames=5, dt=0.1,
        objects=[
            ObjectSpec(object_id="car_0", x0=0.0),
            ObjectSpec(object_id="car_1", x0=10.0),
        ]
    )
    gt = scenario.get_ground_truth(0)
    assert len(gt) == 2


def test_iter_frames_yields_all_frames():
    scenario = SyntheticScenario(
        n_frames=7, dt=0.1,
        objects=[ObjectSpec(object_id="car")]
    )
    frames = list(scenario.iter_frames())
    assert len(frames) == 7
    assert frames[0][0] == 0
    assert frames[6][0] == 6


def test_iter_frames_structure():
    scenario = SyntheticScenario(
        n_frames=3, dt=0.1,
        objects=[ObjectSpec(object_id="car")]
    )
    for frame_idx, detections, ground_truth in scenario.iter_frames():
        assert isinstance(frame_idx, int)
        assert isinstance(detections, list)
        assert isinstance(ground_truth, list)


# ------------------------------------------------------------------
# 定義済みシナリオ
# ------------------------------------------------------------------

def test_highway_scenario_runs():
    scenario = create_highway_scenario(n_frames=50)
    assert scenario.n_frames == 50
    # フレーム0では car_0, car_1 が存在 (car_2 は start_frame=30)
    gt_0 = scenario.get_ground_truth(0)
    ids_0 = {d["object_id"] for d in gt_0}
    assert "car_0" in ids_0
    assert "car_1" in ids_0
    assert "car_2" not in ids_0

    gt_30 = scenario.get_ground_truth(30)
    ids_30 = {d["object_id"] for d in gt_30}
    assert "car_2" in ids_30


def test_intersection_scenario_runs():
    scenario = create_intersection_scenario(n_frames=80)
    assert scenario.n_frames == 80
    gt = scenario.get_ground_truth(0)
    assert len(gt) == 3


def test_occlusion_scenario_detection_rate():
    """occlusion シナリオでは一定数の miss が発生する。"""
    scenario = create_occlusion_scenario(n_frames=100, seed=1)
    miss_count = sum(
        1 for f in range(100) if len(scenario.get_detections(f)) < 2
    )
    assert miss_count > 10  # かなりの miss が発生するはず


# ------------------------------------------------------------------
# tracker との統合確認
# ------------------------------------------------------------------

def test_tracker_runs_with_synthetic_data():
    """合成データをそのまま DictParser + MultiObjectTracker に流せることを確認。"""
    from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
        HungarianCostMatrixAssociation,
    )
    from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
    from bev_3d_multi_object_tracking.io.dict_parser import DictParser
    from bev_3d_multi_object_tracking.multi_object_tracker import (
        MultiObjectTracker,
        TrackingConfig,
    )
    from bev_3d_multi_object_tracking.cost_functions import euclidean_distance

    parser = DictParser()
    tracker = MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=HungarianCostMatrixAssociation(euclidean_distance, max_distance=10.0),
        config=TrackingConfig(min_hits_to_confirm=3, max_misses_to_lose=5),
    )

    scenario = create_highway_scenario(n_frames=30, seed=0)
    for frame_idx, raw_dets, _ in scenario.iter_frames():
        detections = parser.parse(raw_dets)
        tracker.update(detections, dt=scenario.dt)

    confirmed = tracker.get_confirmed_tracks()
    assert len(confirmed) >= 2  # car_0, car_1 は確定されているはず
