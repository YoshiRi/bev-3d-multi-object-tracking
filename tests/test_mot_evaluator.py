import math

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
from bev_3d_multi_object_tracking.evaluation.mot_evaluator import (
    FrameResult,
    MOTEvaluator,
    MOTMetrics,
)
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def make_gt(object_id: str, x: float, y: float) -> DetectedObject3D:
    return DetectedObject3D(
        header=Header(frame_id="map", timestamp=0.0),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(0.0),
        ),
        geometric_info=GeometricInfo(dimensions=(4.0, 2.0, 1.5)),
        object_info=ObjectInfo(object_id=object_id),
    )


def make_tracklet(track_id: str, x: float, y: float) -> Tracklet:
    det = DetectedObject3D(
        header=Header(frame_id="map", timestamp=0.0),
        kinematic_state=KinematicState(
            position=(x, y, 0.0),
            orientation=yaw_to_quaternion(0.0),
        ),
        geometric_info=GeometricInfo(dimensions=(4.0, 2.0, 1.5)),
        object_info=ObjectInfo(),
    )
    return Tracklet(
        filter_instance=VehicleTracker(),
        initial_detection=det,
        track_id=track_id,
    )


# ==================================================================
# FrameResult の基本
# ==================================================================

class TestFrameResult:

    def test_n_gt(self):
        r = FrameResult(tp=3, fn=2)
        assert r.n_gt == 5

    def test_n_tracks(self):
        r = FrameResult(tp=3, fp=1)
        assert r.n_tracks == 4


# ==================================================================
# 1フレームの update
# ==================================================================

class TestMOTEvaluatorUpdate:

    def test_empty_gt_and_tracks(self):
        ev = MOTEvaluator()
        result = ev.update([], [])
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 0

    def test_empty_gt_only_tracks(self):
        ev = MOTEvaluator()
        tracks = [make_tracklet("t0", 0.0, 0.0), make_tracklet("t1", 5.0, 0.0)]
        result = ev.update([], tracks)
        assert result.fp == 2
        assert result.tp == 0
        assert result.fn == 0

    def test_empty_tracks_only_gt(self):
        ev = MOTEvaluator()
        gt = [make_gt("g0", 0.0, 0.0), make_gt("g1", 5.0, 0.0)]
        result = ev.update(gt, [])
        assert result.fn == 2
        assert result.tp == 0
        assert result.fp == 0

    def test_perfect_match_one_object(self):
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 0.1, 0.0)]  # 距離 0.1 < threshold
        result = ev.update(gt, tr)
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0

    def test_distance_over_threshold_is_fn_and_fp(self):
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 10.0, 0.0)]  # 距離 10.0 > threshold
        result = ev.update(gt, tr)
        assert result.tp == 0
        assert result.fn == 1
        assert result.fp == 1

    def test_two_objects_both_matched(self):
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0), make_gt("g1", 10.0, 0.0)]
        tr = [make_tracklet("t0", 0.1, 0.0), make_tracklet("t1", 10.1, 0.0)]
        result = ev.update(gt, tr)
        assert result.tp == 2
        assert result.fp == 0
        assert result.fn == 0

    def test_one_gt_one_track_no_match(self):
        ev = MOTEvaluator(match_threshold=1.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 5.0, 0.0)]
        result = ev.update(gt, tr)
        assert result.tp == 0

    def test_match_distance_recorded(self):
        ev = MOTEvaluator(match_threshold=5.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 3.0, 4.0)]  # 距離 = 5.0 ≤ threshold
        result = ev.update(gt, tr)
        assert result.tp == 1
        assert result.total_match_distance == pytest.approx(5.0, abs=1e-4)


# ==================================================================
# ID Switch 検出
# ==================================================================

class TestIDSwitch:

    def test_no_id_switch_same_track(self):
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 0.0, 0.0)]
        ev.update(gt, tr)  # frame 0: g0 → t0
        result = ev.update(gt, tr)  # frame 1: g0 → t0 (same)
        assert result.id_switches == 0

    def test_id_switch_detected(self):
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr_old = [make_tracklet("t0", 0.0, 0.0)]
        tr_new = [make_tracklet("t1", 0.0, 0.0)]  # 別の track_id
        ev.update(gt, tr_old)   # frame 0: g0 → t0
        result = ev.update(gt, tr_new)  # frame 1: g0 → t1 → IDSW
        assert result.id_switches == 1

    def test_no_id_switch_first_appearance(self):
        """初登場では ID Switch は発生しない。"""
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 0.0, 0.0)]
        result = ev.update(gt, tr)
        assert result.id_switches == 0

    def test_id_switch_persists_across_empty_frame(self):
        """
        前フレームで GT が見えなくても prev_gt_to_track は保持される。
        再登場時に別の track に割り当てられれば IDSW としてカウントされる。
        (CLEARMOT の標準挙動)
        """
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr0 = [make_tracklet("t0", 0.0, 0.0)]
        tr1 = [make_tracklet("t1", 0.0, 0.0)]

        ev.update(gt, tr0)   # frame 0: g0 → t0  (prev: {g0: t0})
        ev.update([], [])    # frame 1: 空フレーム (prev は {g0: t0} のまま保持)
        result = ev.update(gt, tr1)  # frame 2: g0 → t1 → IDSW (t0 から t1 に変化)
        assert result.id_switches == 1


# ==================================================================
# 複数フレーム集計と compute()
# ==================================================================

class TestMOTMetricsCompute:

    def test_perfect_tracking_mota_is_one(self):
        """GT と Track が完全一致ならば MOTA = 1.0。"""
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 0.0, 0.0), make_gt("g1", 10.0, 0.0)]
        tr = [make_tracklet("t0", 0.0, 0.0), make_tracklet("t1", 10.0, 0.0)]
        for _ in range(5):
            ev.update(gt, tr)
        metrics = ev.compute()
        assert metrics.mota == pytest.approx(1.0)

    def test_all_fp_mota(self):
        """GT ゼロでトラックだけ → MOTA = 1.0 (GT=0 の定義)。"""
        ev = MOTEvaluator()
        ev.update([], [make_tracklet("t0", 0.0, 0.0)])
        metrics = ev.compute()
        assert metrics.mota == pytest.approx(1.0)

    def test_all_fn_mota(self):
        """トラックゼロで GT だけ → MOTA = 1 - FN/GT = 0.0。"""
        ev = MOTEvaluator()
        ev.update([make_gt("g0", 0.0, 0.0)], [])
        metrics = ev.compute()
        assert metrics.mota == pytest.approx(0.0)

    def test_motp_perfect_match(self):
        """完全一致 (距離 0) ならば MOTP = 0.0。"""
        ev = MOTEvaluator(match_threshold=2.0)
        gt = [make_gt("g0", 5.0, 5.0)]
        tr = [make_tracklet("t0", 5.0, 5.0)]
        for _ in range(3):
            ev.update(gt, tr)
        metrics = ev.compute()
        assert metrics.motp == pytest.approx(0.0, abs=1e-4)

    def test_motp_known_distance(self):
        """既知の距離でフレームを積み上げた MOTP を確認。"""
        ev = MOTEvaluator(match_threshold=5.0)
        gt = [make_gt("g0", 0.0, 0.0)]
        tr = [make_tracklet("t0", 3.0, 4.0)]  # 距離 5.0
        ev.update(gt, tr)
        ev.update(gt, tr)  # 2フレーム: 合計距離 10.0, matches 2 → MOTP = 5.0
        metrics = ev.compute()
        assert metrics.motp == pytest.approx(5.0, abs=1e-4)

    def test_id_switch_reduces_mota(self):
        """IDSW は MOTA を悪化させる。"""
        ev_clean = MOTEvaluator(match_threshold=2.0)
        ev_idsw = MOTEvaluator(match_threshold=2.0)

        gt = [make_gt("g0", 0.0, 0.0)]
        tr0 = [make_tracklet("t0", 0.0, 0.0)]
        tr1 = [make_tracklet("t1", 0.0, 0.0)]

        for _ in range(5):
            ev_clean.update(gt, tr0)
        ev_idsw.update(gt, tr0)
        ev_idsw.update(gt, tr1)  # IDSW
        for _ in range(3):
            ev_idsw.update(gt, tr1)

        m_clean = ev_clean.compute()
        m_idsw = ev_idsw.compute()
        assert m_idsw.mota < m_clean.mota
        assert m_idsw.id_switches == 1

    def test_precision_and_recall(self):
        ev = MOTEvaluator(match_threshold=2.0)
        # 2 GT, 2 Track だが 1 つだけマッチ (1 FP, 1 FN)
        gt = [make_gt("g0", 0.0, 0.0), make_gt("g1", 50.0, 0.0)]
        tr = [make_tracklet("t0", 0.0, 0.0), make_tracklet("t1", 100.0, 0.0)]
        ev.update(gt, tr)
        m = ev.compute()
        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 1
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)

    def test_n_frames(self):
        ev = MOTEvaluator()
        for _ in range(7):
            ev.update([], [])
        assert ev.compute().n_frames == 7

    def test_reset_clears_state(self):
        ev = MOTEvaluator(match_threshold=2.0)
        ev.update([make_gt("g0", 0.0, 0.0)], [make_tracklet("t0", 0.0, 0.0)])
        ev.reset()
        metrics = ev.compute()
        assert metrics.n_frames == 0
        assert metrics.tp == 0


# ==================================================================
# MOTMetrics.__str__
# ==================================================================

def test_metrics_str_contains_key_fields():
    ev = MOTEvaluator(match_threshold=2.0)
    gt = [make_gt("g0", 0.0, 0.0)]
    tr = [make_tracklet("t0", 0.0, 0.0)]
    for _ in range(3):
        ev.update(gt, tr)
    s = str(ev.compute())
    assert "MOTA" in s
    assert "MOTP" in s
    assert "ID Switch" in s
    assert "Precision" in s
    assert "Recall" in s


# ==================================================================
# 合成データとの統合テスト
# ==================================================================

def test_evaluator_with_synthetic_scenario():
    """合成シナリオで評価パイプライン全体が動くことを確認。"""
    from bev_3d_multi_object_tracking.cost_functions import euclidean_distance
    from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
        HungarianCostMatrixAssociation,
    )
    from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
    from bev_3d_multi_object_tracking.io.dict_parser import DictParser
    from bev_3d_multi_object_tracking.multi_object_tracker import (
        MultiObjectTracker,
        TrackingConfig,
    )
    from scenarios.synthetic_generator import create_highway_scenario

    scenario = create_highway_scenario(n_frames=50, seed=0)
    parser = DictParser()
    tracker = MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=HungarianCostMatrixAssociation(euclidean_distance, max_distance=10.0),
        config=TrackingConfig(min_hits_to_confirm=3, max_misses_to_lose=5),
    )
    evaluator = MOTEvaluator(match_threshold=3.0)

    for _, dets, gt_dicts in scenario.iter_frames():
        gt_detections = parser.parse(gt_dicts) if gt_dicts else []
        tracks = tracker.update(parser.parse(dets) if dets else [], dt=scenario.dt)
        evaluator.update(gt_detections, tracks)

    metrics = evaluator.compute()
    assert metrics.n_frames == 50
    assert metrics.total_gt > 0
    # ノイズあり・perfect detection rate なので MOTA は高いはず
    assert metrics.mota > 0.5
    assert metrics.motp >= 0.0
    assert metrics.id_switches >= 0
