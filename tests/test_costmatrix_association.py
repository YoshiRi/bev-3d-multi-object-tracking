import math

import numpy as np
import pytest

from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.tracklet import Tracklet
from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
    CostMatrixBasedAssociation,
    GNNCostMatrixAssociation,
    HungarianCostMatrixAssociation,
)


#
# 1. ダミーFilter
#
class DummyFilter(BaseFilter):
    """テスト用のダミーフィルタ: 状態ベクトルを大きく変えない簡易的な実装。"""

    def initialize_state(self, detection: DetectedObject3D):
        pos = detection.get_position()
        ori = detection.get_orientation()
        return KinematicState(position=pos, orientation=ori)

    def predict(self, current_state, dt: float):
        return current_state

    def update(self, current_state, detection: DetectedObject3D):
        pos = detection.get_position()
        ori = detection.get_orientation()
        return KinematicState(position=pos, orientation=ori)


#
# 2. ダミーのTracklet/Detection作成ヘルパー
#
def create_tracklet(x, y, z=0.0, track_id=None):
    header = Header(frame_id="map", timestamp=0.0)
    obj_info = ObjectInfo(object_id=track_id)
    geom = GeometricInfo(dimensions=(1, 1, 1))
    detection = DetectedObject3D(
        header=header,
        kinematic_state=KinematicState(position=(x, y, z), orientation=(0, 0, 0, 1)),
        geometric_info=geom,
        object_info=obj_info,
    )
    filt = DummyFilter()
    trk = Tracklet(filter_instance=filt, initial_detection=detection, track_id=track_id)
    return trk


def create_detection(x, y, z=0.0, object_id=None):
    header = Header(frame_id="map", timestamp=1.0)
    obj_info = ObjectInfo(object_id=object_id)
    geom = GeometricInfo(dimensions=(1, 1, 1))
    return DetectedObject3D(
        header=header,
        kinematic_state=KinematicState(position=(x, y, z), orientation=(0, 0, 0, 1)),
        geometric_info=geom,
        object_info=obj_info,
    )


#
# 3. コスト関数(ユークリッド距離): distが大きいほどコスト大
#
def euclidean_cost(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    pos_t = tracklet.get_current_kinematic().position
    pos_d = detection.get_position()
    dx = pos_t[0] - pos_d[0]
    dy = pos_t[1] - pos_d[1]
    dz = pos_t[2] - pos_d[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


#
# 4. テスト: CostMatrixBasedAssociation 抽象クラス自体の振る舞い
#
def test_cost_matrix_based_association_abstract_methods():
    """
    抽象メソッドが実装されていない場合はインスタンス化不可になることを確認
    """

    class IncompleteCostMatrix(CostMatrixBasedAssociation):
        # solve_assignment未実装
        pass

    with pytest.raises(TypeError):
        IncompleteCostMatrix(cost_func=euclidean_cost)


#
# 5. テスト用: ダミーのコストマトリクスアソシエーションクラス
#    solve_assignmentを固定的に返すことでassociateをテスト
#
def test_cost_matrix_based_association_dummy():
    class DummyCostMatrixSolver(CostMatrixBasedAssociation):
        def solve_assignment(self, cost_matrix: np.ndarray):
            # 単純に対角要素を対応付けにする
            row_idx = []
            col_idx = []
            nT, nD = cost_matrix.shape
            for i in range(min(nT, nD)):
                row_idx.append(i)
                col_idx.append(i)
            return np.array(row_idx), np.array(col_idx)

    assoc = DummyCostMatrixSolver(cost_func=euclidean_cost, max_distance=5.0)
    tracklets = [
        create_tracklet(0, 0, track_id="trk1"),
        create_tracklet(5, 5, track_id="trk2"),
    ]
    detections = [
        create_detection(0, 1, object_id="detA"),
        create_detection(6, 4, object_id="detB"),
    ]

    matched, unmatched_t, unmatched_d = assoc.associate(tracklets, detections)

    assert len(matched) == 2  # 2個とも対角割当
    assert len(unmatched_t) == 0
    assert len(unmatched_d) == 0
    # 簡易確認
    match_ids = [(trk.track_id, det.get_object_id()) for trk, det in matched]
    assert match_ids == [("trk1", "detA"), ("trk2", "detB")]


#
# 6. HungarianCostMatrixAssociation テスト
#
@pytest.mark.parametrize("max_dist", [5.0, 8.0])
def test_hungarian_cost_matrix_association(max_dist):
    assoc = HungarianCostMatrixAssociation(
        cost_func=euclidean_cost, max_distance=max_dist
    )

    tracklets = [
        create_tracklet(0, 0, track_id="trk1"),
        create_tracklet(10, 0, track_id="trk2"),
    ]
    detections = [
        create_detection(0, 1, object_id="detA"),  # 近いのはtrk1
        create_detection(10, 1, object_id="detB"),  # 近いのはtrk2
    ]

    matched, unmatched_tracks, unmatched_dets = assoc.associate(tracklets, detections)
    assert len(matched) == 2
    assert len(unmatched_tracks) == 0
    assert len(unmatched_dets) == 0

    # マッチペア確認
    match_dict = {trk.track_id: det.get_object_id() for trk, det in matched}
    assert match_dict["trk1"] == "detA"
    assert match_dict["trk2"] == "detB"


def test_hungarian_cost_matrix_association_unmatched():
    assoc = HungarianCostMatrixAssociation(cost_func=euclidean_cost, max_distance=2.0)

    tracklets = [
        create_tracklet(0, 0, track_id="trk1"),
    ]
    detections = [
        create_detection(5, 5, object_id="detA"),
    ]

    matched, unmatched_tracks, unmatched_dets = assoc.associate(tracklets, detections)
    assert len(matched) == 0
    assert len(unmatched_tracks) == 1
    assert len(unmatched_dets) == 1
    assert unmatched_tracks[0].track_id == "trk1"
    assert unmatched_dets[0].get_object_id() == "detA"


#
# 7. GNNCostMatrixAssociation テスト
#
def test_gnn_cost_matrix_association():
    assoc = GNNCostMatrixAssociation(cost_func=euclidean_cost, max_distance=10.0)

    tracklets = [
        create_tracklet(0, 0, track_id="trk1"),
        create_tracklet(10, 0, track_id="trk2"),
    ]
    detections = [
        create_detection(0, 1, object_id="detA"),  # trk1に近い
        create_detection(10, 2, object_id="detB"),  # trk2に近い
    ]

    matched, unmatched_tracks, unmatched_dets = assoc.associate(tracklets, detections)

    assert len(matched) == 2
    assert len(unmatched_tracks) == 0
    assert len(unmatched_dets) == 0
    match_dict = {trk.track_id: det.get_object_id() for trk, det in matched}
    assert match_dict["trk1"] == "detA"
    assert match_dict["trk2"] == "detB"
