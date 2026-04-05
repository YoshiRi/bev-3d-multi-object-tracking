from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from bev_3d_multi_object_tracking.core.base_data_association import BaseDataAssociation
from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


class CostMatrixBasedAssociation(BaseDataAssociation):
    def __init__(
        self,
        cost_func: Callable[[Tracklet, DetectedObject3D], float],
        max_distance: float = 10.0,
    ):
        """
        :param cost_func: (tracklet, detection) -> スカラーコスト(float) を返す関数
        :param max_distance: これを超えるコストは割当不可とみなし、大きな値を設定する
        """
        self.cost_func = cost_func
        self.max_distance = max_distance

    def associate(
        self, tracklets: List[Tracklet], detections: List[DetectedObject3D]
    ) -> Tuple[
        List[Tuple[Tracklet, DetectedObject3D]], List[Tracklet], List[DetectedObject3D]
    ]:

        if not tracklets or not detections:
            return [], tracklets, detections

        # 1. コスト行列を生成
        cost_matrix = self.build_cost_matrix(tracklets, detections)

        # 2. ソルバで割当を解く
        row_idx, col_idx = self.solve_assignment(cost_matrix)

        # 3. 割当結果を matched/unmatched に仕分け
        matched_pairs = []
        used_tracks = set()
        used_dets = set()

        for r, c in zip(row_idx, col_idx):
            # コストが一定値以上の場合、マッチなしと扱う
            if cost_matrix[r, c] >= 1e5:
                continue
            matched_pairs.append((tracklets[r], detections[c]))
            used_tracks.add(r)
            used_dets.add(c)

        unmatched_tracks = [
            tracklets[i] for i in range(len(tracklets)) if i not in used_tracks
        ]
        unmatched_dets = [
            detections[j] for j in range(len(detections)) if j not in used_dets
        ]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def build_cost_matrix(
        self, tracklets: List[Tracklet], detections: List[DetectedObject3D]
    ) -> np.ndarray:
        """
        トラック数 x 検出数 のコスト行列を作成。
        cost_funcで計算し、max_distanceを超える場合は大きな値(1e6)にする。
        """
        nT = len(tracklets)
        nD = len(detections)
        cost_matrix = np.zeros((nT, nD), dtype=np.float32)
        for i, trk in enumerate(tracklets):
            for j, det in enumerate(detections):
                cost = self.cost_func(trk, det)
                if cost > self.max_distance:
                    cost_matrix[i, j] = 1e6
                else:
                    cost_matrix[i, j] = cost
        return cost_matrix

    @abstractmethod
    def solve_assignment(
        self, cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        コスト行列に対する割当問題を解き、
        - row_idx (trackインデックスの配列)
        - col_idx (detectionインデックスの配列)
        を返す。
        これは線形割当、GNN、MUSSPなどの具体ロジックで実装する。
        """
        pass


class HungarianCostMatrixAssociation(CostMatrixBasedAssociation):

    def solve_assignment(
        self, cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SciPyのlinear_sum_assignment(ハンガリアン法)でコスト最小解を得る。
        """
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return row_idx, col_idx


class GNNCostMatrixAssociation(CostMatrixBasedAssociation):
    def solve_assignment(
        self, cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GNNは最も近い(最小コスト)のペアを確定→それらを除外→再度最小コスト探索... と
        Greedyに繰り返す実装などが考えられます。
        ここでは簡単な貪欲実装をサンプルに。
        """

        row_idx = []
        col_idx = []

        # 複製して操作
        mat = cost_matrix.copy()

        nT, nD = mat.shape
        used_rows = set()
        used_cols = set()

        while True:
            # 最小コスト要素を探す
            rmin, cmin = np.unravel_index(np.argmin(mat), mat.shape)
            min_val = mat[rmin, cmin]

            if min_val == 1e6 or min_val == np.inf:
                # もうマッチできるものが無い
                break
            # マッチ確定
            row_idx.append(rmin)
            col_idx.append(cmin)

            # 使われた行・列を大きな値で埋めて再度検索
            mat[rmin, :] = 1e6
            mat[:, cmin] = 1e6

            used_rows.add(rmin)
            used_cols.add(cmin)

            if len(used_rows) == nT or len(used_cols) == nD:
                break

        return np.array(row_idx), np.array(col_idx)
