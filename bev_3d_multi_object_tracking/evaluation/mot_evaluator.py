"""
MOT (Multiple Object Tracking) 評価指標の実装。

標準指標:
    MOTA  (Multiple Object Tracking Accuracy)
        トラッカー全体の精度。FP・FN・IDSWをすべて考慮する。
        MOTA = 1 - (ΣFP + ΣFN + ΣIDSW) / ΣGT
        範囲: (-∞, 1.0]  ← 1.0 が完全, 負値もあり得る

    MOTP  (Multiple Object Tracking Precision)
        マッチした GT-Track ペアの平均距離。位置精度を表す。
        MOTP = Σ d(gt, track) / Σ|matches|
        範囲: [0, match_threshold]  ← 0 が完全

    IDSW  (ID Switch)
        GT 物体が途中で別の Track ID に割り当てられた回数。

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN) = TP / GT

使い方:
    evaluator = MOTEvaluator(match_threshold=2.0)

    for frame_idx, dets, gt_dicts in scenario.iter_frames():
        gt_detections = DictParser().parse(gt_dicts)
        tracks = tracker.update(DictParser().parse(dets), dt=scenario.dt)
        evaluator.update(gt_detections, tracks)

    metrics = evaluator.compute()
    print(metrics)

注意:
    - ground_truth の各 DetectedObject3D は object_id が設定されている必要がある
      (ID Switch の検出に使用する)
    - tracks は MultiObjectTracker.update() が返す全トラックを渡す
      (confirmed のみにしたい場合は呼び出し元でフィルタする)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


# ------------------------------------------------------------------
# 結果データクラス
# ------------------------------------------------------------------

@dataclass
class FrameResult:
    """1フレームのマッチング結果。"""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    total_match_distance: float = 0.0

    @property
    def n_gt(self) -> int:
        return self.tp + self.fn

    @property
    def n_tracks(self) -> int:
        return self.tp + self.fp


@dataclass
class MOTMetrics:
    """全フレームを集計した MOT 評価指標。"""
    mota: float          # Multiple Object Tracking Accuracy
    motp: float          # Multiple Object Tracking Precision (平均距離)
    id_switches: int     # ID スイッチ総数
    fp: int              # False Positive 総数
    fn: int              # False Negative 総数
    tp: int              # True Positive 総数
    total_gt: int        # GT 物体の総数 (全フレーム合計)
    precision: float     # TP / (TP + FP)
    recall: float        # TP / (TP + FN) = TP / GT
    n_frames: int        # 評価したフレーム数

    def __str__(self) -> str:
        lines = [
            "=" * 42,
            "  MOT Evaluation Results",
            "=" * 42,
            f"  Frames evaluated : {self.n_frames}",
            f"  Total GT objects : {self.total_gt}",
            "-" * 42,
            f"  MOTA             : {self.mota:+.4f}",
            f"  MOTP             : {self.motp:.4f} m",
            "-" * 42,
            f"  TP               : {self.tp}",
            f"  FP               : {self.fp}",
            f"  FN               : {self.fn}",
            f"  ID Switches      : {self.id_switches}",
            "-" * 42,
            f"  Precision        : {self.precision:.4f}",
            f"  Recall           : {self.recall:.4f}",
            "=" * 42,
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------
# 評価器
# ------------------------------------------------------------------

class MOTEvaluator:
    """
    フレームごとの GT と Track のマッチングを積み上げて MOT 指標を計算する。

    Args:
        match_threshold : GT-Track ペアを有効なマッチとみなす最大 x-y 距離 [m]
                          これを超えるペアは FP/FN として扱う (デフォルト: 2.0)
    """

    def __init__(self, match_threshold: float = 2.0):
        self._threshold = match_threshold
        self._frame_results: List[FrameResult] = []
        # GT object_id → 直前フレームでマッチした track_id
        self._prev_gt_to_track: Dict[str, str] = {}

    def reset(self):
        """蓄積された結果をリセットする。"""
        self._frame_results.clear()
        self._prev_gt_to_track.clear()

    def update(
        self,
        ground_truth: List[DetectedObject3D],
        tracks: List[Tracklet],
    ) -> FrameResult:
        """
        1フレーム分の GT と Track をマッチングして結果を蓄積する。

        Args:
            ground_truth : GT 物体のリスト (object_id が設定されていること)
            tracks       : トラッカーが出力した Tracklet のリスト

        Returns:
            このフレームの FrameResult
        """
        result = FrameResult()

        if not ground_truth and not tracks:
            self._frame_results.append(result)
            return result

        if not ground_truth:
            result.fp = len(tracks)
            self._frame_results.append(result)
            return result

        if not tracks:
            result.fn = len(ground_truth)
            self._frame_results.append(result)
            return result

        # ---- コスト行列の構築 (x-y ユークリッド距離) ----
        cost = np.full((len(ground_truth), len(tracks)), fill_value=1e9)
        for gi, gt in enumerate(ground_truth):
            gx, gy, _ = gt.get_position()
            for ti, tr in enumerate(tracks):
                tx, ty, _ = tr.get_current_kinematic().position
                cost[gi, ti] = math.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)

        # ---- Hungarian マッチング ----
        row_idx, col_idx = linear_sum_assignment(cost)

        # ---- TP / FP / FN / IDSW の判定 ----
        matched_gt: set = set()
        matched_tr: set = set()
        current_gt_to_track: Dict[str, str] = {}

        for gi, ti in zip(row_idx, col_idx):
            dist = cost[gi, ti]
            if dist > self._threshold:
                continue  # 距離が閾値超 → マッチなし

            gt = ground_truth[gi]
            tr = tracks[ti]
            gt_id = gt.get_object_id()
            tr_id = tr.get_track_id()

            # ID Switch 検出
            prev_tr_id = self._prev_gt_to_track.get(gt_id)
            if prev_tr_id is not None and prev_tr_id != tr_id:
                result.id_switches += 1

            current_gt_to_track[gt_id] = tr_id
            matched_gt.add(gi)
            matched_tr.add(ti)
            result.tp += 1
            result.total_match_distance += dist

        result.fn = len(ground_truth) - len(matched_gt)
        result.fp = len(tracks) - len(matched_tr)

        self._prev_gt_to_track = current_gt_to_track
        self._frame_results.append(result)
        return result

    def compute(self) -> MOTMetrics:
        """
        蓄積された全フレームの結果から MOT 指標を計算する。

        Returns:
            MOTMetrics
        """
        total_tp = sum(r.tp for r in self._frame_results)
        total_fp = sum(r.fp for r in self._frame_results)
        total_fn = sum(r.fn for r in self._frame_results)
        total_idsw = sum(r.id_switches for r in self._frame_results)
        total_gt = total_tp + total_fn
        total_dist = sum(r.total_match_distance for r in self._frame_results)

        # MOTA
        if total_gt == 0:
            mota = 1.0
        else:
            mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt

        # MOTP (マッチが1件もなければ 0.0)
        motp = total_dist / total_tp if total_tp > 0 else 0.0

        # Precision / Recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0

        return MOTMetrics(
            mota=mota,
            motp=motp,
            id_switches=total_idsw,
            fp=total_fp,
            fn=total_fn,
            tp=total_tp,
            total_gt=total_gt,
            precision=precision,
            recall=recall,
            n_frames=len(self._frame_results),
        )

    @property
    def frame_results(self) -> List[FrameResult]:
        """フレームごとの FrameResult リスト (参照用)。"""
        return list(self._frame_results)
