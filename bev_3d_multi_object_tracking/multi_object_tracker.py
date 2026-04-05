from dataclasses import dataclass, field
from typing import Callable, List

from bev_3d_multi_object_tracking.core.base_data_association import BaseDataAssociation
from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


@dataclass
class TrackingConfig:
    """
    トラッキングのライフサイクル方針を一元管理する設定クラス。

    Tracklet は機械的な処理のみ行い、ここで定義された方針を
    MultiObjectTracker が適用する。

    Attributes:
        min_hits_to_confirm : この回数以上 update されたトラックを confirmed とする
        max_misses_to_lose  : この回数以上連続で miss したトラックを lost とする
        min_existence_prob  : confirmed 判定に必要な最低存在確率
        miss_decay          : miss 時に existence_probability に掛ける減衰係数
        hit_prob_alpha      : update 時の存在確率ブレンド係数 (prior 側の重み)
    """

    min_hits_to_confirm: int = 3
    max_misses_to_lose: int = 3
    min_existence_prob: float = 0.0
    miss_decay: float = 0.8
    hit_prob_alpha: float = 0.5


class MultiObjectTracker:
    """
    多物体追跡のメインクラス。

    フレームごとに update() を呼ぶことで、検出群とトラック群を管理する。

    設計:
        - TrackingConfig でライフサイクル方針を集中管理する。
        - filter_factory で各トラックに独立したフィルタインスタンスを生成する。
        - Tracklet は機械的な処理のみ担い、確定/消滅の判断はここで行う。
        - シングルスレッド前提 (スレッドセーフではない)。

    Args:
        filter_factory : 新規トラック生成時に呼ばれる BaseFilter ファクトリ関数
        association    : データアソシエーション実装
        config         : トラッキング方針設定 (省略時はデフォルト値)
    """

    def __init__(
        self,
        filter_factory: Callable[[], BaseFilter],
        association: BaseDataAssociation,
        config: TrackingConfig = None,
    ):
        self._filter_factory = filter_factory
        self._association = association
        self._config = config if config is not None else TrackingConfig()
        self._tracklets: List[Tracklet] = []

    def update(
        self, detections: List[DetectedObject3D], dt: float
    ) -> List[Tracklet]:
        """
        1フレーム分のトラッキング処理を実行する。

        処理順序:
            1. 全トラックを予測 (predict)
            2. 予測済みトラックと検出をアソシエーション
            3. マッチしたペアをフィルタ更新 (update)
            4. アンマッチ検出から新規トラックを生成
            5. ライフサイクル方針を適用 (confirmed / lost 判定)
            6. lost トラックを除去

        Args:
            detections : 今フレームの DetectedObject3D リスト
            dt         : 前フレームからの経過時間 [秒]

        Returns:
            loss でない全トラックのリスト
        """
        cfg = self._config

        # 1. 全トラックを予測
        for tracklet in self._tracklets:
            tracklet.predict(dt)

        # 2. アソシエーション
        matched, unmatched_tracks, unmatched_dets = self._association.associate(
            self._tracklets, detections
        )

        # 3. マッチしたトラックを更新
        for tracklet, detection in matched:
            tracklet.update(detection)
            # 存在確率のブレンド更新
            obj_info = tracklet.get_object_info()
            obj_info.existence_probability = (
                cfg.hit_prob_alpha * obj_info.existence_probability
                + (1.0 - cfg.hit_prob_alpha) * detection.get_existence_probability()
            )

        # 4. アンマッチ検出 → 新規トラック生成
        for det in unmatched_dets:
            self._tracklets.append(
                Tracklet(filter_instance=self._filter_factory(), initial_detection=det)
            )

        # 5. ライフサイクル方針を適用
        for tracklet in self._tracklets:
            obj_info = tracklet.get_object_info()

            if tracklet.missed_count > 0:
                # miss フレームの存在確率減衰
                obj_info.existence_probability *= cfg.miss_decay ** tracklet.missed_count

            if tracklet.missed_count >= cfg.max_misses_to_lose:
                tracklet.mark_lost()

            if (
                not tracklet.is_confirmed
                and tracklet.age >= cfg.min_hits_to_confirm
                and obj_info.existence_probability >= cfg.min_existence_prob
            ):
                tracklet.mark_confirmed()

        # 6. lost トラックを除去
        self._tracklets = [t for t in self._tracklets if not t.is_lost]

        return list(self._tracklets)

    def get_confirmed_tracks(self) -> List[Tracklet]:
        """confirmed 済みのトラックのみ返す。"""
        return [t for t in self._tracklets if t.is_confirmed]

    def get_all_tracks(self) -> List[Tracklet]:
        """lost でない全トラックを返す。"""
        return list(self._tracklets)

    def reset(self):
        """全トラックをクリアする。"""
        self._tracklets.clear()
