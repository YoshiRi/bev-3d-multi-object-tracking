import datetime
import uuid
from typing import Any, List, Optional, Tuple

from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)


class Tracklet:
    """
    一つのオブジェクトを追跡するための状態管理クラス。

    設計方針:
        - Tracklet は機械的な処理のみ担当する (フィルタ操作・カウンタ管理)。
        - 「何回 miss したら消滅させるか」「何回 hit したら確定とするか」などの
          ライフサイクル方針は MultiObjectTracker が持ち、Tracklet 外から制御する。
        - フィルタの内部状態 (_filter_state) は opaque として扱い、
          内容を解釈せず predict / update / to_kinematic_state にそのまま渡す。
        - 外部読み取りには _kinematic_state (KinematicState) を使う。

    属性 is_confirmed / is_lost は外部から書き込み可能な単純フラグ。
    トラッカーが age / missed_count を見て制御する。
    """

    def __init__(
        self,
        filter_instance: BaseFilter,
        initial_detection: DetectedObject3D,
        track_id: Optional[str] = None,
    ):
        self.track_id = track_id if track_id is not None else str(uuid.uuid4())
        self.filter_instance = filter_instance

        self._header = Header(
            frame_id=initial_detection.get_frame_id(),
            timestamp=initial_detection.get_timestamp(),
        )
        self._object_info = ObjectInfo(
            object_id=initial_detection.get_object_id(),
            class_name=initial_detection.get_class_name(),
            is_stationary=initial_detection.is_stationary(),
            input_sensor=initial_detection.get_input_sensor(),
            existence_probability=initial_detection.get_existence_probability(),
        )
        self._geometric_info = GeometricInfo(dimensions=initial_detection.get_dimensions())

        # フィルタの opaque 状態
        self._filter_state: Any = self.filter_instance.initialize_state(initial_detection)
        # 外部読み取り用 KinematicState
        self._kinematic_state: KinematicState = self.filter_instance.to_kinematic_state(
            self._filter_state
        )

        self.initial_timestamp = self._header.timestamp
        self.last_update_timestamp = self._header.timestamp
        self.creation_time = datetime.datetime.now()

        # カウンタ (方針判断はトラッカー側)
        self.age = 1
        self.missed_count = 0

        # ライフサイクルフラグ (トラッカーが外から制御する)
        self.is_confirmed = False
        self.is_lost = False

        self.history: List[Tuple[float, KinematicState, ObjectInfo]] = []

    # ------------------------------------------------------------------
    # 予測ステップ (観測なし)
    # ------------------------------------------------------------------

    def predict(self, dt: float):
        """フィルタで状態を予測し、missed_count をインクリメントする。"""
        self._filter_state = self.filter_instance.predict(self._filter_state, dt)
        self._kinematic_state = self.filter_instance.to_kinematic_state(self._filter_state)
        self.missed_count += 1

    # ------------------------------------------------------------------
    # 更新ステップ (観測あり)
    # ------------------------------------------------------------------

    def update(self, detection: DetectedObject3D):
        """新たな観測でフィルタを更新し、age をインクリメント・missed_count をリセットする。"""
        self.history.append(
            (self.last_update_timestamp, self._kinematic_state, self._object_info)
        )

        self._filter_state = self.filter_instance.update(self._filter_state, detection)
        self._kinematic_state = self.filter_instance.to_kinematic_state(self._filter_state)

        if detection.get_class_name():
            self._object_info.class_name = detection.get_class_name()
        self._object_info.is_stationary = detection.is_stationary()
        self._object_info.input_sensor = (
            detection.get_input_sensor() or self._object_info.input_sensor
        )
        self._geometric_info = GeometricInfo(dimensions=detection.get_dimensions())
        self.last_update_timestamp = detection.get_timestamp()

        self.age += 1
        self.missed_count = 0

    # ------------------------------------------------------------------
    # ゲッター
    # ------------------------------------------------------------------

    def get_current_kinematic(self) -> KinematicState:
        return self._kinematic_state

    def get_object_info(self) -> ObjectInfo:
        return self._object_info

    def get_geometric_info(self) -> GeometricInfo:
        return self._geometric_info

    def get_frame_id(self) -> str:
        return self._header.frame_id

    def get_timestamp(self) -> float:
        return self.last_update_timestamp

    def get_track_id(self) -> str:
        return self.track_id

    # ------------------------------------------------------------------
    # ライフサイクルフラグ操作 (トラッカーから呼ばれる)
    # ------------------------------------------------------------------

    def mark_confirmed(self):
        self.is_confirmed = True

    def mark_lost(self):
        self.is_lost = True
