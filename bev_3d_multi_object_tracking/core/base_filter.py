from abc import ABC, abstractmethod

from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    KinematicState,
)


class BaseFilter(ABC):
    @abstractmethod
    def initialize_state(self, detection: DetectedObject3D):
        """
        最初の観測情報から初期フィルタ状態を生成する。
        戻り値の形式はサブクラスに委ねる (opaque state)。
        Tracklet はこの戻り値を解釈せず、predict/update/to_kinematic_state にそのまま渡す。
        """
        pass

    @abstractmethod
    def predict(self, filter_state, dt: float):
        """
        filter_state を dt 秒分だけ予測し、新しい filter_state を返す。
        initialize_state / update と同じ形式の state を受け取り、同じ形式を返す。
        """
        pass

    @abstractmethod
    def update(self, filter_state, detection: DetectedObject3D):
        """
        観測 detection で filter_state を更新し、新しい filter_state を返す。
        """
        pass

    @abstractmethod
    def to_kinematic_state(self, filter_state) -> KinematicState:
        """
        フィルタ内部の opaque state を KinematicState に変換する。
        Tracklet が外部への読み取りインターフェースとして使用する。
        """
        pass
