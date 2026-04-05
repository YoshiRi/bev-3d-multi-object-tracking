from abc import ABC, abstractmethod
from typing import List, Tuple

from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


class BaseDataAssociation(ABC):
    @abstractmethod
    def associate(
        self, tracklets: List[Tracklet], detections: List[DetectedObject3D]
    ) -> Tuple[
        List[Tuple[Tracklet, DetectedObject3D]], List[Tracklet], List[DetectedObject3D]
    ]:
        """
        前フレームまでのトラック(Tracklet)集合と、新たなフレームの検出(DetectedObject3D)集合を入力に、
        マッチ結果を返す抽象メソッド。

        Returns:
            - matched_pairs: [(tracklet, detection), ...]
                マッチしたペアのリスト
            - unmatched_tracklets: [Tracklet, ...]
                マッチしなかったトラック（このフレームで対応検出が見つからなかったトラック）
            - unmatched_detections: [DetectedObject3D, ...]
                新たなフレームで検出されたが、既存トラックと対応づけできなかった検出
        """
        pass
