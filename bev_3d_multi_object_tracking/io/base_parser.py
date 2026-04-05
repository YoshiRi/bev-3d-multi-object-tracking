from abc import ABC, abstractmethod
from typing import Any, List

from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D


class BaseParser(ABC):
    """
    任意の外部フォーマットを DetectedObject3D のリストに変換するインターフェース。

    実装例:
        - ROS2 Detection3DArray メッセージのパーサー
        - CSV/JSON ファイルの行をパースするパーサー
        - カスタムバイナリプロトコルのパーサー

    ライブラリ本体は DetectedObject3D のみを扱うため、
    外部フォーマット依存のコードはすべてこのクラスの実装に閉じ込める。
    """

    @abstractmethod
    def parse(self, raw: Any) -> List[DetectedObject3D]:
        """
        外部フォーマットの入力を DetectedObject3D のリストに変換する。

        Args:
            raw : 外部フォーマットのデータ (型はサブクラスで定義)

        Returns:
            DetectedObject3D のリスト
        """
        pass
