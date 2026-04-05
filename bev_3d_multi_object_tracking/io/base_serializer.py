from abc import ABC, abstractmethod
from typing import Any, List

from bev_3d_multi_object_tracking.core.tracklet import Tracklet


class BaseSerializer(ABC):
    """
    Tracklet のリストを任意の外部フォーマットに変換するインターフェース。

    実装例:
        - ROS2 TrackedObject メッセージへの変換
        - CSV/JSON 形式への変換
        - 可視化ライブラリ向けの変換

    ライブラリ本体は Tracklet のみを扱うため、
    外部フォーマット依存のコードはすべてこのクラスの実装に閉じ込める。
    """

    @abstractmethod
    def serialize(self, tracklets: List[Tracklet]) -> Any:
        """
        Tracklet のリストを外部フォーマットに変換する。

        Args:
            tracklets : 出力対象の Tracklet リスト

        Returns:
            外部フォーマットのデータ (型はサブクラスで定義)
        """
        pass
