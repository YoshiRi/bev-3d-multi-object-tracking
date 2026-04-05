from typing import Any, Dict, List

from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.geometry_utils import yaw_to_quaternion
from bev_3d_multi_object_tracking.io.base_parser import BaseParser


class DictParser(BaseParser):
    """
    dict (または dict のリスト) を DetectedObject3D に変換するパーサー。

    入力フォーマット (各 dict のキー):
        必須:
            x, y          : BEV 平面上の位置 [m]
        オプション:
            z             : 高さ方向の位置 [m] (デフォルト: 0.0)
            yaw           : z軸回りの回転角 [rad] (デフォルト: 0.0)
            length        : 物体の長さ [m] (デフォルト: 1.0)
            width         : 物体の幅 [m] (デフォルト: 1.0)
            height        : 物体の高さ [m] (デフォルト: 1.0)
            timestamp     : タイムスタンプ [秒] (デフォルト: 0.0)
            frame_id      : 座標フレームID (デフォルト: "")
            class_name    : クラス名 (デフォルト: None)
            object_id     : オブジェクトID (デフォルト: UUID 自動生成)
            is_stationary : 静止フラグ (デフォルト: False)
            existence_probability : 存在確率 (デフォルト: 1.0)

    CSV の DictReader や JSON の load 結果をそのまま渡せる形式を想定している。
    """

    def parse(self, raw: Any) -> List[DetectedObject3D]:
        """
        Args:
            raw : dict 1件、または dict のリスト

        Returns:
            DetectedObject3D のリスト
        """
        if isinstance(raw, dict):
            return [self._parse_one(raw)]
        return [self._parse_one(item) for item in raw]

    def _parse_one(self, d: Dict) -> DetectedObject3D:
        header = Header(
            frame_id=str(d.get("frame_id", "")),
            timestamp=float(d.get("timestamp", 0.0)),
        )
        yaw = float(d.get("yaw", 0.0))
        kinematic = KinematicState(
            position=(float(d["x"]), float(d["y"]), float(d.get("z", 0.0))),
            orientation=yaw_to_quaternion(yaw),
        )
        geom = GeometricInfo(
            dimensions=(
                float(d.get("length", 1.0)),
                float(d.get("width", 1.0)),
                float(d.get("height", 1.0)),
            )
        )
        obj_info = ObjectInfo(
            object_id=d.get("object_id"),
            class_name=d.get("class_name"),
            is_stationary=bool(d.get("is_stationary", False)),
            existence_probability=float(d.get("existence_probability", 1.0)),
        )
        return DetectedObject3D(
            header=header,
            kinematic_state=kinematic,
            geometric_info=geom,
            object_info=obj_info,
        )
