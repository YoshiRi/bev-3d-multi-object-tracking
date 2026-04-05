from typing import Any, Dict, List

from bev_3d_multi_object_tracking.core.geometry_utils import quaternion_to_yaw
from bev_3d_multi_object_tracking.core.tracklet import Tracklet
from bev_3d_multi_object_tracking.io.base_serializer import BaseSerializer


class DictSerializer(BaseSerializer):
    """
    Tracklet のリストを dict のリストに変換するシリアライザ。

    出力フォーマット (各 dict のキー):
        トラック情報:
            track_id           : トラックID
            age                : 生成からのフレーム数
            missed_count       : 連続 miss フレーム数
            is_confirmed       : 確定フラグ
            is_lost            : 消滅フラグ
        位置・運動:
            x, y, z            : 位置 [m]
            yaw                : z軸回りの回転角 [rad]
            vx, vy, vz         : 速度 [m/s] (None の場合は 0.0)
            yaw_rate           : 角速度 [rad/s] (None の場合は 0.0)
        形状:
            length, width, height : 寸法 [m]
        メタ情報:
            class_name         : クラス名
            existence_probability : 存在確率
            is_stationary      : 静止フラグ
            timestamp          : 最終更新タイムスタンプ
            frame_id           : 座標フレームID

    dict リスト形式のため、csv.DictWriter や json.dump で直接出力できる。
    """

    def serialize(self, tracklets: List[Tracklet]) -> List[Dict[str, Any]]:
        return [self._serialize_one(t) for t in tracklets]

    def _serialize_one(self, tracklet: Tracklet) -> Dict[str, Any]:
        kin = tracklet.get_current_kinematic()
        geom = tracklet.get_geometric_info()
        obj = tracklet.get_object_info()

        pos = kin.position
        yaw = quaternion_to_yaw(kin.orientation)

        vel = kin.velocity or (0.0, 0.0, 0.0)
        ang_vel = kin.angular_velocity or (0.0, 0.0, 0.0)

        dims = geom.dimensions

        return {
            "track_id": tracklet.get_track_id(),
            "age": tracklet.age,
            "missed_count": tracklet.missed_count,
            "is_confirmed": tracklet.is_confirmed,
            "is_lost": tracklet.is_lost,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "yaw": yaw,
            "vx": vel[0],
            "vy": vel[1],
            "vz": vel[2],
            "yaw_rate": ang_vel[2],
            "length": dims[0],
            "width": dims[1],
            "height": dims[2],
            "class_name": obj.class_name,
            "existence_probability": obj.existence_probability,
            "is_stationary": obj.is_stationary,
            "timestamp": tracklet.get_timestamp(),
            "frame_id": tracklet.get_frame_id(),
        }
