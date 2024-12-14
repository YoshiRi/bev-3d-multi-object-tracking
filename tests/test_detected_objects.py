# tests/test_detected_object.py
import math
from bev_3d_multi_object_tracking.common.detected_object_3d import (
    Header, KinematicState, GeometricInfo, ObjectInfo, DetectedObject3D
)

def test_detected_object_basic_info():
    # Setup: 必要なインスタンスを生成
    header = Header(frame_id="map", timestamp=1234567890.0)
    kinematic = KinematicState(
        position=(10.0, 5.0, 0.0),
        orientation=(0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)),  # yaw=45度(=pi/4)
    )
    geometric = GeometricInfo(dimensions=(4.5, 2.0, 1.5))
    info = ObjectInfo(object_id="test_id_123", class_name="car", is_stationary=False)

    obj = DetectedObject3D(header, kinematic, geometric, info)

    # Test: 基本的なゲッターが正しく動作するか確認
    assert obj.get_frame_id() == "map"
    assert obj.get_timestamp() == 1234567890.0
    assert obj.get_object_id() == "test_id_123"
    assert obj.get_class_name() == "car"
    assert obj.is_stationary() == False
    assert obj.get_dimensions() == (4.5, 2.0, 1.5)

def test_detected_object_pose_access():
    header = Header(frame_id="map", timestamp=1000.0)
    kinematic = KinematicState(
        position=(0.0, 0.0, 2.0),
        orientation=(0.0, 0.0, math.sin(math.pi/2/2), math.cos(math.pi/2/2)), 
        # yaw = 90度(pi/2) → quaternionで yaw=90度を表現
    )
    geometric = GeometricInfo(dimensions=(4.0, 2.0, 1.2))
    info = ObjectInfo(class_name="pedestrian", is_stationary=True)

    obj = DetectedObject3D(header, kinematic, geometric, info)

    # 位置とyaw角の取得テスト
    assert obj.get_position() == (0.0, 0.0, 2.0)
    # orientationはyaw=90° → get_yaw()は約1.5708(rad)を返すはず
    assert math.isclose(obj.get_yaw(), math.pi/2, abs_tol=1e-6)


def test_detected_object_defaults():
    # オプションを省略した状態で正しく初期化できるか
    header = Header("base_link", 50.0)
    kinematic = KinematicState(position=(1,2,3), orientation=(0,0,0,1))
    geometric = GeometricInfo(dimensions=(1,1,1))
    info = ObjectInfo()

    obj = DetectedObject3D(header, kinematic, geometric, info)
    # オブジェクトIDはNone指定ならUUID生成になっている想定
    assert obj.get_object_id() is not None
    assert obj.get_class_name() is None  # 未指定ならNone
