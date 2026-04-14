import math

import pytest

from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D
from bev_3d_multi_object_tracking.core.geometry_utils import quaternion_to_yaw
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
from bev_3d_multi_object_tracking.io.dict_parser import DictParser
from bev_3d_multi_object_tracking.io.dict_serializer import DictSerializer
from bev_3d_multi_object_tracking.core.tracklet import Tracklet


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def make_tracklet(x=1.0, y=2.0, yaw=0.5, class_name="Car") -> Tracklet:
    parser = DictParser()
    det = parser.parse({"x": x, "y": y, "yaw": yaw, "class_name": class_name,
                        "length": 4.0, "width": 2.0, "height": 1.5, "frame_id": "map"})[0]
    return Tracklet(filter_instance=VehicleTracker(), initial_detection=det)


# ==================================================================
# DictParser
# ==================================================================

class TestDictParser:

    def test_parse_single_dict_returns_list(self):
        parser = DictParser()
        result = parser.parse({"x": 1.0, "y": 2.0})
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], DetectedObject3D)

    def test_parse_list_of_dicts(self):
        parser = DictParser()
        raw = [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 5.0}]
        result = parser.parse(raw)
        assert len(result) == 2

    def test_position_preserved(self):
        parser = DictParser()
        det = parser.parse({"x": 3.0, "y": -1.5, "z": 0.8})[0]
        pos = det.get_position()
        assert pos[0] == pytest.approx(3.0)
        assert pos[1] == pytest.approx(-1.5)
        assert pos[2] == pytest.approx(0.8)

    def test_yaw_roundtrip(self):
        """parse した yaw が get_yaw() で戻ってくる。"""
        parser = DictParser()
        target_yaw = math.pi / 4
        det = parser.parse({"x": 0.0, "y": 0.0, "yaw": target_yaw})[0]
        assert det.get_yaw() == pytest.approx(target_yaw, abs=1e-5)

    def test_dimensions_defaults(self):
        """length/width/height を省略したときデフォルト (1.0, 1.0, 1.0) になる。"""
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0})[0]
        dims = det.get_dimensions()
        assert dims == (1.0, 1.0, 1.0)

    def test_dimensions_explicit(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "length": 4.5, "width": 2.1, "height": 1.6})[0]
        dims = det.get_dimensions()
        assert dims[0] == pytest.approx(4.5)
        assert dims[1] == pytest.approx(2.1)
        assert dims[2] == pytest.approx(1.6)

    def test_timestamp_default_zero(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0})[0]
        assert det.get_timestamp() == pytest.approx(0.0)

    def test_timestamp_explicit(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "timestamp": 1.23})[0]
        assert det.get_timestamp() == pytest.approx(1.23)

    def test_frame_id_preserved(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "frame_id": "base_link"})[0]
        assert det.get_frame_id() == "base_link"

    def test_frame_id_default_empty(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0})[0]
        assert det.get_frame_id() == ""

    def test_class_name_preserved(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "class_name": "Pedestrian"})[0]
        assert det.get_class_name() == "Pedestrian"

    def test_class_name_default_none(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0})[0]
        assert det.get_class_name() is None

    def test_existence_probability_preserved(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "existence_probability": 0.75})[0]
        assert det.get_existence_probability() == pytest.approx(0.75)

    def test_is_stationary_flag(self):
        parser = DictParser()
        det = parser.parse({"x": 0.0, "y": 0.0, "is_stationary": True})[0]
        assert det.is_stationary() is True

    def test_missing_required_key_raises(self):
        """x または y がないと KeyError が発生する。"""
        parser = DictParser()
        with pytest.raises(KeyError):
            parser.parse({"y": 0.0})  # x がない


# ==================================================================
# DictSerializer
# ==================================================================

class TestDictSerializer:

    def test_serialize_empty_list(self):
        result = DictSerializer().serialize([])
        assert result == []

    def test_serialize_returns_list_of_dicts(self):
        tracklet = make_tracklet()
        result = DictSerializer().serialize([tracklet])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_all_expected_keys_present(self):
        expected_keys = {
            "track_id", "age", "missed_count", "is_confirmed", "is_lost",
            "x", "y", "z", "yaw", "vx", "vy", "vz", "yaw_rate",
            "length", "width", "height",
            "class_name", "existence_probability", "is_stationary",
            "timestamp", "frame_id",
        }
        result = DictSerializer().serialize([make_tracklet()])
        assert expected_keys == set(result[0].keys())

    def test_position_preserved(self):
        tracklet = make_tracklet(x=3.0, y=-1.5)
        result = DictSerializer().serialize([tracklet])[0]
        assert result["x"] == pytest.approx(3.0, abs=1e-3)
        assert result["y"] == pytest.approx(-1.5, abs=1e-3)

    def test_yaw_roundtrip(self):
        """parse → track → serialize で yaw が保持される。"""
        target_yaw = math.pi / 6
        tracklet = make_tracklet(yaw=target_yaw)
        result = DictSerializer().serialize([tracklet])[0]
        assert result["yaw"] == pytest.approx(target_yaw, abs=1e-5)

    def test_class_name_preserved(self):
        tracklet = make_tracklet(class_name="Truck")
        result = DictSerializer().serialize([tracklet])[0]
        assert result["class_name"] == "Truck"

    def test_track_id_is_string(self):
        tracklet = make_tracklet()
        result = DictSerializer().serialize([tracklet])[0]
        assert isinstance(result["track_id"], str)

    def test_lifecycle_flags(self):
        tracklet = make_tracklet()
        tracklet.mark_confirmed()
        result = DictSerializer().serialize([tracklet])[0]
        assert result["is_confirmed"] is True
        assert result["is_lost"] is False

    def test_age_and_missed_count(self):
        tracklet = make_tracklet()
        tracklet.predict(0.1)
        tracklet.predict(0.1)
        result = DictSerializer().serialize([tracklet])[0]
        assert result["age"] == 1       # predict は age を増やさない
        assert result["missed_count"] == 2

    def test_serialize_multiple_tracklets(self):
        tracklets = [make_tracklet(x=float(i)) for i in range(5)]
        result = DictSerializer().serialize(tracklets)
        assert len(result) == 5
        xs = [r["x"] for r in result]
        assert sorted(xs) == pytest.approx(sorted([float(i) for i in range(5)]), abs=1e-2)
