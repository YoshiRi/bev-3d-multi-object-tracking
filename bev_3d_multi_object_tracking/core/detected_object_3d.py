import math
import uuid
from typing import Optional, Tuple


class Header:
    def __init__(self, frame_id: str, timestamp: float):
        self.frame_id = frame_id
        self.timestamp = timestamp


class KinematicState:
    def __init__(
        self,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],  # quaternion
        position_covariance: Optional[Tuple[float, ...]] = None,
        orientation_covariance: Optional[Tuple[float, ...]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        angular_velocity: Optional[Tuple[float, float, float]] = None,
        velocity_covariance: Optional[Tuple[float, ...]] = None,
        angular_velocity_covariance: Optional[Tuple[float, ...]] = None,
        acceleration: Optional[Tuple[float, float, float, float, float, float]] = None,
        acceleration_covariance: Optional[Tuple[float, ...]] = None,
    ):
        self.position = position
        self.orientation = orientation
        self.position_covariance = position_covariance
        self.orientation_covariance = orientation_covariance
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.velocity_covariance = velocity_covariance
        self.angular_velocity_covariance = angular_velocity_covariance
        self.acceleration = acceleration
        self.acceleration_covariance = acceleration_covariance

    def get_yaw(self) -> float:
        x, y, z, w = self.orientation
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


class GeometricInfo:
    def __init__(self, dimensions: Tuple[float, float, float]):
        # dimensions: (length, width, height)
        self.dimensions = dimensions


class ObjectInfo:
    def __init__(
        self,
        object_id: Optional[str] = None,
        class_name: Optional[str] = None,
        is_stationary: bool = False,
        input_sensor: Optional[str] = None,
        existence_probability: float = 1.0,
    ):
        self.object_id = object_id if object_id is not None else str(uuid.uuid4())
        self.class_name = class_name
        self.is_stationary = is_stationary
        self.input_sensor = input_sensor
        self.existence_probability = existence_probability


class DetectedObject3D:
    def __init__(
        self,
        header: Header,
        kinematic_state: KinematicState,
        geometric_info: GeometricInfo,
        object_info: ObjectInfo,
    ):
        self._header = header
        self._kinematic = kinematic_state
        self._geometric = geometric_info
        self._info = object_info

    def get_frame_id(self) -> str:
        return self._header.frame_id

    def get_timestamp(self) -> float:
        return self._header.timestamp

    def get_object_id(self) -> str:
        return self._info.object_id

    def get_class_name(self) -> Optional[str]:
        return self._info.class_name

    def get_existence_probability(self) -> float:
        return self._info.existence_probability

    def is_stationary(self) -> bool:
        return self._info.is_stationary

    def get_input_sensor(self) -> Optional[str]:
        return self._info.input_sensor

    def get_position(self) -> Tuple[float, float, float]:
        return self._kinematic.position

    def get_orientation(self) -> Tuple[float, float, float, float]:
        return self._kinematic.orientation

    def get_yaw(self) -> float:
        return self._kinematic.get_yaw()

    def get_velocity(self) -> Optional[Tuple[float, float, float]]:
        return self._kinematic.velocity

    def get_angular_velocity(self) -> Optional[Tuple[float, float, float]]:
        return self._kinematic.angular_velocity

    def get_acceleration(
        self,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        return self._kinematic.acceleration

    def get_dimensions(self) -> Tuple[float, float, float]:
        return self._geometric.dimensions
