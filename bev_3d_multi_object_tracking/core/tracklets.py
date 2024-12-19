import datetime
import uuid
from typing import List, Optional, Tuple

from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)


class Tracklet:
    def __init__(
        self,
        filter_instance: BaseFilter,
        initial_detection: DetectedObject3D,
        track_id: Optional[str] = None,
    ):
        """
        Trackletは、一つのオブジェクトを追跡するための状態管理クラス。
        オブジェクトの運動状態(KinematicState)やメタ情報(ObjectInfo)、形状(GeometricInfo)を保持し、
        フィルタによる予測・更新を通して状態を発展させる。

        :param filter_instance: 状態推定に使用するフィルタ(BaseFilter実装)
        :param initial_detection: 最初に観測されたオブジェクト情報
        :param track_id: トラックID指定（未指定ならUUID）
        """
        self.track_id = track_id if track_id is not None else str(uuid.uuid4())
        self.filter_instance = filter_instance

        # DetectedObject3Dから情報抽出
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

        self._geometric_info = GeometricInfo(
            dimensions=initial_detection.get_dimensions()
        )

        # KinematicState初期化
        # ここではフィルタに初期観測を渡し、初期状態(KinematicState)を生成させることを想定
        init_kinematic = self.filter_instance.initialize_state(initial_detection)
        # initialize_stateからは、状態ベクトルや共分散などの生状態が返る想定だが、
        # KinematicStateに格納し直す必要がある場合はここで変換する。
        # 今はフィルタが直接KinematicStateを返す想定をしてもよい。
        if isinstance(init_kinematic, KinematicState):
            self._kinematic_state = init_kinematic
        else:
            # フィルタが (state_vec, covariance) のようなタプルを返すなら
            # ここでKinematicStateを生成するロジックを挿入
            state_vec, cov = init_kinematic
            # 例: state_vec = [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz, ax...]
            # ここはモーションモデルに応じて適宜変更
            self._kinematic_state = KinematicState(
                position=(state_vec[0], state_vec[1], state_vec[2]),
                orientation=(state_vec[3], state_vec[4], state_vec[5], state_vec[6]),
                velocity=(state_vec[7], state_vec[8], state_vec[9]),
                angular_velocity=(state_vec[10], state_vec[11], state_vec[12]),
                acceleration=(
                    (state_vec[13], state_vec[14], state_vec[15], 0.0, 0.0, 0.0)
                    if len(state_vec) > 15
                    else None
                ),
                # covariance類は適宜covから抜粋
                position_covariance=None,
                orientation_covariance=None,
                velocity_covariance=None,
                angular_velocity_covariance=None,
                acceleration_covariance=None,
            )

        self.initial_timestamp = self._header.timestamp
        self.last_update_timestamp = self._header.timestamp
        self.creation_time = datetime.datetime.now()

        # トラックメタ情報
        self.age = 1
        self.missed_count = 0
        self.is_confirmed = False
        self.is_lost = False

        # 状態履歴: [(timestamp, kinematic_state, object_info)]
        self.history: List[Tuple[float, KinematicState, ObjectInfo]] = []

    def predict(self, dt: float):
        """
        フィルタを用いて状態予測を実行し、_kinematic_stateを更新する。
        """
        predicted_state = self.filter_instance.predict(self._kinematic_state, dt)
        if isinstance(predicted_state, KinematicState):
            self._kinematic_state = predicted_state
        else:
            # 同様にpredictが生状態を返す場合はここでKinematicStateに再変換
            # 省略: predictで返る形式に合わせて処理
            pass

        self.missed_count += 1
        if self.missed_count > 10:
            self.is_lost = True

    def update(self, detection: DetectedObject3D):
        """
        新たな観測でフィルタを更新し、状態を更新する。
        必要に応じてObjectInfoやGeometricInfoも更新可能。
        """
        # 履歴に現在の状態を保存
        self.history.append(
            (self.last_update_timestamp, self._kinematic_state, self._object_info)
        )

        updated_state = self.filter_instance.update(self._kinematic_state, detection)
        if isinstance(updated_state, KinematicState):
            self._kinematic_state = updated_state
        else:
            # 同上: updateが生状態を返すならKinematicStateへ変換
            pass

        # ObjectInfo更新（クラス名や確信度などが観測に応じて変化する場合）
        if detection.get_class_name():
            self._object_info.class_name = detection.get_class_name()
        self._object_info.is_stationary = detection.is_stationary()
        self._object_info.input_sensor = (
            detection.get_input_sensor() or self._object_info.input_sensor
        )
        self._object_info.existence_probability = detection.get_existence_probability()

        self.last_update_timestamp = detection.get_timestamp()

        self.age += 1
        self.missed_count = 0
        if self.age > 2:
            self.is_confirmed = True

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

    def mark_lost(self):
        self.is_lost = True

    def get_track_id(self) -> str:
        return self.track_id
