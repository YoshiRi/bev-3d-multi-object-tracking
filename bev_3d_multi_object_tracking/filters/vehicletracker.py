import math
from typing import Optional, Tuple

import numpy as np

from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.geometry_utils import yaw_to_quaternion


class VehicleTracker(BaseFilter):
    """
    EKFベースの車両トラッカー。CTRVモーションモデルを使用する。

    フィルタ内部状態 (filter_state) の形式:
        ((pos_state, pos_cov), (shape_state, shape_cov))
        - pos_state  : np.ndarray shape=(5,)  [x, y, yaw, v, yaw_rate]
        - pos_cov    : np.ndarray shape=(5,5)
        - shape_state: np.ndarray shape=(3,)  [length, width, height]
        - shape_cov  : np.ndarray shape=(3,3)

    この形式は BaseFilter の規約に従い Tracklet からは不透明 (opaque) として扱われる。
    外部への変換は to_kinematic_state() で行う。
    """

    def __init__(self):
        # プロセスノイズ
        self._Q_pos = np.diag([0.2, 0.2, 0.01, 0.5, 0.02])
        self._Q_shape = np.diag([0.1, 0.1, 0.1])

        # 観測ノイズ (x, y, yaw)
        self._R_pos = np.diag([0.2, 0.2, 0.05])
        # 観測ノイズ (length, width, height)
        self._R_shape = np.diag([0.5, 0.3, 0.3])

        # 観測行列: デフォルトは x, y, yaw の3次元観測
        self._pos_meas_matrix = np.zeros((3, 5), dtype=float)
        self._pos_meas_matrix[0, 0] = 1.0  # x
        self._pos_meas_matrix[1, 1] = 1.0  # y
        self._pos_meas_matrix[2, 2] = 1.0  # yaw

        # 形状は length, width, height の恒等観測
        self._shape_meas_matrix = np.eye(3, dtype=float)

    def set_measurement_matrix(
        self,
        pos_H: Optional[np.ndarray] = None,
        shape_H: Optional[np.ndarray] = None,
    ):
        """観測行列を外部から差し替える。"""
        if pos_H is not None:
            self._pos_meas_matrix = pos_H
        if shape_H is not None:
            self._shape_meas_matrix = shape_H

    # ------------------------------------------------------------------
    # BaseFilter インターフェース実装
    # ------------------------------------------------------------------

    def initialize_state(self, detection: DetectedObject3D):
        pos = detection.get_position()
        yaw = detection.get_yaw()
        pos_state = np.array([pos[0], pos[1], yaw, 0.0, 0.0], dtype=float)
        pos_cov = np.eye(5, dtype=float) * 0.5

        lwh = detection.get_dimensions()
        shape_state = np.array(lwh, dtype=float)
        shape_cov = np.eye(3, dtype=float) * 0.5

        return (pos_state, pos_cov), (shape_state, shape_cov)

    def predict(self, filter_state, dt: float):
        (pos_state, pos_cov), (shape_state, shape_cov) = filter_state

        pos_pred, pos_cov_pred = self._predict_position(pos_state, pos_cov, dt)
        shape_pred, shape_cov_pred = self._predict_shape(shape_state, shape_cov)

        return (pos_pred, pos_cov_pred), (shape_pred, shape_cov_pred)

    def update(self, filter_state, detection: DetectedObject3D):
        (pos_pred, pos_cov_pred), (shape_pred, shape_cov_pred) = filter_state

        pos_upd, pos_cov_upd = self._update_position(pos_pred, pos_cov_pred, detection)
        shape_upd, shape_cov_upd = self._update_shape(
            shape_pred, shape_cov_pred, detection
        )

        return (pos_upd, pos_cov_upd), (shape_upd, shape_cov_upd)

    def to_kinematic_state(self, filter_state) -> KinematicState:
        """
        内部 filter_state から KinematicState を生成する。
        位置の x-y 共分散は KinematicState.position_covariance に格納する
        (Mahalanobis 距離コスト関数で利用可能)。
        """
        (pos_state, pos_cov), (shape_state, _shape_cov) = filter_state
        x, y, yaw, v, yaw_rate = pos_state

        orientation = yaw_to_quaternion(yaw)
        vx = v * math.cos(yaw)
        vy = v * math.sin(yaw)

        # x-y の 2x2 共分散を flat tuple に変換
        xy_cov = tuple(pos_cov[:2, :2].flatten().tolist())

        return KinematicState(
            position=(x, y, 0.0),
            orientation=orientation,
            velocity=(vx, vy, 0.0),
            angular_velocity=(0.0, 0.0, yaw_rate),
            position_covariance=xy_cov,
        )

    def to_detected_object_3d(
        self, filter_state, frame_id: str, timestamp: float
    ) -> DetectedObject3D:
        """filter_state から DetectedObject3D を生成する (外部出力用)。"""
        (pos_state, _pos_cov), (shape_state, _shape_cov) = filter_state
        x, y, yaw, v, yaw_rate = pos_state
        length, width, height = shape_state

        orientation = yaw_to_quaternion(yaw)
        vx = v * math.cos(yaw)
        vy = v * math.sin(yaw)

        return DetectedObject3D(
            header=Header(frame_id=frame_id, timestamp=timestamp),
            kinematic_state=KinematicState(
                position=(x, y, 0.0),
                orientation=orientation,
                velocity=(vx, vy, 0.0),
                angular_velocity=(0.0, 0.0, yaw_rate),
            ),
            geometric_info=GeometricInfo(dimensions=(length, width, height)),
            object_info=ObjectInfo(is_stationary=(v < 0.1)),
        )

    # ------------------------------------------------------------------
    # 内部実装: 位置の予測・更新
    # ------------------------------------------------------------------

    def _predict_position(
        self, pos_state: np.ndarray, pos_cov: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y, yaw, v, yr = pos_state
        if abs(yr) < 1e-5:
            x_new = x + v * math.cos(yaw) * dt
            y_new = y + v * math.sin(yaw) * dt
            yaw_new = yaw
        else:
            x_new = x + (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            y_new = y + (v / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
            yaw_new = yaw + yr * dt

        pos_pred = np.array([x_new, y_new, yaw_new, v, yr], dtype=float)
        F = self._calc_ctrv_jacobian(pos_state, dt)
        pos_cov_pred = F @ pos_cov @ F.T + self._Q_pos
        return pos_pred, pos_cov_pred

    def _update_position(
        self,
        pos_pred: np.ndarray,
        pos_cov_pred: np.ndarray,
        detection: DetectedObject3D,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = np.array(
            [detection.get_position()[0], detection.get_position()[1], detection.get_yaw()],
            dtype=float,
        )
        H = self._pos_meas_matrix
        y = z - H @ pos_pred
        S = H @ pos_cov_pred @ H.T + self._R_pos
        K = pos_cov_pred @ H.T @ np.linalg.inv(S)

        pos_upd = pos_pred + K @ y
        pos_cov_upd = (np.eye(5) - K @ H) @ pos_cov_pred
        return pos_upd, pos_cov_upd

    # ------------------------------------------------------------------
    # 内部実装: 形状の予測・更新
    # ------------------------------------------------------------------

    def _predict_shape(
        self, shape_state: np.ndarray, shape_cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return shape_state.copy(), shape_cov + self._Q_shape

    def _update_shape(
        self,
        shape_pred: np.ndarray,
        shape_cov_pred: np.ndarray,
        detection: DetectedObject3D,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z_shape = np.array(list(detection.get_dimensions()), dtype=float)
        H_s = self._shape_meas_matrix
        y_s = z_shape - H_s @ shape_pred
        S_s = H_s @ shape_cov_pred @ H_s.T + self._R_shape
        K_s = shape_cov_pred @ H_s.T @ np.linalg.inv(S_s)

        shape_upd = shape_pred + K_s @ y_s
        shape_cov_upd = (np.eye(3) - K_s @ H_s) @ shape_cov_pred
        return shape_upd, shape_cov_upd

    # ------------------------------------------------------------------
    # CTRVヤコビアン
    # ------------------------------------------------------------------

    def _calc_ctrv_jacobian(self, pos_state: np.ndarray, dt: float) -> np.ndarray:
        _, _, yaw, v, yr = pos_state
        F = np.eye(5)
        if abs(yr) < 1e-5:
            F[0, 2] = -v * math.sin(yaw) * dt
            F[0, 3] = math.cos(yaw) * dt
            F[1, 2] = v * math.cos(yaw) * dt
            F[1, 3] = math.sin(yaw) * dt
        else:
            F[0, 2] = (v / yr) * (math.cos(yaw + yr * dt) - math.cos(yaw))
            F[0, 3] = (1.0 / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            F[0, 4] = (v / yr**2) * (math.sin(yaw) - math.sin(yaw + yr * dt)) + (
                v / yr
            ) * math.cos(yaw + yr * dt) * dt
            F[1, 2] = (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            F[1, 3] = (1.0 / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
            F[1, 4] = (v / yr**2) * (-math.cos(yaw) + math.cos(yaw + yr * dt)) + (
                v / yr
            ) * math.sin(yaw + yr * dt) * dt
            F[2, 4] = dt
        return F
