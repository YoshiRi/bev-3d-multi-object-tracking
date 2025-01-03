import math
from typing import Optional

import numpy as np
from bev_3d_multi_object_tracking.common.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)

from bev_3d_multi_object_tracking.core.base_filter import BaseFilter


class VehicleTracker(BaseFilter):
    """
    シンプルなEKFベースの車両トラッカー
    - Position State: [x, y, yaw, v, yaw_rate]
    - Shape State: [length, width, height]

    予測 (predict) と 更新 (update) をそれぞれ行い、
    set_measurement_matrix()で観測行列を外部から設定可能にする。
    また、to_detected_object_3d()で現在の推定状態をDetectedObject3Dに変換。
    """

    def __init__(self):
        # ---------------------------
        # (1) 状態ベクトルおよび共分散初期化
        # ---------------------------
        self._pos_state = None  # shape=(5,)  -> [x, y, yaw, v, yaw_rate]
        self._pos_cov = None  # shape=(5,5)
        self._shape_state = None  # shape=(3,)  -> [length, width, height]
        self._shape_cov = None  # shape=(3,3)

        # ---------------------------
        # (2) プロセスノイズ・観測ノイズ
        # ---------------------------
        # (2a) 位置のプロセスノイズ行列
        self._Q_pos = np.diag([0.2, 0.2, 0.01, 0.5, 0.02])
        # (2b) 形状のプロセスノイズ行列
        self._Q_shape = np.diag([0.1, 0.1, 0.1])

        # (2c) 観測ノイズ(位置)
        # 例: x, y, yaw を観測
        self._R_pos = np.diag([0.2, 0.2, 0.05])
        # (2d) 観測ノイズ(形状)
        self._R_shape = np.diag([0.5, 0.3, 0.3])

        # ---------------------------
        # (3) 観測行列(Measurement Matrix)のデフォルト設定
        # ---------------------------
        # デフォルトでは x, y, yaw の3次元観測を想定した 3x5 行列
        self._pos_meas_matrix = np.zeros((3, 5), dtype=float)
        self._pos_meas_matrix[0, 0] = 1.0  # x
        self._pos_meas_matrix[1, 1] = 1.0  # y
        self._pos_meas_matrix[2, 2] = 1.0  # yaw

        # 形状は [length, width, height] 3x3 の恒等行列
        self._shape_meas_matrix = np.eye(3, dtype=float)

    # --------------------------------------------------------------------------
    # 追加メソッド: 観測行列を外部から設定
    # --------------------------------------------------------------------------
    def set_measurement_matrix(
        self, pos_H: Optional[np.ndarray] = None, shape_H: Optional[np.ndarray] = None
    ):
        """
        位置・形状の観測行列を外部から指定し、計算ロジックを動的に切り替え可能にする。

        :param pos_H:   shape=(m,5) の観測行列(位置)
        :param shape_H: shape=(n,3) の観測行列(形状)
        """
        if pos_H is not None:
            self._pos_meas_matrix = pos_H
        if shape_H is not None:
            self._shape_meas_matrix = shape_H

    # --------------------------------------------------------------------------
    # 追加メソッド: フィルタ内部状態をDetectedObject3Dとして出力
    # --------------------------------------------------------------------------
    def to_detected_object_3d(
        self, frame_id: str, timestamp: float
    ) -> DetectedObject3D:
        """
        現在の推定状態(_pos_state, _shape_state)からDetectedObject3Dを生成し返す。

        :param frame_id:    座標系のID
        :param timestamp:   タイムスタンプ
        :return:            DetectedObject3D
        """
        if self._pos_state is None or self._shape_state is None:
            raise ValueError("Filter state not initialized.")

        x, y, yaw, v, yaw_rate = self._pos_state
        length, width, height = self._shape_state

        # orientationはz軸回りのyawのみ考慮 (roll,pitchは0)
        # quaternion化(簡易):
        # yaw -> (qx,qy,qz,qw) = (0,0,sin(yaw/2), cos(yaw/2))
        half_yaw = yaw * 0.5
        sin_hy = math.sin(half_yaw)
        cos_hy = math.cos(half_yaw)
        orientation = (0.0, 0.0, sin_hy, cos_hy)

        # velocityは簡単化して [vx, vy, 0]
        vx = v * math.cos(yaw)
        vy = v * math.sin(yaw)

        header = Header(frame_id=frame_id, timestamp=timestamp)
        kinematic = KinematicState(
            position=(x, y, 0.0),
            orientation=orientation,
            velocity=(vx, vy, 0.0),
            angular_velocity=(0.0, 0.0, yaw_rate),
        )
        geom = GeometricInfo(dimensions=(length, width, height))
        obj_info = ObjectInfo(
            object_id=None,  # 必要に応じて
            class_name=None,  # 車種など
            is_stationary=(v < 0.1),  # 簡易判断
            input_sensor=None,
        )

        return DetectedObject3D(
            header=header,
            kinematic_state=kinematic,
            geometric_info=geom,
            object_info=obj_info,
        )

    # --------------------------------------------------------------------------
    # BaseFilterインターフェイスの実装
    # --------------------------------------------------------------------------
    def initialize_state(self, detection: DetectedObject3D):
        """
        初期状態ベクトルの設定:
        - 位置は x, y, yaw を検出から取得し, v=0, yaw_rate=0 で開始
        - 形状は length, width, height を検出から取得
        """
        pos = detection.get_position()  # (x, y, z)
        yaw = detection.get_yaw()
        v = 0.0
        yr = 0.0
        self._pos_state = np.array([pos[0], pos[1], yaw, v, yr], dtype=float)
        self._pos_cov = np.eye(5, dtype=float) * 0.5

        lwh = detection.get_dimensions()  # (l, w, h)
        self._shape_state = np.array([lwh[0], lwh[1], lwh[2]], dtype=float)
        self._shape_cov = np.eye(3, dtype=float) * 0.5

        return (self._pos_state, self._pos_cov), (self._shape_state, self._shape_cov)

    def predict(self, current_state, dt: float):
        (pos_state, pos_cov), (shape_state, shape_cov) = current_state

        # 1. 位置の予測
        pos_pred, pos_cov_pred = self._predict_position(pos_state, pos_cov, dt)

        # 2. 形状の予測（ほとんど変化しない想定で共分散を広げるだけ）
        shape_pred, shape_cov_pred = self._predict_shape(shape_state, shape_cov)

        # 内部にも保存しておく(状況に応じて)
        self._pos_state, self._pos_cov = pos_pred, pos_cov_pred
        self._shape_state, self._shape_cov = shape_pred, shape_cov_pred

        return (pos_pred, pos_cov_pred), (shape_pred, shape_cov_pred)

    def update(self, predicted_state, detection: DetectedObject3D):
        (pos_pred, pos_cov_pred), (shape_pred, shape_cov_pred) = predicted_state

        # 位置更新
        pos_upd, pos_cov_upd = self._update_position(pos_pred, pos_cov_pred, detection)

        # 形状更新
        shape_upd, shape_cov_upd = self._update_shape(
            shape_pred, shape_cov_pred, detection
        )

        # 内部にも保存
        self._pos_state, self._pos_cov = pos_upd, pos_cov_upd
        self._shape_state, self._shape_cov = shape_upd, shape_cov_upd

        return (pos_upd, pos_cov_upd), (shape_upd, shape_cov_upd)

    # --------------------------------------------------------------------------
    # 個別の予測/更新ロジック (位置)
    # --------------------------------------------------------------------------
    def _predict_position(self, pos_state: np.ndarray, pos_cov: np.ndarray, dt: float):
        """
        CTRVモデルで予測 (x, y, yaw, v, yaw_rate)
        """
        x, y, yaw, v, yr = pos_state
        if abs(yr) < 1e-5:
            x_new = x + v * math.cos(yaw) * dt
            y_new = y + v * math.sin(yaw) * dt
            yaw_new = yaw
        else:
            x_new = x + (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            y_new = y + (v / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
            yaw_new = yaw + yr * dt

        pos_state_pred = np.array([x_new, y_new, yaw_new, v, yr], dtype=float)
        # ヤコビアンF (簡易)
        F = self._calc_ctrv_jacobian(pos_state, dt)
        pos_cov_pred = F @ pos_cov @ F.T + self._Q_pos
        return pos_state_pred, pos_cov_pred

    def _update_position(
        self,
        pos_state_pred: np.ndarray,
        pos_cov_pred: np.ndarray,
        detection: DetectedObject3D,
    ):
        """
        pos_meas_matrix( _pos_meas_matrix )に従って観測更新。
        デフォルト: x,y,yaw の3次元観測 (速度等は観測されない)
        """
        # 観測ベクトル z を組み立て(例: [x, y, yaw])
        z = np.array(
            [
                detection.get_position()[0],
                detection.get_position()[1],
                detection.get_yaw(),
            ],
            dtype=float,
        )

        H = self._pos_meas_matrix  # shape=(m,5)
        # 観測モデル h(x) = H * x
        # x_pred: 5次元
        x_pred = pos_state_pred
        hx = H @ x_pred  # shape=(m,)

        # イノベーション
        y = z - hx
        S = H @ pos_cov_pred @ H.T + self._R_pos
        K = pos_cov_pred @ H.T @ np.linalg.inv(S)

        pos_state_upd = x_pred + K @ y
        I5 = np.eye(5)
        pos_cov_upd = (I5 - K @ H) @ pos_cov_pred

        return pos_state_upd, pos_cov_upd

    # --------------------------------------------------------------------------
    # 個別の予測/更新ロジック (形状)
    # --------------------------------------------------------------------------
    def _predict_shape(self, shape_state: np.ndarray, shape_cov: np.ndarray):
        # シンプルに共分散にプロセスノイズを加えるのみ
        shape_state_pred = shape_state.copy()
        shape_cov_pred = shape_cov + self._Q_shape
        return shape_state_pred, shape_cov_pred

    def _update_shape(
        self,
        shape_state_pred: np.ndarray,
        shape_cov_pred: np.ndarray,
        detection: DetectedObject3D,
    ):
        """
        shape_meas_matrix(_shape_meas_matrix) に従って観測更新。
        デフォルト: length, width, height の3次元観測
        """
        z_shape = np.array(list(detection.get_dimensions()), dtype=float)

        H_s = self._shape_meas_matrix  # shape=(n,3)
        hx_s = H_s @ shape_state_pred  # shape=(n,)

        y_s = z_shape - hx_s
        S_s = H_s @ shape_cov_pred @ H_s.T + self._R_shape
        K_s = shape_cov_pred @ H_s.T @ np.linalg.inv(S_s)

        shape_state_upd = shape_state_pred + K_s @ y_s
        I3 = np.eye(3)
        shape_cov_upd = (I3 - K_s @ H_s) @ shape_cov_pred

        return shape_state_upd, shape_cov_upd

    # --------------------------------------------------------------------------
    # CTRVヤコビアン(簡易)
    # --------------------------------------------------------------------------
    def _calc_ctrv_jacobian(self, pos_state: np.ndarray, dt: float):
        """
        pos_state: [x, y, yaw, v, yaw_rate]
        戻り値: 5x5行列
        """
        x, y, yaw, v, yr = pos_state
        F = np.eye(5)
        # 最小限の近似
        if abs(yr) < 1e-5:
            F[0, 2] = -v * math.sin(yaw) * dt
            F[0, 3] = math.cos(yaw) * dt
            F[1, 2] = v * math.cos(yaw) * dt
            F[1, 3] = math.sin(yaw) * dt
        else:
            # 省略 (厳密には d(px)/d(yaw), d(px)/d(v) などを計算)
            pass
        return F
