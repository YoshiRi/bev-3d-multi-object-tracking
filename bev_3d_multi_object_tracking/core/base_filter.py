from abc import ABC, abstractmethod

from bev_3d_multi_object_tracking.common.detected_object_3d import DetectedObject3D


class BaseFilter(ABC):
    @abstractmethod
    def initialize_state(self, detection: DetectedObject3D):
        """
        最初の観測情報から初期状態を生成するメソッド。
        戻り値はKinematicState、あるいは (state_vec, covariance) など。
        """
        pass

    @abstractmethod
    def predict(self, current_state, dt: float):
        """
        現在状態を元に、dt秒後の状態を予測するメソッド。
        current_stateには initialize_state や updateで返された形式を渡す。
        戻り値も同様に同じ形式（KinematicStateや(state_vec, covariance)）で返す。
        """
        pass

    @abstractmethod
    def update(self, current_state, detection: DetectedObject3D):
        """
        観測detectionを元に、current_stateを修正する（更新）メソッド。
        current_stateは predictで返された形式を受け取り、戻り値として
        同様の形式を返す。
        """
        pass
