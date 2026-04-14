"""
合成データジェネレータ。

実データなしでトラッカーの動作確認・デバッグができるよう、
既知の真値から検出ノイズを付加した「疑似検出結果」を生成する。

生成されるデータは DictParser が受け取れる dict 形式。

使い方:
    from scenarios.synthetic_generator import SyntheticScenario, ObjectSpec

    scenario = SyntheticScenario(
        n_frames=50,
        dt=0.1,
        objects=[
            ObjectSpec(object_id="car_0", x0=0.0, y0=0.0, yaw0=0.0, v=10.0),
            ObjectSpec(object_id="car_1", x0=20.0, y0=5.0, yaw0=0.1, v=8.0, yaw_rate=0.05),
        ]
    )

    for frame_idx in range(scenario.n_frames):
        detections = scenario.get_detections(frame_idx)   # ノイズあり
        ground_truth = scenario.get_ground_truth(frame_idx)  # 真値

または定義済みシナリオを使う:
    from scenarios.synthetic_generator import create_highway_scenario
    scenario = create_highway_scenario()
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ObjectSpec:
    """
    シナリオ内の1物体の定義。

    Attributes:
        object_id       : 物体ID (真値ラベルとして使用)
        class_name      : クラス名
        x0, y0          : 初期位置 [m]
        yaw0            : 初期向き [rad]
        v               : 速度 [m/s]
        yaw_rate        : 角速度 [rad/s] (0 = 直進)
        length, width, height : 寸法 [m]
        start_frame     : 出現フレーム (0-indexed)
        end_frame       : 消滅フレーム (None = 最終フレームまで存在)
        pos_noise_std   : 位置ノイズの標準偏差 [m]
        yaw_noise_std   : 向きノイズの標準偏差 [rad]
        detection_rate  : 検出率 (0.0-1.0, 1.0 = 必ず検出)
    """

    object_id: str
    class_name: str = "Car"
    x0: float = 0.0
    y0: float = 0.0
    yaw0: float = 0.0
    v: float = 0.0
    yaw_rate: float = 0.0
    length: float = 4.5
    width: float = 2.0
    height: float = 1.5
    start_frame: int = 0
    end_frame: Optional[int] = None
    pos_noise_std: float = 0.3
    yaw_noise_std: float = 0.05
    detection_rate: float = 1.0


class SyntheticScenario:
    """
    複数物体のシナリオを管理し、フレームごとの検出結果と真値を返す。

    物体の運動モデルは CTRV (Constant Turn Rate and Velocity)。
    フィルタと同じモデルを使うため、フィルタが原理的に収束できる設定になる。
    """

    def __init__(
        self,
        n_frames: int,
        dt: float,
        objects: List[ObjectSpec],
        seed: Optional[int] = None,
    ):
        self.n_frames = n_frames
        self.dt = dt
        self._specs = objects
        self._rng = random.Random(seed)

        # 真値トラジェクトリを事前計算
        # _trajectories[obj_id][frame] = (x, y, yaw)
        self._trajectories: Dict[str, List[Optional[Tuple[float, float, float]]]] = {}
        for spec in objects:
            self._trajectories[spec.object_id] = self._compute_trajectory(spec)

    # ------------------------------------------------------------------
    # 外部インターフェース
    # ------------------------------------------------------------------

    def get_detections(self, frame_idx: int) -> List[dict]:
        """
        ノイズを付加した疑似検出結果を返す。DictParser にそのまま渡せる形式。
        detection_rate に応じて検出を確率的に欠落させる。
        """
        result = []
        for spec in self._specs:
            pose = self._trajectories[spec.object_id][frame_idx]
            if pose is None:
                continue
            if self._rng.random() > spec.detection_rate:
                continue  # 検出ミス

            x, y, yaw = pose
            noisy_x = x + self._rng.gauss(0.0, spec.pos_noise_std)
            noisy_y = y + self._rng.gauss(0.0, spec.pos_noise_std)
            noisy_yaw = yaw + self._rng.gauss(0.0, spec.yaw_noise_std)

            result.append({
                "x": noisy_x,
                "y": noisy_y,
                "yaw": noisy_yaw,
                "length": spec.length,
                "width": spec.width,
                "height": spec.height,
                "class_name": spec.class_name,
                "timestamp": frame_idx * self.dt,
                "frame_id": "map",
            })
        return result

    def get_ground_truth(self, frame_idx: int) -> List[dict]:
        """ノイズなしの真値を返す。評価指標計算などに使用する。"""
        result = []
        for spec in self._specs:
            pose = self._trajectories[spec.object_id][frame_idx]
            if pose is None:
                continue
            x, y, yaw = pose
            result.append({
                "object_id": spec.object_id,
                "x": x,
                "y": y,
                "yaw": yaw,
                "length": spec.length,
                "width": spec.width,
                "height": spec.height,
                "class_name": spec.class_name,
                "timestamp": frame_idx * self.dt,
                "frame_id": "map",
            })
        return result

    def iter_frames(self):
        """(frame_idx, detections, ground_truth) のイテレータ。"""
        for i in range(self.n_frames):
            yield i, self.get_detections(i), self.get_ground_truth(i)

    # ------------------------------------------------------------------
    # 内部: トラジェクトリ計算
    # ------------------------------------------------------------------

    def _compute_trajectory(
        self, spec: ObjectSpec
    ) -> List[Optional[Tuple[float, float, float]]]:
        """各フレームの (x, y, yaw) を CTRV モデルで計算する。存在しないフレームは None。"""
        end = spec.end_frame if spec.end_frame is not None else self.n_frames - 1
        traj: List[Optional[Tuple[float, float, float]]] = [None] * self.n_frames

        x, y, yaw = spec.x0, spec.y0, spec.yaw0
        for f in range(self.n_frames):
            if f < spec.start_frame or f > end:
                traj[f] = None
            else:
                traj[f] = (x, y, yaw)

            # 次フレームへ CTRV 伝搬 (最終フレームの後は不要だが計算は続ける)
            yr = spec.yaw_rate
            v = spec.v
            dt = self.dt
            if abs(yr) < 1e-5:
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
            else:
                x += (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
                y += (v / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
                yaw += yr * dt

        return traj


# ------------------------------------------------------------------
# 定義済みシナリオ
# ------------------------------------------------------------------

def create_highway_scenario(
    n_frames: int = 100,
    dt: float = 0.1,
    seed: int = 42,
) -> SyntheticScenario:
    """
    高速道路シナリオ: 複数車両が同方向に直進。途中で1台が出現・消滅。

    car_0: 先行車 (v=15m/s)
    car_1: 後続車 (v=20m/s, 追い越し)
    car_2: 途中から出現する合流車 (start_frame=30)
    """
    return SyntheticScenario(
        n_frames=n_frames,
        dt=dt,
        seed=seed,
        objects=[
            ObjectSpec(
                object_id="car_0",
                x0=0.0, y0=0.0, yaw0=0.0,
                v=15.0,
                pos_noise_std=0.3,
            ),
            ObjectSpec(
                object_id="car_1",
                x0=-30.0, y0=3.5, yaw0=0.0,
                v=20.0,
                pos_noise_std=0.3,
            ),
            ObjectSpec(
                object_id="car_2",
                x0=20.0, y0=7.0, yaw0=-0.1,
                v=18.0,
                start_frame=30,
                end_frame=80,
                pos_noise_std=0.4,
            ),
        ],
    )


def create_intersection_scenario(
    n_frames: int = 80,
    dt: float = 0.1,
    seed: int = 42,
) -> SyntheticScenario:
    """
    交差点シナリオ: 直進車と右折車が交差。

    car_0: x方向に直進 (v=10m/s)
    car_1: y方向から来てx方向へ右折 (yaw_rate あり)
    pedestrian_0: ゆっくり歩く歩行者
    """
    return SyntheticScenario(
        n_frames=n_frames,
        dt=dt,
        seed=seed,
        objects=[
            ObjectSpec(
                object_id="car_0",
                x0=-40.0, y0=0.0, yaw0=0.0,
                v=10.0,
                pos_noise_std=0.3,
            ),
            ObjectSpec(
                object_id="car_1",
                x0=0.0, y0=-30.0, yaw0=math.pi / 2,
                v=8.0,
                yaw_rate=-0.15,   # 右折
                end_frame=60,
                pos_noise_std=0.4,
            ),
            ObjectSpec(
                object_id="pedestrian_0",
                class_name="Pedestrian",
                x0=5.0, y0=-10.0, yaw0=math.pi / 2,
                v=1.5,
                length=0.5, width=0.5, height=1.7,
                pos_noise_std=0.15,
            ),
        ],
    )


def create_occlusion_scenario(
    n_frames: int = 60,
    dt: float = 0.1,
    seed: int = 42,
) -> SyntheticScenario:
    """
    隠蔽シナリオ: 検出率を下げて miss が多発する状況を再現。

    トラッカーの max_misses_to_lose チューニングに使用する。
    """
    return SyntheticScenario(
        n_frames=n_frames,
        dt=dt,
        seed=seed,
        objects=[
            ObjectSpec(
                object_id="car_0",
                x0=0.0, y0=0.0, yaw0=0.0,
                v=10.0,
                detection_rate=0.7,   # 30% miss
                pos_noise_std=0.5,
            ),
            ObjectSpec(
                object_id="car_1",
                x0=10.0, y0=5.0, yaw0=0.0,
                v=12.0,
                detection_rate=0.5,   # 50% miss
                pos_noise_std=0.5,
            ),
        ],
    )
