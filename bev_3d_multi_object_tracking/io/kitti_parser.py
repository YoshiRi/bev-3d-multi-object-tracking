"""
KITTI Tracking Dataset パーサー。

KITTI のラベルファイル (tracking) を読み込み、フレームごとに
DetectedObject3D のリストを返す。

対応データ:
    - KITTI Tracking Benchmark のラベルファイル (training/label_02/*.txt)
    - 同フォーマットの検出結果ファイル (detector の出力をそのまま使う場合)

ダウンロード先:
    https://www.cvlibs.net/datasets/kitti/eval_tracking.php
    → "Download left color images of tracking data set" + "Download label files"

KITTI ラベルフォーマット (1行 = 1物体 x 1フレーム):
    frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y

フィールド説明:
    frame       : フレーム番号 (0-indexed)
    track_id    : トラックID (-1 = DontCare)
    type        : Car / Van / Truck / Pedestrian / Cyclist / Misc / DontCare
    truncated   : 画像端での切れ具合 [0, 1]
    occluded    : 遮蔽レベル (0=見えている, 1=部分遮蔽, 2=大部分遮蔽, 3=不明)
    alpha       : 観測角 [-pi, pi]
    x1 y1 x2 y2 : 画像上の2Dバウンディングボックス
    h w l       : 3D寸法 (高さ・横幅・奥行き) [m]
    x y z       : カメラ座標でのボックス底面中心位置 [m]
    rotation_y  : カメラY軸回りの回転角 [-pi, pi]

座標変換 (カメラ座標 → BEV座標):
    KITTI カメラ座標系:
        X_cam : 右
        Y_cam : 下
        Z_cam : 前方 (奥行き)

    本ライブラリの BEV 座標系:
        x_bev : 前方  (= Z_cam)
        y_bev : 左    (= -X_cam)
        z_bev : 上    (= -Y_cam)

    位置変換 (底面中心 → ボックス中心):
        x_bev = z_cam
        y_bev = -x_cam
        z_bev = -(y_cam - h/2) = -y_cam + h/2

    寸法変換:
        KITTI (h, w, l) → 本ライブラリ (length=l, width=w, height=h)
        ※ KITTI の l が前後方向の長さ、w が左右方向の幅

    向き変換:
        yaw_bev = -rotation_y
        (カメラY下向き回転 = BEV Z上向き逆回転)
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D,
    GeometricInfo,
    Header,
    KinematicState,
    ObjectInfo,
)
from bev_3d_multi_object_tracking.core.geometry_utils import yaw_to_quaternion
from bev_3d_multi_object_tracking.io.base_parser import BaseParser

# 追跡対象とするクラス名 (DontCare などは除外する)
_VALID_TYPES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"}


class KittiTrackingParser(BaseParser):
    """
    KITTI Tracking ラベルファイルを読み込み、フレームごとに DetectedObject3D を返す。

    Args:
        label_path : ラベルファイルのパス (例: training/label_02/0000.txt)
        dt         : フレーム間隔 [秒] (KITTIのカメラは約10Hz → dt=0.1)
        frame_id   : 出力する DetectedObject3D の frame_id
        valid_types: 追跡対象のクラス名セット (None = 全クラス)
    """

    def __init__(
        self,
        label_path: str,
        dt: float = 0.1,
        frame_id: str = "map",
        valid_types: Optional[set] = None,
    ):
        self._label_path = Path(label_path)
        self._dt = dt
        self._frame_id = frame_id
        self._valid_types = valid_types if valid_types is not None else _VALID_TYPES

        # ファイルを読み込んでフレーム別に整理
        self._frames: Dict[int, List[_KittiBox]] = self._load(self._label_path)
        self._n_frames: int = max(self._frames.keys()) + 1 if self._frames else 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def parse(self, raw: int) -> List[DetectedObject3D]:
        """
        フレーム番号を受け取り、そのフレームの DetectedObject3D リストを返す。

        Args:
            raw : フレーム番号 (0-indexed)

        Returns:
            DetectedObject3D のリスト (そのフレームに存在する全物体)
        """
        boxes = self._frames.get(int(raw), [])
        return [self._to_detected_object(b) for b in boxes]

    def iter_frames(self) -> Iterator[Tuple[int, List[DetectedObject3D]]]:
        """(frame_idx, detections) を順に yield する。"""
        for frame_idx in range(self._n_frames):
            yield frame_idx, self.parse(frame_idx)

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> Dict[int, List["_KittiBox"]]:
        frames: Dict[int, List[_KittiBox]] = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                box = _KittiBox.from_line(line)
                if box is None:
                    continue
                if box.type not in self._valid_types:
                    continue
                frames.setdefault(box.frame, []).append(box)
        return frames

    def _to_detected_object(self, box: "_KittiBox") -> DetectedObject3D:
        # 座標変換: カメラ座標 → BEV座標
        x_bev = box.z_cam
        y_bev = -box.x_cam
        z_bev = -box.y_cam + box.h / 2.0  # 底面中心 → ボックス中心

        yaw_bev = -box.rotation_y

        return DetectedObject3D(
            header=Header(
                frame_id=self._frame_id,
                timestamp=box.frame * self._dt,
            ),
            kinematic_state=KinematicState(
                position=(x_bev, y_bev, z_bev),
                orientation=yaw_to_quaternion(yaw_bev),
            ),
            geometric_info=GeometricInfo(
                # KITTI: (h, w, l) → ライブラリ: (length=l, width=w, height=h)
                dimensions=(box.l, box.w, box.h),
            ),
            object_info=ObjectInfo(
                object_id=str(box.track_id),
                class_name=box.type,
            ),
        )


class _KittiBox:
    """KITTI 1行分のデータを保持する内部クラス。"""

    __slots__ = (
        "frame", "track_id", "type",
        "h", "w", "l",
        "x_cam", "y_cam", "z_cam",
        "rotation_y",
    )

    def __init__(
        self,
        frame: int,
        track_id: int,
        type_: str,
        h: float, w: float, l: float,
        x_cam: float, y_cam: float, z_cam: float,
        rotation_y: float,
    ):
        self.frame = frame
        self.track_id = track_id
        self.type = type_
        self.h = h
        self.w = w
        self.l = l
        self.x_cam = x_cam
        self.y_cam = y_cam
        self.z_cam = z_cam
        self.rotation_y = rotation_y

    @classmethod
    def from_line(cls, line: str) -> Optional["_KittiBox"]:
        """
        1行をパースする。パース失敗や DontCare の場合は None を返す。

        フォーマット:
            frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
            0     1        2    3         4        5     6  7  8  9  10 11 12 13 14 15 16
        """
        tokens = line.split()
        if len(tokens) < 17:
            return None
        try:
            frame = int(tokens[0])
            track_id = int(tokens[1])
            type_ = tokens[2]
            # tokens[3]=truncated, [4]=occluded, [5]=alpha, [6-9]=2D bbox (スキップ)
            h = float(tokens[10])
            w = float(tokens[11])
            l = float(tokens[12])
            x_cam = float(tokens[13])
            y_cam = float(tokens[14])
            z_cam = float(tokens[15])
            rotation_y = float(tokens[16])
        except (ValueError, IndexError):
            return None

        return cls(frame, track_id, type_, h, w, l, x_cam, y_cam, z_cam, rotation_y)
