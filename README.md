# BEV 3D Multi-Object Tracking

BEV (Bird's Eye View) 空間における 3D 多物体追跡ライブラリ。

センサーの種類や入出力フォーマットに依存しない設計を目指しており、
**入力パーサー → 内部処理 → 出力パーサー** という 3 層のアーキテクチャで構成される。

---

## 設計思想

### 入出力の分離

追跡アルゴリズムの本質はデータフォーマットと無関係であるべき、という考え方に基づいている。

```
[ 任意の入力 ]  →  [ 内部表現 ]  →  [ トラッキング処理 ]  →  [ 内部表現 ]  →  [ 任意の出力 ]

 ROS2 msg            DetectedObject3D     EKF / Data               Tracklet          ROS2 msg
 CSV                                      Association                                 CSV
 カスタム形式                                                                          カスタム形式
```

- **入力パーサー**: 任意のフォーマット (ROS msg, CSV, JSON, etc.) を `DetectedObject3D` に変換する責務のみを持つ
- **内部処理**: `DetectedObject3D` と `Tracklet` だけを扱い、フォーマットを一切知らない
- **出力パーサー**: `Tracklet` を任意のフォーマットに変換する責務のみを持つ

この構造により、入出力フォーマットの追加・変更がトラッキングロジックに影響しない。

### 内部表現の役割

内部表現は「このライブラリが必要とする最小限の情報」を定義したもの。
外部フォーマットが持つ余分な情報はパーサー側で捨て、足りない情報はデフォルト値で補う。

```python
# 入力パーサーの例 (ROS2 → 内部表現)
def from_ros2_detection(msg: Detection3D) -> DetectedObject3D:
    header = Header(frame_id=msg.header.frame_id, timestamp=msg.header.stamp.sec)
    kinematic = KinematicState(
        position=(msg.bbox.center.position.x, msg.bbox.center.position.y, msg.bbox.center.position.z),
        orientation=(msg.bbox.center.orientation.x, ...),
    )
    ...
    return DetectedObject3D(header, kinematic, geom, obj_info)

# 出力パーサーの例 (Tracklet → ROS2)
def to_ros2_tracked_object(tracklet: Tracklet) -> TrackedObject:
    kin = tracklet.get_current_kinematic()
    msg = TrackedObject()
    msg.object_id = tracklet.get_track_id()
    msg.kinematics.pose_with_covariance.pose.position.x = kin.position[0]
    ...
    return msg
```

---

## アーキテクチャ

### モジュール構成

```
bev_3d_multi_object_tracking/
├── core/                        # 内部表現・抽象インターフェース
│   ├── detected_object_3d.py   # 検出物体の内部表現
│   ├── tracklet.py             # トラック状態管理
│   ├── base_filter.py          # フィルタの抽象インターフェース
│   └── base_data_association.py # データアソシエーションの抽象インターフェース
├── filters/                     # フィルタ実装
│   └── vehicletracker.py       # EKF (CTRVモデル) による車両追跡
└── data_association/            # データアソシエーション実装
    └── cost_matrix_based_association.py  # Hungarian / GNN
```

パーサー (入出力変換) はこのライブラリの外側で実装する想定。

### 処理フロー

1フレームの処理は以下の流れになる:

```
新フレームの検出群 (List[DetectedObject3D])
    │
    ▼
[Data Association]  既存トラックと検出をマッチング
    │
    ├─ マッチしたペア → Tracklet.update(detection)
    ├─ マッチしなかったトラック → Tracklet.predict() のみ (missed_count 増加)
    └─ マッチしなかった検出 → 新規 Tracklet 生成
    │
    ▼
更新済み Tracklet 群 (List[Tracklet])
```

---

## 内部表現の詳細

### DetectedObject3D

1フレームで得られた1物体の検出情報を表す。追跡前の「生の観測」に相当する。

| クラス | 役割 |
|--------|------|
| `Header` | タイムスタンプ・座標フレームID |
| `KinematicState` | 位置・姿勢(クォータニオン)・速度・角速度・加速度、および各共分散 |
| `GeometricInfo` | 物体の寸法 (length, width, height) |
| `ObjectInfo` | オブジェクトID・クラス名・静止フラグ・存在確率 |

### Tracklet

複数フレームにわたるトラックの状態を管理する。

| 属性 | 説明 |
|------|------|
| `track_id` | 追跡 ID (UUID) |
| `age` | トラックが生成されてからのフレーム数 |
| `missed_count` | 連続してマッチしなかったフレーム数 |
| `is_confirmed` | 一定フレーム以上マッチが続き確定したか |
| `is_lost` | 一定フレーム以上見失い消滅判定されたか |
| `history` | 過去の状態履歴 |

---

## 拡張方法

### 新しいフィルタを追加する

`BaseFilter` を継承し、3つのメソッドを実装する。

```python
from bev_3d_multi_object_tracking.core.base_filter import BaseFilter
from bev_3d_multi_object_tracking.core.detected_object_3d import DetectedObject3D

class MyFilter(BaseFilter):
    def initialize_state(self, detection: DetectedObject3D):
        # 初期状態を生成して返す
        # 戻り値の形式は predict/update と一致していれば自由
        ...

    def predict(self, current_state, dt: float):
        # current_state は initialize_state / update の戻り値と同じ形式
        ...

    def update(self, current_state, detection: DetectedObject3D):
        ...
```

### 新しいデータアソシエーションを追加する

`BaseDataAssociation` を継承する。

```python
from bev_3d_multi_object_tracking.core.base_data_association import BaseDataAssociation

class MyAssociation(BaseDataAssociation):
    def associate(self, tracklets, detections):
        # matched_pairs, unmatched_tracklets, unmatched_detections を返す
        ...
```

コスト行列ベースのアソシエーションを使う場合は `CostMatrixBasedAssociation` を継承し、
`solve_assignment()` だけを実装すれば割当ロジックを差し替えられる。

```python
from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
    CostMatrixBasedAssociation,
)

class MyAssignment(CostMatrixBasedAssociation):
    def solve_assignment(self, cost_matrix):
        # 独自の割当アルゴリズムを実装
        row_idx, col_idx = ...
        return row_idx, col_idx
```

---

## 使用例

```python
from bev_3d_multi_object_tracking.core.detected_object_3d import (
    DetectedObject3D, Header, KinematicState, GeometricInfo, ObjectInfo,
)
from bev_3d_multi_object_tracking.core.tracklet import Tracklet
from bev_3d_multi_object_tracking.filters.vehicletracker import VehicleTracker
from bev_3d_multi_object_tracking.data_association.cost_matrix_based_association import (
    HungarianCostMatrixAssociation,
)

# --- 入力パーサー (任意のフォーマットから変換する箇所) ---
def parse_detection(raw) -> DetectedObject3D:
    return DetectedObject3D(
        header=Header(frame_id="base_link", timestamp=raw.timestamp),
        kinematic_state=KinematicState(position=(raw.x, raw.y, 0.0), orientation=(0, 0, 0, 1)),
        geometric_info=GeometricInfo(dimensions=(raw.length, raw.width, raw.height)),
        object_info=ObjectInfo(class_name=raw.class_name),
    )

# --- コスト関数の定義 ---
def euclidean_cost(tracklet: Tracklet, detection: DetectedObject3D) -> float:
    tp = tracklet.get_current_kinematic().position
    dp = detection.get_position()
    return ((tp[0] - dp[0]) ** 2 + (tp[1] - dp[1]) ** 2) ** 0.5

# --- トラッカー初期化 ---
association = HungarianCostMatrixAssociation(cost_func=euclidean_cost, max_distance=5.0)
tracklets = []

# --- フレームループ ---
for frame in sensor_frames:
    detections = [parse_detection(d) for d in frame.detections]
    dt = frame.dt

    # 予測ステップ
    for t in tracklets:
        t.predict(dt)

    # アソシエーション
    matched, unmatched_tracks, unmatched_dets = association.associate(tracklets, detections)

    # 更新・消滅・生成
    for tracklet, detection in matched:
        tracklet.update(detection)
    for tracklet in unmatched_tracks:
        if tracklet.missed_count > 5:
            tracklets.remove(tracklet)
    for det in unmatched_dets:
        tracklets.append(Tracklet(VehicleTracker(), det))

    # --- 出力パーサー (任意のフォーマットに変換する箇所) ---
    output = [serialize_tracklet(t) for t in tracklets if t.is_confirmed]
```

---

## 開発環境

```bash
# 依存関係のインストール
pip install numpy scipy

# テスト実行
python -m pytest tests/ -v
```

### 依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `numpy` | 行列演算 (EKF, コスト行列) |
| `scipy` | Hungarian法 (`linear_sum_assignment`) |
