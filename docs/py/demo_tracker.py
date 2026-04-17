"""
BEV 3D Multi-Object Tracking - Pyodide Demo Bundle

Self-contained Python module that runs entirely in the browser via Pyodide.
All tracking classes are inlined (no relative imports).

Entry point: run_demo(params_json: str) -> str
  params_json: JSON with scenario/tracking parameters
  returns:    JSON with per-frame GT + track data and final metrics
"""

import json
import math
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ===========================================================================
# Core data structures
# ===========================================================================

def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    h = yaw * 0.5
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _quat_to_yaw(q: Tuple[float, float, float, float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Header:
    def __init__(self, frame_id: str = "", timestamp: float = 0.0):
        self.frame_id = frame_id
        self.timestamp = timestamp


class KinematicState:
    def __init__(self, position, orientation, position_covariance=None,
                 velocity=None, angular_velocity=None):
        self.position = position
        self.orientation = orientation
        self.position_covariance = position_covariance
        self.velocity = velocity
        self.angular_velocity = angular_velocity

    def get_yaw(self) -> float:
        return _quat_to_yaw(self.orientation)


class GeometricInfo:
    def __init__(self, dimensions):
        self.dimensions = dimensions  # (length, width, height)


class ObjectInfo:
    def __init__(self, object_id=None, class_name=None,
                 is_stationary=False, existence_probability=1.0):
        self.object_id = object_id if object_id is not None else str(uuid.uuid4())
        self.class_name = class_name
        self.is_stationary = is_stationary
        self.existence_probability = existence_probability


class DetectedObject3D:
    def __init__(self, header, kinematic_state, geometric_info, object_info):
        self._header = header
        self._kinematic = kinematic_state
        self._geometric = geometric_info
        self._info = object_info

    def get_frame_id(self): return self._header.frame_id
    def get_timestamp(self): return self._header.timestamp
    def get_object_id(self): return self._info.object_id
    def get_class_name(self): return self._info.class_name
    def get_existence_probability(self): return self._info.existence_probability
    def is_stationary(self): return self._info.is_stationary
    def get_input_sensor(self): return None
    def get_position(self): return self._kinematic.position
    def get_yaw(self): return self._kinematic.get_yaw()
    def get_dimensions(self): return self._geometric.dimensions


# ===========================================================================
# DictParser
# ===========================================================================

class DictParser:
    def parse(self, raw):
        if isinstance(raw, dict):
            return [self._one(raw)]
        return [self._one(d) for d in raw]

    def _one(self, d):
        yaw = float(d.get("yaw", 0.0))
        return DetectedObject3D(
            header=Header(
                frame_id=str(d.get("frame_id", "")),
                timestamp=float(d.get("timestamp", 0.0)),
            ),
            kinematic_state=KinematicState(
                position=(float(d["x"]), float(d["y"]), float(d.get("z", 0.0))),
                orientation=_yaw_to_quat(yaw),
            ),
            geometric_info=GeometricInfo(dimensions=(
                float(d.get("length", 1.0)),
                float(d.get("width", 1.0)),
                float(d.get("height", 1.0)),
            )),
            object_info=ObjectInfo(
                object_id=d.get("object_id"),
                class_name=d.get("class_name"),
                is_stationary=bool(d.get("is_stationary", False)),
                existence_probability=float(d.get("existence_probability", 1.0)),
            ),
        )


# ===========================================================================
# VehicleTracker (CTRV EKF)
# ===========================================================================

class VehicleTracker:
    """EKF-based CTRV vehicle tracker."""

    def __init__(self):
        self._Q_pos = np.diag([0.2, 0.2, 0.01, 0.5, 0.02])
        self._Q_shape = np.diag([0.1, 0.1, 0.1])
        self._R_pos = np.diag([0.2, 0.2, 0.05])
        self._R_shape = np.diag([0.5, 0.3, 0.3])

        self._H_pos = np.zeros((3, 5))
        self._H_pos[0, 0] = 1.0
        self._H_pos[1, 1] = 1.0
        self._H_pos[2, 2] = 1.0
        self._H_shape = np.eye(3)

    def initialize_state(self, det: DetectedObject3D):
        pos = det.get_position()
        yaw = det.get_yaw()
        ps = np.array([pos[0], pos[1], yaw, 0.0, 0.0])
        pc = np.eye(5) * 0.5
        ss = np.array(list(det.get_dimensions()), dtype=float)
        sc = np.eye(3) * 0.5
        return (ps, pc), (ss, sc)

    def predict(self, state, dt: float):
        (ps, pc), (ss, sc) = state
        return self._pred_pos(ps, pc, dt), self._pred_shape(ss, sc)

    def update(self, state, det: DetectedObject3D):
        (pp, cp), (sp, cs) = state
        return self._upd_pos(pp, cp, det), self._upd_shape(sp, cs, det)

    def to_kinematic_state(self, state) -> KinematicState:
        (ps, pc), _ = state
        x, y, yaw, v, yr = ps
        xy_cov = tuple(pc[:2, :2].flatten().tolist())
        return KinematicState(
            position=(x, y, 0.0),
            orientation=_yaw_to_quat(yaw),
            velocity=(v * math.cos(yaw), v * math.sin(yaw), 0.0),
            angular_velocity=(0.0, 0.0, yr),
            position_covariance=xy_cov,
        )

    def _pred_pos(self, ps, pc, dt):
        x, y, yaw, v, yr = ps
        if abs(yr) < 1e-5:
            xn = x + v * math.cos(yaw) * dt
            yn = y + v * math.sin(yaw) * dt
            yaw_n = yaw
        else:
            xn = x + (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            yn = y + (v / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
            yaw_n = yaw + yr * dt
        pn = np.array([xn, yn, yaw_n, v, yr])
        F = self._jacobian(ps, dt)
        return pn, F @ pc @ F.T + self._Q_pos

    def _pred_shape(self, ss, sc):
        return ss.copy(), sc + self._Q_shape

    def _upd_pos(self, pp, cp, det):
        z = np.array([det.get_position()[0], det.get_position()[1], det.get_yaw()])
        H = self._H_pos
        inn = z - H @ pp
        S = H @ cp @ H.T + self._R_pos
        K = cp @ H.T @ np.linalg.inv(S)
        return pp + K @ inn, (np.eye(5) - K @ H) @ cp

    def _upd_shape(self, sp, sc, det):
        z = np.array(list(det.get_dimensions()), dtype=float)
        H = self._H_shape
        inn = z - H @ sp
        S = H @ sc @ H.T + self._R_shape
        K = sc @ H.T @ np.linalg.inv(S)
        return sp + K @ inn, (np.eye(3) - K @ H) @ sc

    def _jacobian(self, ps, dt):
        _, _, yaw, v, yr = ps
        F = np.eye(5)
        if abs(yr) < 1e-5:
            F[0, 2] = -v * math.sin(yaw) * dt
            F[0, 3] = math.cos(yaw) * dt
            F[1, 2] = v * math.cos(yaw) * dt
            F[1, 3] = math.sin(yaw) * dt
        else:
            F[0, 2] = (v / yr) * (math.cos(yaw + yr * dt) - math.cos(yaw))
            F[0, 3] = (1.0 / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            F[0, 4] = ((v / yr**2) * (math.sin(yaw) - math.sin(yaw + yr * dt))
                       + (v / yr) * math.cos(yaw + yr * dt) * dt)
            F[1, 2] = (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
            F[1, 3] = (1.0 / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
            F[1, 4] = ((v / yr**2) * (-math.cos(yaw) + math.cos(yaw + yr * dt))
                       + (v / yr) * math.sin(yaw + yr * dt) * dt)
            F[2, 4] = dt
        return F


# ===========================================================================
# Tracklet
# ===========================================================================

class Tracklet:
    def __init__(self, filter_instance, initial_detection, track_id=None):
        self.track_id = track_id if track_id is not None else str(uuid.uuid4())
        self._filter = filter_instance
        self._filter_state = self._filter.initialize_state(initial_detection)
        self._kin = self._filter.to_kinematic_state(self._filter_state)
        self._geom = GeometricInfo(dimensions=initial_detection.get_dimensions())
        self._info = ObjectInfo(
            object_id=initial_detection.get_object_id(),
            class_name=initial_detection.get_class_name(),
            is_stationary=initial_detection.is_stationary(),
            existence_probability=initial_detection.get_existence_probability(),
        )
        self.age = 1
        self.missed_count = 0
        self.is_confirmed = False
        self.is_lost = False

    def predict(self, dt: float):
        self._filter_state = self._filter.predict(self._filter_state, dt)
        self._kin = self._filter.to_kinematic_state(self._filter_state)
        self.missed_count += 1

    def update(self, detection):
        self._filter_state = self._filter.update(self._filter_state, detection)
        self._kin = self._filter.to_kinematic_state(self._filter_state)
        self._geom = GeometricInfo(dimensions=detection.get_dimensions())
        if detection.get_class_name():
            self._info.class_name = detection.get_class_name()
        self.age += 1
        self.missed_count = 0

    def get_current_kinematic(self): return self._kin
    def get_geometric_info(self): return self._geom
    def get_object_info(self): return self._info
    def get_track_id(self): return self.track_id
    def mark_confirmed(self): self.is_confirmed = True
    def mark_lost(self): self.is_lost = True


# ===========================================================================
# Data Association (Hungarian)
# ===========================================================================

def _euclidean(tracklet, detection) -> float:
    tx, ty, _ = tracklet.get_current_kinematic().position
    dx, dy, _ = detection.get_position()
    return math.sqrt((tx - dx) ** 2 + (ty - dy) ** 2)


class HungarianAssociation:
    def __init__(self, cost_func, max_distance: float = 10.0):
        self._cost_func = cost_func
        self._max_dist = max_distance

    def associate(self, tracklets, detections):
        if not tracklets or not detections:
            return [], tracklets, detections

        nT, nD = len(tracklets), len(detections)
        cost = np.zeros((nT, nD), dtype=np.float32)
        for i, t in enumerate(tracklets):
            for j, d in enumerate(detections):
                c = self._cost_func(t, d)
                cost[i, j] = c if c <= self._max_dist else 1e6

        row_idx, col_idx = linear_sum_assignment(cost)

        matched, used_t, used_d = [], set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] >= 1e5:
                continue
            matched.append((tracklets[r], detections[c]))
            used_t.add(r)
            used_d.add(c)

        unmatched_t = [tracklets[i] for i in range(nT) if i not in used_t]
        unmatched_d = [detections[j] for j in range(nD) if j not in used_d]
        return matched, unmatched_t, unmatched_d


# ===========================================================================
# Multi-Object Tracker
# ===========================================================================

@dataclass
class TrackingConfig:
    min_hits_to_confirm: int = 3
    max_misses_to_lose: int = 3


class MultiObjectTracker:
    def __init__(self, filter_factory, association, config=None):
        self._factory = filter_factory
        self._assoc = association
        self._cfg = config if config is not None else TrackingConfig()
        self._tracklets: List[Tracklet] = []

    def update(self, detections, dt: float):
        for t in self._tracklets:
            t.predict(dt)

        matched, unmatched_t, unmatched_d = self._assoc.associate(
            self._tracklets, detections
        )

        for t, d in matched:
            t.update(d)

        for d in unmatched_d:
            self._tracklets.append(Tracklet(self._factory(), d))

        for t in self._tracklets:
            if t.missed_count >= self._cfg.max_misses_to_lose:
                t.mark_lost()
            if not t.is_confirmed and t.age >= self._cfg.min_hits_to_confirm:
                t.mark_confirmed()

        self._tracklets = [t for t in self._tracklets if not t.is_lost]
        return list(self._tracklets)

    def reset(self):
        self._tracklets.clear()


# ===========================================================================
# MOT Evaluator
# ===========================================================================

@dataclass
class FrameResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    total_match_distance: float = 0.0


class MOTEvaluator:
    def __init__(self, match_threshold: float = 3.0):
        self._thresh = match_threshold
        self._results: List[FrameResult] = []
        self._prev: Dict[str, str] = {}

    def reset(self):
        self._results.clear()
        self._prev.clear()

    def update(self, ground_truth, tracks) -> FrameResult:
        r = FrameResult()
        if not ground_truth and not tracks:
            self._results.append(r)
            return r
        if not ground_truth:
            r.fp = len(tracks)
            self._results.append(r)
            return r
        if not tracks:
            r.fn = len(ground_truth)
            self._results.append(r)
            return r

        cost = np.full((len(ground_truth), len(tracks)), 1e9)
        for gi, gt in enumerate(ground_truth):
            gx, gy, _ = gt.get_position()
            for ti, tr in enumerate(tracks):
                tx, ty, _ = tr.get_current_kinematic().position
                cost[gi, ti] = math.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_g, matched_t = set(), set()
        current: Dict[str, str] = {}

        for gi, ti in zip(row_idx, col_idx):
            if cost[gi, ti] > self._thresh:
                continue
            gt = ground_truth[gi]
            tr = tracks[ti]
            gid = gt.get_object_id()
            tid = tr.get_track_id()
            prev = self._prev.get(gid)
            if prev is not None and prev != tid:
                r.id_switches += 1
            current[gid] = tid
            matched_g.add(gi)
            matched_t.add(ti)
            r.tp += 1
            r.total_match_distance += cost[gi, ti]

        r.fn = len(ground_truth) - len(matched_g)
        r.fp = len(tracks) - len(matched_t)
        self._prev = current
        self._results.append(r)
        return r

    def compute(self):
        tp = sum(r.tp for r in self._results)
        fp = sum(r.fp for r in self._results)
        fn = sum(r.fn for r in self._results)
        idsw = sum(r.id_switches for r in self._results)
        gt_total = tp + fn
        dist = sum(r.total_match_distance for r in self._results)

        mota = 1.0 if gt_total == 0 else 1.0 - (fp + fn + idsw) / gt_total
        motp = dist / tp if tp > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / gt_total if gt_total > 0 else 0.0

        return dict(mota=mota, motp=motp, id_switches=idsw,
                    tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                    n_frames=len(self._results), total_gt=gt_total)


# ===========================================================================
# Synthetic Scenario
# ===========================================================================

@dataclass
class ObjectSpec:
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
    def __init__(self, n_frames, dt, objects, seed=42):
        self.n_frames = n_frames
        self.dt = dt
        self._specs = objects
        self._rng = random.Random(seed)
        self._traj = {s.object_id: self._compute(s) for s in objects}

    def get_detections(self, f):
        result = []
        for spec in self._specs:
            pose = self._traj[spec.object_id][f]
            if pose is None:
                continue
            if self._rng.random() > spec.detection_rate:
                continue
            x, y, yaw = pose
            result.append({
                "x": x + self._rng.gauss(0, spec.pos_noise_std),
                "y": y + self._rng.gauss(0, spec.pos_noise_std),
                "yaw": yaw + self._rng.gauss(0, spec.yaw_noise_std),
                "length": spec.length, "width": spec.width, "height": spec.height,
                "class_name": spec.class_name,
                "timestamp": f * self.dt, "frame_id": "map",
            })
        return result

    def get_ground_truth(self, f):
        result = []
        for spec in self._specs:
            pose = self._traj[spec.object_id][f]
            if pose is None:
                continue
            x, y, yaw = pose
            result.append({
                "object_id": spec.object_id,
                "x": x, "y": y, "yaw": yaw,
                "length": spec.length, "width": spec.width, "height": spec.height,
                "class_name": spec.class_name,
                "timestamp": f * self.dt, "frame_id": "map",
            })
        return result

    def iter_frames(self):
        for i in range(self.n_frames):
            yield i, self.get_detections(i), self.get_ground_truth(i)

    def _compute(self, spec):
        end = spec.end_frame if spec.end_frame is not None else self.n_frames - 1
        traj = [None] * self.n_frames
        x, y, yaw = spec.x0, spec.y0, spec.yaw0
        for f in range(self.n_frames):
            if spec.start_frame <= f <= end:
                traj[f] = (x, y, yaw)
            yr, v, dt = spec.yaw_rate, spec.v, self.dt
            if abs(yr) < 1e-5:
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
            else:
                x += (v / yr) * (math.sin(yaw + yr * dt) - math.sin(yaw))
                y += (v / yr) * (-math.cos(yaw + yr * dt) + math.cos(yaw))
                yaw += yr * dt
        return traj


def _make_highway(n_frames, dt, pos_noise, det_rate, seed):
    return SyntheticScenario(n_frames=n_frames, dt=dt, seed=seed, objects=[
        ObjectSpec("car_0", x0=0.0,   y0=0.0, yaw0=0.0, v=15.0,
                   pos_noise_std=pos_noise, detection_rate=det_rate),
        ObjectSpec("car_1", x0=-30.0, y0=3.5, yaw0=0.0, v=20.0,
                   pos_noise_std=pos_noise, detection_rate=det_rate),
        ObjectSpec("car_2", x0=20.0,  y0=7.0, yaw0=-0.1, v=18.0,
                   start_frame=max(1, n_frames // 3),
                   end_frame=min(n_frames - 1, (n_frames * 4) // 5),
                   pos_noise_std=pos_noise * 1.3, detection_rate=det_rate),
    ])


def _make_intersection(n_frames, dt, pos_noise, det_rate, seed):
    return SyntheticScenario(n_frames=n_frames, dt=dt, seed=seed, objects=[
        ObjectSpec("car_0", x0=-40.0, y0=0.0,   yaw0=0.0,         v=10.0,
                   pos_noise_std=pos_noise, detection_rate=det_rate),
        ObjectSpec("car_1", x0=0.0,   y0=-30.0, yaw0=math.pi/2,   v=8.0,
                   yaw_rate=-0.15, end_frame=min(n_frames-1, int(n_frames*0.75)),
                   pos_noise_std=pos_noise*1.3, detection_rate=det_rate),
        ObjectSpec("ped_0", class_name="Pedestrian",
                   x0=5.0, y0=-10.0, yaw0=math.pi/2, v=1.5,
                   length=0.5, width=0.5, height=1.7,
                   pos_noise_std=pos_noise*0.5, detection_rate=det_rate),
    ])


def _make_occlusion(n_frames, dt, pos_noise, det_rate, seed):
    det_rate_low = max(0.3, det_rate * 0.6)
    return SyntheticScenario(n_frames=n_frames, dt=dt, seed=seed, objects=[
        ObjectSpec("car_0", x0=0.0,  y0=0.0, yaw0=0.0, v=10.0,
                   detection_rate=det_rate_low,
                   pos_noise_std=pos_noise * 1.5),
        ObjectSpec("car_1", x0=10.0, y0=5.0, yaw0=0.0, v=12.0,
                   detection_rate=max(0.3, det_rate * 0.5),
                   pos_noise_std=pos_noise * 1.5),
    ])


# ===========================================================================
# Main demo entry point
# ===========================================================================

def run_demo(params_json: str) -> str:
    """
    Run tracking simulation with the given parameters.

    Args:
        params_json: JSON string with fields:
            scenario          : "highway" | "intersection" | "occlusion"
            n_frames          : int
            min_hits_to_confirm : int
            max_misses_to_lose  : int
            max_distance      : float  (association gate distance [m])
            pos_noise_std     : float
            detection_rate    : float  [0, 1]
            seed              : int

    Returns:
        JSON string with:
            bounds   : {x_min, x_max, y_min, y_max}
            frames   : list of {frame_idx, gt: [...], tracks: [...]}
            metrics  : {mota, motp, id_switches, precision, recall,
                        tp, fp, fn, n_frames, total_gt}
    """
    params = json.loads(params_json)

    scenario_name = params.get("scenario", "highway")
    n_frames = int(params.get("n_frames", 60))
    min_hits = int(params.get("min_hits_to_confirm", 3))
    max_misses = int(params.get("max_misses_to_lose", 5))
    max_dist = float(params.get("max_distance", 10.0))
    pos_noise = float(params.get("pos_noise_std", 0.3))
    det_rate = float(params.get("detection_rate", 1.0))
    seed = int(params.get("seed", 42))
    dt = 0.1

    factories = {
        "highway": _make_highway,
        "intersection": _make_intersection,
        "occlusion": _make_occlusion,
    }
    factory = factories.get(scenario_name, _make_highway)
    scenario = factory(n_frames, dt, pos_noise, det_rate, seed)

    tracker = MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=HungarianAssociation(_euclidean, max_distance=max_dist),
        config=TrackingConfig(
            min_hits_to_confirm=min_hits,
            max_misses_to_lose=max_misses,
        ),
    )
    evaluator = MOTEvaluator(match_threshold=max(2.0, max_dist * 0.5))
    parser = DictParser()

    frames_out = []
    all_gx, all_gy = [], []

    for frame_idx, dets, gt_dicts in scenario.iter_frames():
        gt_dets = parser.parse(gt_dicts) if gt_dicts else []
        tracks = tracker.update(parser.parse(dets) if dets else [], dt=dt)
        evaluator.update(gt_dets, tracks)

        gt_list = []
        for g in gt_dets:
            x, y, _ = g.get_position()
            all_gx.append(x)
            all_gy.append(y)
            gt_list.append({
                "id": g.get_object_id(),
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(g.get_yaw(), 4),
                "l": round(g.get_dimensions()[0], 2),
                "w": round(g.get_dimensions()[1], 2),
            })

        track_list = []
        for t in tracks:
            kin = t.get_current_kinematic()
            x, y, _ = kin.position
            track_list.append({
                "id": t.get_track_id()[:8],
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(kin.get_yaw(), 4),
                "l": round(t.get_geometric_info().dimensions[0], 2),
                "w": round(t.get_geometric_info().dimensions[1], 2),
                "confirmed": t.is_confirmed,
                "age": t.age,
            })

        frames_out.append({
            "frame_idx": frame_idx,
            "gt": gt_list,
            "tracks": track_list,
        })

    metrics = evaluator.compute()

    pad = 15.0
    bounds = {
        "x_min": (min(all_gx) - pad) if all_gx else -50.0,
        "x_max": (max(all_gx) + pad) if all_gx else 50.0,
        "y_min": (min(all_gy) - pad) if all_gy else -20.0,
        "y_max": (max(all_gy) + pad) if all_gy else 20.0,
    }

    return json.dumps({
        "bounds": bounds,
        "frames": frames_out,
        "metrics": {
            "mota":       round(metrics["mota"], 4),
            "motp":       round(metrics["motp"], 4),
            "id_switches": metrics["id_switches"],
            "precision":  round(metrics["precision"], 4),
            "recall":     round(metrics["recall"], 4),
            "tp":         metrics["tp"],
            "fp":         metrics["fp"],
            "fn":         metrics["fn"],
            "n_frames":   metrics["n_frames"],
            "total_gt":   metrics["total_gt"],
        },
    })


# ===========================================================================
# KITTI file support
# ===========================================================================

_KITTI_VALID_TYPES = {
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
    "Cyclist", "Tram", "Misc",
}


def _parse_kitti_text(content: str) -> Dict[int, List[dict]]:
    """
    Parse KITTI tracking label file content (string) into a dict of
    {frame_idx: [box_dict, ...]}.

    Each box_dict has keys: track_id, class_name, x, y, yaw, length, width, height
    (already converted to BEV coordinates).
    """
    frames: Dict[int, List[dict]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 17:
            continue
        try:
            frame    = int(tokens[0])
            track_id = int(tokens[1])
            type_    = tokens[2]
            if type_ not in _KITTI_VALID_TYPES:
                continue
            h     = float(tokens[10])
            w     = float(tokens[11])
            l     = float(tokens[12])
            x_cam = float(tokens[13])
            y_cam = float(tokens[14])
            z_cam = float(tokens[15])
            ry    = float(tokens[16])
        except (ValueError, IndexError):
            continue

        # Camera → BEV coordinate conversion (same as KittiTrackingParser)
        frames.setdefault(frame, []).append({
            "track_id":   str(track_id),
            "class_name": type_,
            "x":          z_cam,
            "y":          -x_cam,
            "yaw":        -ry,
            "length":     l,
            "width":      w,
            "height":     h,
        })
    return frames


def _build_metrics_dict(metrics: dict) -> dict:
    return {
        "mota":       round(metrics["mota"], 4),
        "motp":       round(metrics["motp"], 4),
        "id_switches": metrics["id_switches"],
        "precision":  round(metrics["precision"], 4),
        "recall":     round(metrics["recall"], 4),
        "tp":         metrics["tp"],
        "fp":         metrics["fp"],
        "fn":         metrics["fn"],
        "n_frames":   metrics["n_frames"],
        "total_gt":   metrics["total_gt"],
    }


def run_demo_kitti(kitti_content: str, params_json: str) -> str:
    """
    Run tracker on a KITTI tracking label file.

    Args:
        kitti_content : raw text content of the label file (e.g. label_02/0000.txt)
        params_json   : same parameter JSON as run_demo()

    Returns:
        Same JSON format as run_demo() — compatible with the JS visualizer.

    Notes:
        - KITTI labels are used as ground truth (GT).
        - Noisy detections are synthesised from GT so that the noise/detection-rate
          sliders still work meaningfully.
        - Frame indices follow the file (0-indexed, may be sparse).
    """
    params = json.loads(params_json)
    min_hits  = int(params.get("min_hits_to_confirm", 3))
    max_misses = int(params.get("max_misses_to_lose", 5))
    max_dist  = float(params.get("max_distance", 10.0))
    pos_noise = float(params.get("pos_noise_std", 0.3))
    det_rate  = float(params.get("detection_rate", 1.0))
    seed      = int(params.get("seed", 42))
    dt        = 0.1

    kitti_frames = _parse_kitti_text(kitti_content)
    if not kitti_frames:
        return json.dumps({"error": "ファイルに有効な KITTI データが見つかりませんでした。",
                           "frames": [], "metrics": {}})

    n_frames = max(kitti_frames.keys()) + 1
    rng = random.Random(seed)
    parser = DictParser()
    tracker = MultiObjectTracker(
        filter_factory=VehicleTracker,
        association=HungarianAssociation(_euclidean, max_distance=max_dist),
        config=TrackingConfig(min_hits_to_confirm=min_hits, max_misses_to_lose=max_misses),
    )
    evaluator = MOTEvaluator(match_threshold=max(2.0, max_dist * 0.5))

    frames_out = []
    all_gx, all_gy = [], []

    for f in range(n_frames):
        boxes = kitti_frames.get(f, [])

        # GT dicts (exact KITTI labels, with object_id = track_id for IDSW tracking)
        gt_dicts = [{
            "object_id": b["track_id"],
            "x": b["x"], "y": b["y"], "yaw": b["yaw"],
            "length": b["length"], "width": b["width"], "height": b["height"],
            "class_name": b["class_name"],
            "timestamp": f * dt, "frame_id": "map",
        } for b in boxes]

        # Noisy detections synthesised from GT (simulate a detector)
        det_dicts = []
        for b in boxes:
            if rng.random() > det_rate:
                continue
            det_dicts.append({
                "x":   b["x"] + rng.gauss(0, pos_noise),
                "y":   b["y"] + rng.gauss(0, pos_noise),
                "yaw": b["yaw"] + rng.gauss(0, 0.05),
                "length": b["length"], "width": b["width"], "height": b["height"],
                "class_name": b["class_name"],
                "timestamp": f * dt, "frame_id": "map",
            })

        gt_dets = parser.parse(gt_dicts) if gt_dicts else []
        tracks  = tracker.update(parser.parse(det_dicts) if det_dicts else [], dt=dt)
        evaluator.update(gt_dets, tracks)

        gt_list = []
        for g in gt_dets:
            x, y, _ = g.get_position()
            all_gx.append(x)
            all_gy.append(y)
            gt_list.append({
                "id": g.get_object_id(),
                "x": round(x, 3), "y": round(y, 3),
                "yaw": round(g.get_yaw(), 4),
                "l": round(g.get_dimensions()[0], 2),
                "w": round(g.get_dimensions()[1], 2),
            })

        track_list = []
        for t in tracks:
            kin = t.get_current_kinematic()
            x, y, _ = kin.position
            track_list.append({
                "id": t.get_track_id()[:8],
                "x": round(x, 3), "y": round(y, 3),
                "yaw": round(kin.get_yaw(), 4),
                "l": round(t.get_geometric_info().dimensions[0], 2),
                "w": round(t.get_geometric_info().dimensions[1], 2),
                "confirmed": t.is_confirmed,
                "age": t.age,
            })

        frames_out.append({"frame_idx": f, "gt": gt_list, "tracks": track_list})

    metrics = evaluator.compute()
    pad = 15.0
    bounds = {
        "x_min": (min(all_gx) - pad) if all_gx else -50.0,
        "x_max": (max(all_gx) + pad) if all_gx else 50.0,
        "y_min": (min(all_gy) - pad) if all_gy else -20.0,
        "y_max": (max(all_gy) + pad) if all_gy else 20.0,
    }
    return json.dumps({"bounds": bounds, "frames": frames_out,
                       "metrics": _build_metrics_dict(metrics)})
