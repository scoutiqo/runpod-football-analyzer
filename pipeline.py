# -*- coding: utf-8 -*-

# ---- SAFE HOMOGRAPHY WRAPPER (inserted) ----
try:
    from calibrate import image_to_field as _orig_image_to_field
except Exception:
    _orig_image_to_field = None

def safe_image_to_field(H, x, y):
    """
    Try homography; if missing/bad, fall back to raw pixel (x, y).
    Keeps the pipeline and live UIs running even when calibration fails.
    """
    if _orig_image_to_field and H is not None:
        try:
            return _orig_image_to_field(H, x, y)
        except Exception:
            pass
    # fallback: pixel coords
    try:
        return float(x), float(y)
    except Exception:
        return x, y
# ---- END SAFE WRAPPER ----

import cv2
import mediapipe as mp
import numpy as np

from detectors_local import MultiDetector
from deepsort_wrapper import PlayerTrackerDS
from ball_tracker import BallTracker
from calibrate import estimate_homography
from team_assign import TeamAssigner
from smooth import smooth_and_speed
from events import infer_possession, detect_passes





class StableIdAssigner:
    """
    Remap volatile tracker ids to small, persistent ids 1..max_slots.
    Keeps mapping alive for ttl frames after an object disappears.
    """
    def __init__(self, max_slots=30, ttl=300):
        self.max_slots = max_slots
        self.ttl = ttl
        self.map = {}          # tracker_id -> (stable_id, last_frame_seen)
        self.rev = {}          # stable_id -> tracker_id
        self.next_free = 1
        self.frame_idx = 0

    def _alloc(self):
        # reuse gaps if any, else increment
        for sid in range(1, self.max_slots + 1):
            if sid not in self.rev:
                return sid
        return self.max_slots  # clamp if overflow (shouldn't happen often)

    def tick(self):
        self.frame_idx += 1
        # expire stale links
        to_drop = []
        for tid, (sid, lastf) in self.map.items():
            if self.frame_idx - lastf > self.ttl:
                to_drop.append(tid)
        for tid in to_drop:
            sid = self.map[tid][0]
            self.map.pop(tid, None)
            self.rev.pop(sid, None)

    def assign_many(self, players):
        """
        players: list of dicts with volatile 'id' (tracker id)
        Returns list with extra field 'sid' (stable id)
        """
        self.tick()
        out = []
        for p in players:
            tid = p.get("id")
            if tid is None:
                out.append(p); continue
            if tid in self.map:
                sid = self.map[tid][0]
                self.map[tid] = (sid, self.frame_idx)
            else:
                sid = self._alloc()
                # if sid already taken, evict old mapping
                if sid in self.rev:
                    old_tid = self.rev[sid]
                    self.map.pop(old_tid, None)
                self.map[tid] = (sid, self.frame_idx)
                self.rev[sid] = tid
            p = dict(p)
            p["sid"] = sid
            out.append(p)
        return out
class StableIdAssigner:
    """
    Remap volatile tracker ids to small, persistent ids 1..max_slots.
    Keeps mapping alive for ttl frames after an object disappears.
    """
    def __init__(self, max_slots=30, ttl=300):
        self.max_slots = max_slots
        self.ttl = ttl
        self.map = {}          # tracker_id -> (stable_id, last_frame_seen)
        self.rev = {}          # stable_id -> tracker_id
        self.next_free = 1
        self.frame_idx = 0

    def _alloc(self):
        # reuse gaps if any, else increment
        for sid in range(1, self.max_slots + 1):
            if sid not in self.rev:
                return sid
        return self.max_slots  # clamp if overflow (shouldn't happen often)

    def tick(self):
        self.frame_idx += 1
        # expire stale links
        to_drop = []
        for tid, (sid, lastf) in self.map.items():
            if self.frame_idx - lastf > self.ttl:
                to_drop.append(tid)
        for tid in to_drop:
            sid = self.map[tid][0]
            self.map.pop(tid, None)
            self.rev.pop(sid, None)

    def assign_many(self, players):
        """
        players: list of dicts with volatile 'id' (tracker id)
        Returns list with extra field 'sid' (stable id)
        """
        self.tick()
        out = []
        for p in players:
            tid = p.get("id")
            if tid is None:
                out.append(p); continue
            if tid in self.map:
                sid = self.map[tid][0]
                self.map[tid] = (sid, self.frame_idx)
            else:
                sid = self._alloc()
                # if sid already taken, evict old mapping
                if sid in self.rev:
                    old_tid = self.rev[sid]
                    self.map.pop(old_tid, None)
                self.map[tid] = (sid, self.frame_idx)
                self.rev[sid] = tid
            p = dict(p)
            p["sid"] = sid
            out.append(p)
        return out
from ball_kf import BallKF
# ---------- Ball motion helpers (KF + optical flow + speed gate) ----------
class BallKF:
    def __init__(self):
        import numpy as _np
        self.x = _np.zeros((4,1), dtype=_np.float32)  # [x,y,vx,vy]
        self.P = _np.eye(4, dtype=_np.float32)*100.0
        self.Q = _np.diag([5,5,50,50]).astype(_np.float32)
        self.R = _np.diag([30,30]).astype(_np.float32)
        self.F = _np.eye(4, dtype=_np.float32)
        self.H = _np.zeros((2,4), dtype=_np.float32); self.H[0,0]=1; self.H[1,1]=1
        self.t_prev = None

    def predict(self, t):
        import numpy as _np
        if self.t_prev is None:
            self.t_prev = t
            return self.x[:2].ravel()
        dt = max(1e-3, float(t - self.t_prev))
        self.F[0,2] = dt; self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.t_prev = t
        return self.x[:2].ravel()

    def update(self, z_xy):
        import numpy as _np
        z = _np.array(z_xy, dtype=_np.float32).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ _np.linalg.inv(S)
        self.x = self.x + K @ y
        I = _np.eye(4, dtype=_np.float32)
        self.P = (I - K @ self.H) @ self.P

_ball_kf = BallKF()
_prev_frame_gray = None

def _lk_flow_predict(frame_bgr, last_xy):
    """Single-point Lucas–Kanade optical flow; returns next xy or None."""
    import cv2, numpy as _np
    global _prev_frame_gray
    if last_xy is None: return None
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if _prev_frame_gray is None:
        _prev_frame_gray = g
        return None
    p0 = _np.array([[last_xy]], dtype=_np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(_prev_frame_gray, g, p0, None,
                                           winSize=(15,15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    _prev_frame_gray = g
    if st is None or st.ravel()[0] == 0:
        return None
    return tuple(map(float, p1[0,0]))

def _cap_speed(prev_xy, xy, dt, W, H, factor=0.8):
    """Clamp jump based on image scale per second."""
    import numpy as _np
    if prev_xy is None: return xy
    max_per_s = factor * float(min(W, H))
    max_step = max_per_s * max(1e-3, dt)
    dx, dy = xy[0]-prev_xy[0], xy[1]-prev_xy[1]
    d = (dx*dx + dy*dy)**0.5
    if d <= max_step: return xy
    s = max_step/d
    return (prev_xy[0]+dx*s, prev_xy[1]+dy*s)
# --------------------------------------------------------------------------
# Learn teams only during early frames, then freeze
WARMUP_FRAMES = 200
# -------- Ball stabilizer helpers (inserted) --------
_ball_state = {
    "xy": None,         # last accepted (x,y) in px
    "t":  None,         # last time in seconds
    "miss": 0           # consecutive frames without a confident update
}

def _nearest_player_dist(px, py, players):
    best = 1e9
    for p in (players or []):
        cx = (p["x1"] + p["x2"]) / 2.0
        cy = (p["y1"] + p["y2"]) / 2.0
        d = ((px-cx)**2 + (py-cy)**2) ** 0.5
        if d < best: best = d
    return best

def stabilize_ball(cx, cy, t_now, W, H, alpha=0.28, max_speed_mph=45.0):
    """
    Limit ball displacement by an upper bound on speed, then EMA smooth.
    max_speed_mph ~45 mph (~20 m/s). Convert to pixels via pitch scale guess.
    We don't know exact meters/pixel, so approximate by image size:
      assume ball shouldn't move faster than ~0.7 * min(W,H) per second.
    """
    global _ball_state
    max_px_per_s = 0.9 * float(min(W, H))   # conservative cap
    if _ball_state["xy"] is None:
        _ball_state["xy"] = (cx, cy)
        _ball_state["t"] = t_now
        _ball_state["miss"] = 0
        return cx, cy

    lx, ly = _ball_state["xy"]
    dt = max(1e-3, float(t_now - _ball_state["t"]))
    # speed gate
    max_step = max_px_per_s * dt
    dx, dy = cx - lx, cy - ly
    dist = (dx*dx + dy*dy) ** 0.5
    if dist > max_step * 1.4:
        # clamp to allowed step toward the new point
        scale = (max_step / dist)
        cx = lx + dx * scale
        cy = ly + dy * scale

    # EMA smoothing
    sx = alpha * cx + (1 - alpha) * lx
    sy = alpha * cy + (1 - alpha) * ly

    _ball_state["xy"] = (sx, sy)
    _ball_state["t"] = t_now
    _ball_state["miss"] = 0
    return sx, sy
# -------- End helpers --------
# Toggle: disable pose to avoid CPU stalls
USE_POSE = False
POSE = None
if USE_POSE:
    POSE = mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    )

# ---------- Pitch mask (keep only grass area) ----------
def compute_pitch_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.medianBlur(m, 7)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    big = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [big], -1, 255, thickness=cv2.FILLED)
# --- extra cleanup to avoid stands/ghosts ---
mask = cv2.erode(mask, np.ones((17,17), np.uint8), iterations=1)
mask[: int(mask.shape[0] * 0.10), :] = 0
return mask

def inside_pitch(x: float, y: float, mask: np.ndarray) -> bool:
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1]:
        return False
    return mask[yi, xi] > 0

# ---------- Ball fallback scoring ----------
def _ball_color_score(patch_bgr: np.ndarray) -> float:
    """Score 0..1 for white/orange-like ball colors."""
    if patch_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = ((S < 60) & (V > 170)).mean()
    orange = (((H < 20) | (H > 160)) & (S > 90) & (V > 100)).mean()
    return float(max(white, orange))

def _aspect_roundness(w: float, h: float) -> float:
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(w, h) / max(w, h))

def _player_feet_points(players):
    pts = []
    for p in (players or []):
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        bx = (x1 + x2) * 0.5
        by = y2  # bottom of bbox
        pts.append((bx, by))
    return pts
def pick_ball_candidate(\1, players=None):
    """
    candidates: list of (x1,y1,x2,y2, conf, cls)
    last_xy: (x,y) or None
    players: list of tracker dicts (for proximity scoring) or None
    Return best [x1,y1,x2,y2] or None
    """
    if not candidates:
        return None

    min_side = 0.010 * min(W, H)
    max_side = 0.060 * min(W, H)

    best_score, best_box = -1e9, None
    for (x1, y1, x2, y2, conf, cls_) in candidates:
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        side_ok = (min_side <= max(w, h) <= max_side)

        x1i, y1i = int(max(0, min(x1, frame.shape[1] - 1))), int(max(0, min(y1, frame.shape[0] - 1)))
        x2i, y2i = int(max(0, min(x2, frame.shape[1] - 1))), int(max(0, min(y2, frame.shape[0] - 1)))
        if x2i <= x1i or y2i <= y1i:
            continue
        patch = frame[y1i:y2i, x1i:x2i]

        color = _ball_color_score(patch)
        roundness = _aspect_roundness(w, h)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = 0.0 if last_xy is None else (np.hypot(cx - last_xy[0], cy - last_xy[1]) / float(min(W, H)))
        feet = _player_feet_points(players) if players is not None else []
        feet_d = min([np.hypot(cx-fx, cy-fy) for (fx,fy) in feet] + [1e9])
        foot_prox = 1.0 / (1.0 + feet_d / max(1.0, 0.12*min(W, H)))

        # Proximity to nearest player (closer => higher score); normalized to ~0..1
        if players:
            npx = _nearest_player_dist(cx, cy, players)
            prox = 1.0 / (1.0 + npx / max(1.0, 0.15 * min(W, H)))
        else:
            prox = 0.0

        # Weighted score: color + roundness + plausible size + coco hint + conf - distance + proximity
        score = (
            2.5 * color
            + 1.5 * roundness
            + (0.6 if side_ok else 0.0)
            + 0.4 * float(cls_ == 32)
            + 0.4 * conf - 0.5 * dist + 1.0 * foot_prox - 0.3 * (abs(cx - pred_xy[0]) + abs(cy - pred_xy[1]))
            + 0.8 * prox
        )
        if score > best_score:
            best_score, best_box = score, [x1, y1, x2, y2]
    return best_box


def _normalize_dets(raw):
    """Normalize detector output to dict-of-arrays."""
    if isinstance(raw, dict) and "xyxy" in raw:
        return {
            "xyxy": raw.get("xyxy") or [],
            "conf": raw.get("conf") or [],
            "cls": raw.get("cls") or [],
        }
    xyxy, conf, cls = [], [], []
    if isinstance(raw, (list, tuple)):
        for d in raw:
            if not isinstance(d, dict):
                continue
            x1 = d.get("x1"); y1 = d.get("y1"); x2 = d.get("x2"); y2 = d.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            conf.append(float(d.get("conf", 0.0)))
            cval = d.get("cls", d.get("class_id", d.get("class", -1)))
            try:
                c = int(getattr(cval, "item", lambda: cval)())
            except Exception:
                c = -1
            cls.append(c)
    return {"xyxy": xyxy, "conf": conf, "cls": cls}

def run_pipeline(video_path: str, cfg: dict | None = None, max_frames=150, frame_skip=2):
    """Adds robust ball fallback so ball points are always produced."""
    cfg = cfg or {}

    # ---- state ----
    Hmat = None
    points = []
    ball_series = []
    players_by_t = {}
    pitch_mask = None
    last_ball_xy = None
    # ---- end state ----

    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
    except Exception as e:
        raise RuntimeError(f"Video open error: {str(e)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Config split
    detector_cfg = cfg.get("detector", {})
    tracking_cfg = cfg.get("tracking", {"max_age": 60, "min_hits": 5, "iou_threshold": 0.3})
    ball_cfg = cfg.get("ball", {"min_conf": 0.10, "class_id": 32})

    detector = MultiDetector(player_weights="weights/players_soccer.pt", ball_weights="weights/ball_soccer.pt")
    player_tracker = PlayerTrackerDS(tracking_cfg)
    ball_tracker = BallTracker(ball_cfg)
    
    stable_ids = StableIdAssigner(max_slots=30, ttl=300)

# ---------- Compact display ids per team (stable labels) ----------
_display_map = {"home": {}, "away": {}}
_next_num = {"home": 1, "away": 1}
_last_seen = {}

def _label_for(track_id, team):
    global _display_map, _next_num, _last_seen
    pool = "home" if team == 0 else ("away" if team == 1 else "home")
    if track_id in _display_map[pool]:
        return _display_map[pool][track_id]
    n = _next_num[pool]
    _next_num[pool] = min(11, n+1)
    tag = f"H{n:02d}" if pool=="home" else f"A{n:02d}"
    _display_map[pool][track_id] = tag
    return tag

# ---------- Compact display ids per team (stable labels) ----------
_display_map = {"home": {}, "away": {}}
_next_num = {"home": 1, "away": 1}
_last_seen = {}

def _label_for(track_id, team):
    global _display_map, _next_num, _last_seen
    pool = "home" if team == 0 else ("away" if team == 1 else "home")
    if track_id in _display_map[pool]:
        return _display_map[pool][track_id]
    n = _next_num[pool]
    _next_num[pool] = min(11, n+1)
    tag = f"H{n:02d}" if pool=="home" else f"A{n:02d}"
    _display_map[pool][track_id] = tag
    return tag
    WARMUP_FRAMES = 120
    team_frozen = False

    t = 0.0
    frames = 0
    try:
        while cap.isOpened():
            # advance by frame_skip
            for _ in range(frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break
                t += 1.0 / fps
            if not ret:
                break

            # refresh pitch mask first frame and every ~10s
            if pitch_mask is None or (frames % 300 == 0):
                pitch_mask = compute_pitch_mask(frame)

            # Detect
            try:
                det_raw = detector.infer(frame)
                dets = _normalize_dets(det_raw)
            except Exception as e:
                print(f"Detection error at t={t}: {str(e)}")
                continue

            # Split players/ball + robust fallback that ALWAYS picks a ball candidate
            player_dets = {"xyxy": [], "conf": [], "cls": []}
            ball_dets = {"xyxy": [], "conf": [], "cls": []}
            candidates = []  # (area, conf, box, cls)

            for i in range(len(dets["xyxy"])):
                x1, y1, x2, y2 = dets["xyxy"][i]
                conf = float(dets["conf"][i])
                cls_raw = dets["cls"][i]
                cls = int(cls_raw) if isinstance(cls_raw, (int, float)) else -1

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

                # Person size gate (relative to frame); drop tiny/huge people
                if cls == 0:
                    min_side = 0.02 * min(W, H)
                    max_side = 0.35 * min(W, H)
                    if not (min_side <= max(w, h) <= max_side):
                        continue
                # drop detections whose centers are off-pitch
                cx_det, cy_det = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if pitch_mask is not None and not inside_pitch(cx_det, cy_det, pitch_mask):
                    continue

                candidates.append((area, conf, [x1, y1, x2, y2], cls))

                if cls == 0:  # person
                    player_dets["xyxy"].append([x1, y1, x2, y2])
                    player_dets["conf"].append(conf)
                    player_dets["cls"].append(cls)

                if cls == 32:  # sports ball
                    ball_dets["xyxy"].append([x1, y1, x2, y2])
                    ball_dets["conf"].append(conf)
                    ball_dets["cls"].append(cls)

            # if no ball dets, pick best candidate using color/shape/size/temporal proximity
            if not ball_dets["xyxy"] and candidates:
                cand2 = []
                for (_a, c, box, cl) in candidates:
                    x1, y1, x2, y2 = box
                    cand2.append((x1, y1, x2, y2, c, cl))
                pick = pick_ball_candidate(frame, cand2, last_ball_xy, W, H, players=None)
                if pick is not None:
                    x1, y1, x2, y2 = pick
                    ball_dets["xyxy"].append([x1, y1, x2, y2])
                    ball_dets["conf"].append(0.5)  # synthetic conf for fallback
                    ball_dets["cls"].append(32)

            # Track
            players = player_tracker.update(player_dets)
            players = stable_ids.assign_many(players or [])
            ball = ball_tracker.update(ball_dets)

            # Normalize ball (list/dict/empty) and provide fallback box
            ball_box = None
            # Kalman prediction (image space)
            pred_xy = _ball_kf.predict(t)

            if isinstance(ball, list) and ball and isinstance(ball[0], dict) and all(
                k in ball[0] for k in ("x1", "y1", "x2", "y2")
            ):
                ball_box = ball[0]
            elif isinstance(ball, dict) and all(k in ball for k in ("x1", "y1", "x2", "y2")):
                ball_box = ball
            elif ball_dets["xyxy"]:
                x1, y1, x2, y2 = ball_dets["xyxy"][0]
                ball_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            else:
                # no dets at all -> try optical flow around last KF state
                flow_xy = _lk_flow_predict(frame, tuple(_ball_kf.x[:2].ravel()))
                if flow_xy is not None:
                    cx, cy = flow_xy
                else:
                    cx, cy = pred_xy
                # synthesize a tiny box around (cx,cy)
                sbox = 6
                ball_box = {"x1": cx - sbox, "y1": cy - sbox, "x2": cx + sbox, "y2": cy + sbox}

            # Calibrate (homography) occasionally
            if Hmat is None and frames % 15 == 0:
                keypoints = []
                for p in (players or []):
                    keypoints.append(((p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2))
                if ball_box:
                    keypoints.append(((ball_box["x1"] + ball_box["x2"]) / 2,
                                      (ball_box["y1"] + ball_box["y2"]) / 2))
                if len(keypoints) >= 4:
                    try:
                        Hmat = estimate_homography(keypoints)
                    except Exception:
                        Hmat = None

            # Team assignment
            if players:
                if frames < WARMUP_FRAMES:
                    tass.observe(
                        frame,
                        [
                            {"id": p.get("sid", p.get("id")), "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]}
                            for p in players
                        ],
                    )# Players -> points (with (optional) skeletal)
            cur_players = []
            for p in (players or []):
                cx = (p["x1"] + p["x2"]) / 2.0
                cy = (p["y1"] + p["y2"]) / 2.0
                rec = {
                    "t": round(t, 3),
                    "type": "player",
                    "id": p.get("sid", p.get("id")),
                    "team": tass.get_team(p["id"]),
                    "x_px": float(cx),
                    "y_px": float(cy),
                }
                xy = safe_image_to_field(Hmat, cx, cy)
                if xy:
                    rec["x_m"], rec["y_m"] = xy

                if USE_POSE and POSE is not None:
                    try:
                        patch = frame[int(p["y1"]): int(p["y2"]), int(p["x1"]): int(p["x2"])]
                        if patch.size > 0:
                            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                            results = POSE.process(rgb_patch)
                            if results.pose_landmarks:
                                kps = []
                                for lm in results.pose_landmarks.landmark:
                                    lx = p["x1"] + lm.x * (p["x2"] - p["x1"])
                                    ly = p["y1"] + lm.y * (p["y2"] - p["y1"])
                                    lz = lm.z * 1000
                                    m_xy = safe_image_to_field(Hmat, lx, ly)
                                    kps.append({
                                        "x_m": m_xy[0] if m_xy else lx,
                                        "y_m": m_xy[1] if m_xy else ly,
                                        "z_m": lz
                                    })
                                rec["keypoints"] = kps
                    except Exception as e:
                        print(f"Skeletal error for player {p.get('id')} at t={t}: {str(e)}")

                points.append(rec)
                cur_players.append(rec)

            # Ball -> points (use normalized tracker output or fallback)
            if ball_box:
                cx = (ball_box["x1"] + ball_box["x2"]) / 2.0
                cy = (ball_box["y1"] + ball_box["y2"]) / 2.0
                # stabilize ball center
                cx, cy = stabilize_ball(cx, cy, t, W, H)
# stabilize ball center\n                cx, cy = stabilize_ball(cx, cy, t, W, H)\n# correct KF and clamp to plausible step
                prev_xy = tuple(_ball_kf.x[:2].ravel())
                dt_k = max(1e-3, float(t - (_ball_kf.t_prev or t)))
                cx, cy = _cap_speed(prev_xy, (cx, cy), dt_k, W, H)
                _ball_kf.update((cx, cy))
                cx, cy = ball_kf.update(cx, cy)
                rec = {"t": round(t, 3), "type": "ball", "x_px": float(cx), "y_px": float(cy)}
                xy = safe_image_to_field(Hmat, cx, cy)
                if xy:
                    rec["x_m"], rec["y_m"] = xy
                points.append(rec)
                ball_series.append(rec)
                last_ball_xy = (cx, cy)

            # For events
            players_by_t[t] = cur_players

            frames += 1
            if frames >= max_frames:
                break
    finally:
        cap.release()

    # Smooth
    tracks = smooth_and_speed(points, have_metric=(Hmat is not None))

    # Events (basic)
    events = detect_passes(ball_series, players_by_t)

    # Possession per event: use closest player set in time
    def _players_at(time_q):
        if not players_by_t:
            return []
        key = min(players_by_t.keys(), key=lambda tt: abs(tt - time_q))
        return players_by_t.get(key, [])

    for e in events:
        e["possession_team"] = infer_possession(_players_at(e["t"]), ball_series, e["t"])

    return {
        "version": 2,
        "meta": {
            "fps": fps,
            "width": W,
            "height": H,
            "frame_skip": frame_skip,
            "pitch_m": [105, 68],
            "homography": (Hmat.tolist() if Hmat is not None else None),
        },
        "tracks": tracks,
        "events": events,
        "metrics": {},
    }

if __name__ == "__main__":
    import os
    import argparse
    import logging
    import json
    import config

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    log = logging.getLogger("pipeline")

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--frame_skip", type=int, default=3)
    ap.add_argument("--out", default="tracks.json", help="Where to save output JSON")
    args = ap.parse_args()

    log.info("Starting: video=%s max_frames=%s frame_skip=%s",
             args.video, args.max_frames, args.frame_skip)

    # Fetch config and force COCO weights so class 32 (sports ball) exists
    try:
        cfg = config.fetch_config("default")
        cfg.setdefault("detector", {})
        cfg["detector"]["weights_bucket"] = "local"
        cfg["detector"]["weights_path"] = "yolov8n.pt"
    except Exception as e:
        log.warning("fetch_config failed (%s); using local yolov8n.pt", e)
        cfg = {"detector": {"weights_bucket": "local", "weights_path": "yolov8n.pt"}}

    try:
        out = run_pipeline(args.video, cfg=cfg, max_frames=args.max_frames, frame_skip=args.frame_skip)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        log.info("Saved %s (%.1f KB)", args.out, os.path.getsize(args.out) / 1024.0)
    except Exception as e:
        log.exception("Pipeline crashed: %s", e)
        raise



























