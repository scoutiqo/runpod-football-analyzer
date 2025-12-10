# live_server.py — robust overlay with: eroded pitch mask (ball), 22 players, ref ignore,
# calibration to meters (m/s touches), shift-click ignore, save/load session.
import os, time, json, datetime as dt
from collections import deque, defaultdict
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# --- your modules ---
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from team_assign import TeamAssigner

# -------------- CONFIG --------------
VIDEO_PATH = os.environ.get("VIDEO_PATH", "test_match.mp4")
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "2"))
DRAW_TRAIL_SECONDS = 1.0
PITCH_MASK_RECOMP_EVERY = 30
MIN_PLAYER_H = 20

REID_MAX_DIST_PX = 120
LABEL_KEEP_SEC    = 3.0

STRICT_MAX_PLAYERS   = 22
STRICT_MAX_PER_TEAM  = 11

TOP_IGNORE_FRAC = 0.15
BALL_NEAR_EDGE_WEIGHT = 0.6
BALL_MOTION_WEIGHT    = 0.6
BALL_SIZE_PRIOR_W     = 0.5
BALL_CLICK_BIAS_W     = 0.8
BALL_TOUCH_RADIUS     = 28
BALL_REQUIRE_MOTION_IF_UNLOCKED = True

BALL_MIN_SPEED_TOUCH_PX = 80.0   # px/s when not calibrated
BALL_MIN_SPEED_TOUCH_MS = 3.0    # m/s when calibrated

HEAT_W, HEAT_H = 64, 36
# ------------------------------------

app = Flask(__name__)

# ---- UI/shared state ----
app.config["MODE"] = "tag"          # "tag" | "name" | "ball" | "cal"
app.config["NAME_INPUT"] = ""
app.config["BALL_LOCK"] = False
app.config["NAME_OVERRIDES"] = {}   # label(int) -> string name
app.config["SELECTED_LABEL"] = None

# Stats
stats = {"fps": 0.0, "frames": 0, "players_drawn": 0, "ball_seen": 0, "last_t": 0.0}

# Analysis
touches = defaultdict(int)          # label -> touch count
heats   = defaultdict(lambda: np.zeros((HEAT_H, HEAT_W), dtype=np.int32))
last_touch_time = defaultdict(lambda: -10.0)

# Click/shared
_last_centroids = []                 # [{"cx","cy","label","team","tid"}...]
_last_centroids_lock = Lock()
_override_ball_xy = None
_override_lock = Lock()
IGNORED_LABELS = set()

# playback
control = {"paused": False, "seek": 0, "restart": False}
control_lock = Lock()

# calibration (pixel -> meters)
H_PIX2M = None
PITCH_W_M, PITCH_H_M = 105.0, 68.0
calib_pts_px = []  # four points

def pix_to_m(x, y):
    global H_PIX2M
    if H_PIX2M is None:
        return None, None
    pt = np.array([[[x, y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H_PIX2M)[0,0]
    return float(dst[0]), float(dst[1])

# ---------- Pitch mask ----------
def compute_pitch_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.medianBlur(m, 7)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m, m
    big = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [big], -1, 255, thickness=cv2.FILLED)
    # Erode to avoid adboards/lines at edges affecting ball picks
    eroded = cv2.erode(mask, np.ones((21, 21), np.uint8))
    return mask, eroded

def inside_mask(x: float, y: float, mask: np.ndarray) -> bool:
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1]:
        return False
    return mask[yi, xi] > 0

# ---------- Ball scoring / fallback ----------
def _ball_color_score(patch_bgr: np.ndarray) -> float:
    if patch_bgr.size == 0: return 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white  = ((S < 60) & (V > 170)).mean()
    orange = (((H < 20) | (H > 160)) & (S > 90) & (V > 100)).mean()
    return float(max(white, orange))

def _aspect_roundness(w: float, h: float) -> float:
    if w <= 0 or h <= 0: return 0.0
    return float(min(w, h) / max(w, h))

def pick_ball_candidate(frame, prev_gray, pitch_dt, candidates, last_xy, W, H, pitch_mask_eroded, bias_xy=None):
    if not candidates: return None
    min_side = 0.010 * min(W, H)
    max_side = 0.085 * min(W, H)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_score, best_box = -1e9, None
    for (x1,y1,x2,y2,conf,cls_) in candidates:
        w, h = max(1.0, x2-x1), max(1.0, y2-y1)
        side_ok = (min_side <= max(w,h) <= max_side)
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

        if cy < TOP_IGNORE_FRAC * H:  # ignore scoreboard band
            continue
        if pitch_mask_eroded is not None and not inside_mask(cx, cy, pitch_mask_eroded):
            continue

        x1i, y1i = int(max(0, min(x1, frame.shape[1]-1))), int(max(0, min(y1, frame.shape[0]-1)))
        x2i, y2i = int(max(0, min(x2, frame.shape[1]-1))), int(max(0, min(y2, frame.shape[0]-1)))
        if x2i <= x1i or y2i <= y1i: continue

        patch = frame[y1i:y2i, x1i:x2i]
        color = _ball_color_score(patch)
        roundness = _aspect_roundness(w, h)

        motion = 0.0
        if prev_gray is not None:
            pg = prev_gray[y1i:y2i, x1i:x2i]
            cg = gray[y1i:y2i, x1i:x2i]
            if pg.size and cg.size and pg.shape == cg.shape:
                diff = cv2.absdiff(pg, cg)
                motion = float(np.clip(diff.mean()/20.0, 0.0, 1.0))

        if BALL_REQUIRE_MOTION_IF_UNLOCKED and bias_xy is None and motion < 0.08:
            continue

        edge_bonus = 0.0
        if pitch_dt is not None:
            yi = int(np.clip(cy, 0, pitch_dt.shape[0]-1))
            xi = int(np.clip(cx, 0, pitch_dt.shape[1]-1))
            dist = float(pitch_dt[yi, xi])
            edge_bonus = min(dist/20.0, 1.0)

        max_dim = max(w, h)
        expected = (0.011 + 0.045*(cy/float(H))) * min(W, H)
        size_prior = float(np.exp(-abs(max_dim - expected)/expected))

        dist_last = 0.0 if last_xy is None else (np.hypot(cx-last_xy[0], cy-last_xy[1]) / float(min(W, H)))

        bias_term = 0.0
        if bias_xy is not None:
            d = np.hypot(cx-bias_xy[0], cy-bias_xy[1])
            bias_term = max(0.0, 1.0 - d/200.0) * BALL_CLICK_BIAS_W

        score = (
            2.2*color + 1.3*roundness + (0.6 if side_ok else 0.0) + 0.5*float(cls_==32) + 0.4*conf
            - 0.5*dist_last + BALL_NEAR_EDGE_WEIGHT*edge_bonus + BALL_MOTION_WEIGHT*motion
            + BALL_SIZE_PRIOR_W*size_prior + bias_term
        )
        if score > best_score:
            best_score, best_box = score, [x1,y1,x2,y2]
    return best_box

# ---------- Detector output normalization ----------
def _normalize_dets(raw):
    if isinstance(raw, dict) and "xyxy" in raw:
        return {"xyxy": raw.get("xyxy") or [], "conf": raw.get("conf") or [], "cls": raw.get("cls") or []}
    xyxy, conf, cls = [], [], []
    if isinstance(raw, (list, tuple)):
        for d in raw:
            if not isinstance(d, dict): continue
            x1 = d.get("x1"); y1 = d.get("y1"); x2 = d.get("x2"); y2 = d.get("y2")
            if None in (x1,y1,x2,y2): continue
            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            conf.append(float(d.get("conf", 0.0)))
            cval = d.get("cls", d.get("class_id", d.get("class", -1)))
            try: c = int(getattr(cval, "item", lambda: cval)())
            except Exception: c = -1
            cls.append(c)
    return {"xyxy": xyxy, "conf": conf, "cls": cls}

# ---------- MJPEG generator ----------
def gen_stream():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last_frame = None

    detector       = Detector({"weights_bucket": "local", "weights_path": "yolov8n.pt"})
    player_tracker = PlayerTracker({"max_age": 30, "min_hits": 3})
    ball_tracker   = BallTracker({"min_conf": 0.10, "class_id": 32})
    tass           = TeamAssigner()

    pitch_mask = None
    pitch_mask_eroded = None
    pitch_dt   = None
    last_ball_xy = None
    last_ball_t  = None
    last_stats_t = time.time()
    frame_idx = 0
    trail = deque(maxlen=int(DRAW_TRAIL_SECONDS * fps) + 3)
    prev_gray = None

    # persistent labels
    tid_to_label = {}            # tracker id -> label
    label_state  = {}            # label -> {pos, team, tid, last_t}
    label_age    = defaultdict(int)
    next_label = 1

    while True:
        # playback control
        with control_lock:
            paused  = control["paused"]
            seek    = control["seek"]
            restart = control["restart"]
            control["seek"] = 0
            control["restart"] = False

        if restart:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_idx = 0
            prev_gray = None; trail.clear()
            player_tracker = PlayerTracker({"max_age": 30, "min_hits": 3})
            ball_tracker   = BallTracker({"min_conf": 0.10, "class_id": 32})

        if seek:
            new_idx = max(0, min((total_frames-1) if total_frames>0 else 10**9, frame_idx + seek))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx); frame_idx = new_idx
            prev_gray = None; trail.clear()
            player_tracker = PlayerTracker({"max_age": 30, "min_hits": 3})
            ball_tracker   = BallTracker({"min_conf": 0.10, "class_id": 32})

        if paused and last_frame is not None:
            frame = last_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret: break
            for _ in range(FRAME_SKIP-1):
                r2, _ = cap.read()
                if not r2: break
            last_frame = frame.copy()

        frame_idx += 1
        t_video = frame_idx * (FRAME_SKIP / fps)

        if pitch_mask is None or frame_idx % PITCH_MASK_RECOMP_EVERY == 0:
            pitch_mask, pitch_mask_eroded = compute_pitch_mask(frame)
            dt_src = (pitch_mask > 0).astype(np.uint8)
            pitch_dt = cv2.distanceTransform(dt_src, cv2.DIST_L2, 5)

        dets = _normalize_dets(detector.infer(frame))

        # Split + collect
        player_dets = {"xyxy": [], "conf": [], "cls": []}
        ball_dets   = {"xyxy": [], "conf": [], "cls": []}
        ball_cands  = []
        for i in range(len(dets["xyxy"])):
            x1,y1,x2,y2 = dets["xyxy"][i]
            conf = float(dets["conf"][i]); cls = int(dets["cls"][i]) if isinstance(dets["cls"][i], (int,float)) else -1
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            ball_cands.append((x1,y1,x2,y2,conf,cls))
            if cls == 0:
                if pitch_mask is not None and not inside_mask(cx, cy, pitch_mask): continue
                if (y2-y1) < MIN_PLAYER_H: continue
                player_dets["xyxy"].append([x1,y1,x2,y2]); player_dets["conf"].append(conf); player_dets["cls"].append(cls)
            if cls == 32:
                ball_dets["xyxy"].append([x1,y1,x2,y2]); ball_dets["conf"].append(conf); ball_dets["cls"].append(cls)

        # Track
        players = player_tracker.update(player_dets)
        ball    = ball_tracker.update(ball_dets)

        # Ball fallback
        ball_box = None
        if isinstance(ball, list) and ball and isinstance(ball[0], dict) and all(k in ball[0] for k in ("x1","y1","x2","y2")):
            ball_box = ball[0]
        elif isinstance(ball, dict) and all(k in ball for k in ("x1","y1","x2","y2")):
            ball_box = ball
        else:
            with _override_lock:
                bias_xy = _override_ball_xy if app.config.get("BALL_LOCK") else None
            pick = pick_ball_candidate(frame, prev_gray, pitch_dt, ball_cands, last_ball_xy, W, H, pitch_mask_eroded, bias_xy=bias_xy)
            if pick is not None:
                x1, y1, x2, y2 = pick
                ball_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        # Team assignment
        if players:
            tass.observe(frame, [{"id": p["id"], "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]} for p in players])

        # ---- assign persistent labels ----
        active_tids = set()
        centroids_raw = []
        for p in (players or []):
            cx = int((p["x1"] + p["x2"]) / 2.0)
            cy = int((p["y1"] + p["y2"]) / 2.0)
            tid = int(p["id"])
            team = tass.get_team(p["id"])
            active_tids.add(tid)

            if tid in tid_to_label:
                label = tid_to_label[tid]
            else:
                best_label, best_score = None, -1e9
                for L, st in list(label_state.items()):
                    if (t_video - st["last_t"]) > LABEL_KEEP_SEC: continue
                    if team and st["team"] and team != st["team"]: continue
                    d = float(np.hypot(cx - st["pos"][0], cy - st["pos"][1]))
                    if d > REID_MAX_DIST_PX: continue
                    score = 0.7*(1.0/(1.0+d))
                    if score > best_score:
                        best_label, best_score = L, score
                if best_label is None:
                    best_label = next_label; next_label += 1
                tid_to_label[tid] = best_label
                label = best_label

            label_state[label] = {"pos": (cx, cy), "team": team, "tid": tid, "last_t": t_video}
            label_age[label] += 1
            centroids_raw.append({"cx": cx, "cy": cy, "tid": tid, "team": team, "label": label})

        for tid in list(tid_to_label.keys()):
            if tid not in active_tids:
                del tid_to_label[tid]

        # ---- ENFORCE 11 per team / 22 total, drop refs & ignored ----
        filtered = [it for it in centroids_raw
                    if it["label"] not in IGNORED_LABELS and it["team"] != "ref"]
        homes   = [it for it in filtered if it["team"] == "home"]
        aways   = [it for it in filtered if it["team"] == "away"]
        unknown = [it for it in filtered if it["team"] not in ("home","away")]

        def sort_key(it):   # prefer older, more stable tracks
            return (-label_age[it["label"]],)

        homes.sort(key=sort_key); aways.sort(key=sort_key); unknown.sort(key=sort_key)
        homes  = homes[:STRICT_MAX_PER_TEAM]
        aways  = aways[:STRICT_MAX_PER_TEAM]
        keep   = homes + aways
        if len(keep) < STRICT_MAX_PLAYERS and unknown:
            keep += unknown[:(STRICT_MAX_PLAYERS - len(keep))]
        centroids = keep

        with _last_centroids_lock:
            _last_centroids[:] = centroids

        # ---- draw overlay ----
        disp = frame.copy()
        name_map = dict(app.config.get("NAME_OVERRIDES", {}))
        selected_label = app.config.get("SELECTED_LABEL")

        draw_count = 0
        for it in centroids:
            cx, cy, label, team = it["cx"], it["cy"], it["label"], it["team"]
            color = (204,204,204)
            if team == "home": color = (255, 200, 30)
            if team == "away": color = (30, 200, 255)
            r = max(3, int(0.006*H))
            if label == selected_label:
                cv2.circle(disp, (cx, cy), r+3, (0,0,0), -1)
            cv2.circle(disp, (cx, cy), r, color, -1)
            tag = name_map.get(label) or f"P{label}"
            cv2.putText(disp, tag, (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, tag, (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            draw_count += 1

        # ball + trail + touches
        if ball_box:
            bx = (ball_box["x1"] + ball_box["x2"]) / 2.0
            by = (ball_box["y1"] + ball_box["y2"]) / 2.0
            now_t = time.time()

            # speed in px/s and (if calibrated) m/s
            speed_px = 0.0; speed_ms = None
            if last_ball_xy is not None and last_ball_t is not None:
                dt = max(1e-3, now_t - last_ball_t)
                speed_px = float(np.hypot(bx-last_ball_xy[0], by-last_ball_xy[1]) / dt)
                if H_PIX2M is not None:
                    bxm, bym = pix_to_m(bx, by)
                    lbxm, lbym = pix_to_m(last_ball_xy[0], last_ball_xy[1])
                    if bxm is not None and lbxm is not None:
                        speed_ms = float(np.hypot(bxm-lbxm, bym-lbym) / dt)

            last_ball_xy = (bx, by); last_ball_t = now_t
            stats["ball_seen"] += 1
            trail.append((now_t, (int(bx), int(by))))
            cv2.circle(disp, (int(bx), int(by)), max(4, int(0.005*H)), (40,40,255), -1)
            cv2.circle(disp, (int(bx), int(by)), max(4, int(0.005*H)), (255,255,255), 1)

            if centroids:
                dists = [np.hypot(bx-c["cx"], by-c["cy"]) for c in centroids]
                j = int(np.argmin(dists))
                lab = int(centroids[j]["label"])
                # choose appropriate speed threshold
                meet_speed = (speed_ms is not None and speed_ms >= BALL_MIN_SPEED_TOUCH_MS) or \
                             (speed_ms is None and speed_px >= BALL_MIN_SPEED_TOUCH_PX)
                if dists[j] <= BALL_TOUCH_RADIUS and meet_speed:
                    if (now_t - last_touch_time[lab]) >= 0.7:
                        touches[lab] += 1
                        last_touch_time[lab] = now_t
                        xi = int(np.clip(bx / W * HEAT_W, 0, HEAT_W-1))
                        yi = int(np.clip(by / H * HEAT_H, 0, HEAT_H-1))
                        heats[lab][yi, xi] += 1

        cutoff = time.time() - DRAW_TRAIL_SECONDS
        while trail and trail[0][0] < cutoff:
            trail.popleft()
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                p1 = trail[i-1][1]; p2 = trail[i][1]
                cv2.line(disp, p1, p2, (50,50,255), 2)

        stats["players_drawn"] = draw_count
        stats["frames"] += 1
        stats["last_t"] = t_video

        now = time.time()
        dt_ = now - last_stats_t
        if dt_ >= 0.5:
            stats["fps"] = stats["frames"] / max(dt_, 1e-6)
            stats["frames"] = 0
            last_stats_t = now

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()

# ---------- Routes ----------
@app.route("/")
def index():
    html = """
<!doctype html><html><head>
<meta charset="utf-8"/><title>Football Live Overlay</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:system-ui,Arial}
.top{max-width:1200px;margin:12px auto;padding:8px 12px;background:#181818;border:1px solid #2a2a2a;border-radius:8px;display:flex;align-items:center;gap:10px}
img{display:block;margin:8px auto;max-width:1200px;width:100%;background:#000}
.stat{margin-right:8px;display:inline-block}
input,select,button{background:#222;color:#eee;border:1px solid #333;border-radius:6px;padding:6px;cursor:pointer}
.switch{display:inline-flex;align-items:center;gap:6px}
</style></head><body>
<div class="top">
  <span class="stat">FPS: <b id="fps">-</b></span>
  <span class="stat">t≈ <b id="t">0.0</b>s</span>
  <span class="stat">players: <b id="pc">0</b></span>
  <span class="stat">ball seen: <b id="bc">0</b></span>

  <button id="btn_play">Play</button>
  <button id="btn_pause">Pause</button>
  <button id="btn_back">–1s</button>
  <button id="btn_fwd">+1s</button>
  <button id="btn_restart">Restart</button>
  <button id="btn_save">Save</button>
  <button id="btn_load">Load</button>

  <span>mode:
    <select id="mode">
      <option value="tag">Tag player</option>
      <option value="name">Name player</option>
      <option value="ball">Tag ball</option>
      <option value="cal">Calibrate</option>
    </select>
  </span>
  <span class="switch"><input type="checkbox" id="ball_lock"/><label for="ball_lock">Ball lock</label></span>
  <span><input id="name_in" placeholder="Type name (Name mode)" size="18"/></span>

  <div id="ps" style="margin-left:auto; background:#101010;border:1px solid #2a2a2a;border-radius:6px;padding:6px 10px">
    <b>Touches</b>
    <div id="ps_rows" style="font-size:12px; line-height:1.3"></div>
    <div style="opacity:.7;font-size:11px">Tips: Alt-click = 2nd nearest, Shift-click = ignore label, Calibrate: click 4 pitch corners</div>
  </div>
</div>
<img id="stream" src="/stream" />
<script>
const img = document.getElementById('stream');
const modeSel = document.getElementById('mode');
const nameIn  = document.getElementById('name_in');
const ballLock= document.getElementById('ball_lock');
let FPS = 30;

async function loopStats(){
  try{
    const r = await fetch('/stats'); const s = await r.json();
    document.getElementById('fps').innerText = (s.fps||0).toFixed(1);
    document.getElementById('t').innerText   = (s.last_t||0).toFixed(2);
    document.getElementById('pc').innerText  = s.players_drawn||0;
    document.getElementById('bc').innerText  = s.ball_seen||0;
    if (s.fps) FPS = s.fps;
  }catch(e){}
  setTimeout(loopStats, 700);
}
loopStats();

function imgCoords(ev){
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth  ? img.naturalWidth  / img.clientWidth  : 1;
  const scaleY = img.naturalHeight ? img.naturalHeight / img.clientHeight : 1;
  const x = (ev.clientX - rect.left) * scaleX;
  const y = (ev.clientY - rect.top)  * scaleY;
  return {x: Math.round(x), y: Math.round(y)};
}

img.addEventListener('click', async (ev) => {
  const {x,y} = imgCoords(ev);
  const mode = modeSel.value;
  const name = nameIn.value || "";
  const rank = ev.altKey ? 1 : 0;  // Alt = 2nd closest
  const ignore = ev.shiftKey ? true : false; // Shift = ignore nearest label
  try{
    const r = await fetch('/click', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({x,y,mode,name,rank,ignore,ball_lock: ballLock.checked})
    });
    const s = await r.json();
    if(!s.ok){ console.warn('click error', s); }
  }catch(e){ console.warn(e); }
});

ballLock.addEventListener('change', async ()=>{ try{ await fetch('/ui',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ball_lock:ballLock.checked})}); }catch(e){} });
modeSel.addEventListener('change', async ()=>{ try{ await fetch('/ui',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:modeSel.value})}); }catch(e){} });
nameIn.addEventListener('change', async ()=>{ try{ await fetch('/ui',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name_in:nameIn.value})}); }catch(e){} });

async function ctrl(cmd, payload={}){ try{ await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'}, body:JSON.stringify({cmd,...payload})}); }catch(e){} }
document.getElementById('btn_pause').onclick   = ()=> ctrl('pause');
document.getElementById('btn_play').onclick    = ()=> ctrl('play');
document.getElementById('btn_restart').onclick = ()=> ctrl('restart');
document.getElementById('btn_back').onclick    = ()=> ctrl('step', {frames: -Math.max(1, Math.round(FPS))});
document.getElementById('btn_fwd').onclick     = ()=> ctrl('step', {frames:  Math.max(1, Math.round(FPS))});

document.getElementById('btn_save').onclick = async ()=>{ try{ await fetch('/save_session',{method:'POST'});}catch(e){} };
document.getElementById('btn_load').onclick = async ()=>{ try{ await fetch('/load_session',{method:'POST'});}catch(e){} };

async function loopPlayerStats(){
  try{
    const r = await fetch('/player_stats');
    const m = await r.json();
    const box = document.getElementById('ps_rows');
    const rows = Object.entries(m)
      .sort((a,b)=> b[1].touches - a[1].touches)
      .slice(0,14)
      .map(([lab, v])=> `<div>${(v.name||('P'+lab))}: <b>${v.touches}</b></div>` )
      .join('');
    box.innerHTML = rows || '<div style="opacity:.7">no touches yet</div>';
  }catch(e){}
  setTimeout(loopPlayerStats, 1000);
}
loopPlayerStats();
</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/stream")
def stream():
    return Response(gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def get_stats():
    return jsonify({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in stats.items()})

@app.route("/player_stats")
def player_stats():
    names = app.config.get("NAME_OVERRIDES", {})
    out = {int(k): {"touches": int(v), "name": names.get(int(k))} for k, v in touches.items()}
    return jsonify(out)

@app.route("/heatmap/<int:label>.json")
def heatmap(label: int):
    grid = heats[int(label)]
    return jsonify(grid.astype(int).tolist())

@app.route("/ui", methods=["POST"])
def set_ui():
    data = request.get_json(force=True) or {}
    if "mode" in data: app.config["MODE"] = data["mode"]
    if "name_in" in data: app.config["NAME_INPUT"] = data["name_in"]
    if "ball_lock" in data: app.config["BALL_LOCK"] = bool(data["ball_lock"])
    return jsonify({"ok": True, "mode": app.config["MODE"], "ball_lock": app.config["BALL_LOCK"]})

@app.route("/click", methods=["POST"])
def click():
    data = request.get_json(force=True) or {}
    x = float(data.get("x", -1)); y = float(data.get("y", -1))
    mode = str(data.get("mode", app.config.get("MODE","tag")))
    name = str(data.get("name", app.config.get("NAME_INPUT",""))).strip()
    rank = int(data.get("rank", 0))
    ignore = bool(data.get("ignore", False))
    app.config["BALL_LOCK"] = bool(data.get("ball_lock", app.config.get("BALL_LOCK", False)))

    # calibration mode: collect 4 points and compute H
    if mode == "cal":
        calib_pts_px.append((x,y))
        have = False
        if len(calib_pts_px) >= 4:
            pts = np.array(calib_pts_px[:4], dtype=np.float32)
            idx = np.argsort(pts[:,0])
            L = pts[idx[:2]][np.argsort(pts[idx[:2],1])]
            R = pts[idx[2:]][np.argsort(pts[idx[2:],1])]
            src = np.array([L[0], L[1], R[0], R[1]], dtype=np.float32)
            dst = np.array([[0,0],[0,PITCH_H_M],[PITCH_W_M,0],[PITCH_W_M,PITCH_H_M]], dtype=np.float32)
            global H_PIX2M
            H_PIX2M, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
            calib_pts_px.clear()
            have = H_PIX2M is not None
        return jsonify({"ok": True, "action":"calib", "have_H": have, "collected": len(calib_pts_px)})

    with _last_centroids_lock:
        cents = list(_last_centroids)
    if not cents:
        return jsonify({"ok": False, "error": "no players this frame"})

    ranked = sorted(cents, key=lambda it: float(np.hypot(x - it["cx"], y - it["cy"])))
    best = ranked[min(max(rank,0), len(ranked)-1)]
    label = int(best["label"])
    app.config["SELECTED_LABEL"] = label

    if ignore:
        IGNORED_LABELS.add(label)
        return jsonify({"ok": True, "action": "ignored", "label": label})

    if mode == "name":
        if not name:
            return jsonify({"ok": False, "error": "empty name"})
        app.config.setdefault("NAME_OVERRIDES", {})[label] = name
        return jsonify({"ok": True, "action": "named", "label": label, "name": name})

    if mode == "ball":
        with _override_lock:
            global _override_ball_xy
            _override_ball_xy = (x, y)
        app.config["BALL_LOCK"] = True
        return jsonify({"ok": True, "action": "ball_bias", "x": x, "y": y, "ball_lock": True})

    return jsonify({"ok": True, "action": "tagged", "label": label})

@app.route("/control", methods=["POST"])
def control_route():
    data = request.get_json(force=True) or {}
    cmd = str(data.get("cmd"))
    frames = int(data.get("frames") or 0)
    with control_lock:
        if cmd == "pause":
            control["paused"] = True
        elif cmd == "play":
            control["paused"] = False
        elif cmd == "restart":
            control["restart"] = True
        elif cmd == "step":
            control["seek"] += frames
    return jsonify({"ok": True, **control})

@app.route("/save_session", methods=["POST"])
def save_session():
    payload = {
        "saved_at": dt.datetime.utcnow().isoformat()+"Z",
        "names": app.config.get("NAME_OVERRIDES", {}),
        "touches": {int(k): int(v) for k,v in touches.items()},
        "ignored": list(IGNORED_LABELS),
    }
    try:
        with open("session_cache.json","w") as f: json.dump(payload,f)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/load_session", methods=["POST"])
def load_session():
    try:
        with open("session_cache.json","r") as f: payload = json.load(f)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    app.config["NAME_OVERRIDES"] = {int(k): v for k,v in payload.get("names",{}).items()}
    IGNORED_LABELS.clear(); IGNORED_LABELS.update(payload.get("ignored",[]))
    for k,v in payload.get("touches",{}).items(): touches[int(k)] = int(v)
    return jsonify({"ok": True})
    
if __name__ == "__main__":
    # Run:  VIDEO_PATH=test_match.mp4 python live_server.py
    app.run(host="0.0.0.0", port=8000, threaded=True)
