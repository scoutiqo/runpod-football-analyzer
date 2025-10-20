import cv2, numpy as np, time, math
from ultralytics import YOLO
from .pitch import build_pitch_mask, box_kept_by_mask

# ---------- State ----------
class TrackerState:
    def __init__(self):
        self.team_of = {}
        self.color_hist = {}
        self.prev_center = {}     # tid -> (cx, cy, t)
        self.pitch_mask = None
        self.exclude_rect = None
        self.m_per_px = 0.25      # will be estimated on first frame

def init_tracker_state():
    return TrackerState()

# ---------- Detector (YOLOv8 person) ----------
_model = None
def run_player_detector(frame, conf=0.5):
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    res = _model.predict(source=frame, verbose=False, conf=conf, classes=[0])  # person
    dets=[]
    for b in res[0].boxes:
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0].item())
        dets.append([x1,y1,x2,y2,conf])
    return dets

# ---------- Tracker (ByteTrack via supervision) ----------
_bt = None
def _ensure_bt(fps: float):
    global _bt
    if _bt is None:
        import supervision as sv
        _bt = sv.ByteTrack(frame_rate=max(1.0, float(fps or 25.0)))
    return _bt

def run_tracker(state, dets, fps=25.0):
    import supervision as sv
    tracker = _ensure_bt(fps)
    if len(dets)==0:
        tracks = tracker.update_with_detections(sv.Detections.empty())
        ids = [] if tracks.tracker_id is None else list(map(int, tracks.tracker_id))
        return ids, {}
    xyxy = np.array([d[:4] for d in dets], dtype=float)
    conf = np.array([d[4] for d in dets], dtype=float)
    cls   = np.zeros(len(dets), dtype=int)
    det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
    tracks = tracker.update_with_detections(det)
    ids = [] if tracks.tracker_id is None else list(map(int, tracks.tracker_id))
    boxes = tracks.xyxy if tracks.xyxy is not None else np.zeros((0,4))
    return ids, {tid: box.tolist() for tid, box in zip(ids, boxes)}

# ---------- Ball (placeholder heuristic) ----------
def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,0,185), (180,60,255))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w*h < 25: return None
    return [x,y,x+w,y+h]

# ---------- Team assign (color hist k=2) ----------
def _jersey_patch(frame, box):
    x1,y1,x2,y2 = map(int, box); y_mid = y1 + int(0.35*(y2-y1))
    return frame[y1:max(y1,y_mid), x1:x2]

def assign_teams(state, frame, tracks):
    samples=[]; tids=[]
    for tid,box in tracks.items():
        crop=_jersey_patch(frame, box)
        if crop.size==0: continue
        hsv=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256])
        hist=cv2.normalize(hist,None).flatten()
        samples.append(hist); tids.append(tid)
        state.color_hist[tid]=hist
    if len(samples)<2: return
    X=np.vstack(samples); c1=X[0]; c2=X[-1]
    for _ in range(5):
        d1=np.linalg.norm(X-c1,axis=1); d2=np.linalg.norm(X-c2,axis=1)
        g1=X[d1<=d2]; g2=X[d1>d2]
        if len(g1): c1=g1.mean(axis=0)
        if len(g2): c2=g2.mean(axis=0)
    for i,tid in enumerate(tids):
        d1=np.linalg.norm(X[i]-c1); d2=np.linalg.norm(X[i]-c2)
        state.team_of[tid]=0 if d1<=d2 else 1

# ---------- Scale from pitch mask ----------
def estimate_m_per_px(mask, pitch_width_m: float = 68.0) -> float:
    """Estimate meters-per-pixel from median pitch width in the mask."""
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) < 10:
        return 0.25  # fallback
    widths=[]
    for y in ys[:: max(1, len(ys)//120)]:  # sample up to ~120 rows
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < 5: continue
        widths.append(xs.max() - xs.min())
    if not widths:
        return 0.25
    median_w = float(np.median(widths))
    if median_w <= 1:
        return 0.25
    return pitch_width_m / median_w

# ---------- Speed estimator with smoothing & self-calibration ----------
class SpeedEstimator:
    def __init__(self, m_per_px: float, max_kmh: float = 38.0, static_kmh: float = 0.4,
                 target_p95_kmh: float = 28.0, adapt_rate: float = 0.08):
        self.m_per_px = m_per_px
        self.max_kmh = max_kmh
        self.static_kmh = static_kmh
        self.samples = []            # recent km/h samples for auto-calibration
        self.target_p95 = target_p95_kmh
        self.adapt_rate = adapt_rate
        self.last_adapt = time.time()

    def _adapt_scale(self):
        if len(self.samples) < 60:
            return
        obs = float(np.percentile(np.array(self.samples), 95))
        if obs > 1.0:
            ratio = self.target_p95 / obs
            # multiplicative update (slow, stable)
            self.m_per_px = max(1e-5, self.m_per_px * (1 - self.adapt_rate + self.adapt_rate * ratio))
        self.samples.clear()

    def update(self, state: TrackerState, tracks: dict):
        out={}
        t=time.time()
        for tid,b in tracks.items():
            x1,y1,x2,y2=b
            cx=(x1+x2)/2; cy=(y1+y2)/2
            prev=state.prev_center.get(tid)
            if prev:
                px,py,pt=prev
                dt=max(t-pt, 1e-3)
                # distance in meters
                d_pix = ((cx-px)**2+(cy-py)**2)**0.5
                d_m   = d_pix * self.m_per_px
                kmh   = (d_m / dt) * 3.6
                # spike guard & clamp
                if kmh > self.max_kmh*1.5:
                    kmh = self.max_kmh
                # small motions = 0
                if kmh < self.static_kmh:
                    kmh = 0.0
                # exponential smoothing per id
                sm_prev = getattr(self, f"s_{tid}", None)
                alpha = 0.35
                kmh_s = (1-alpha)*sm_prev + alpha*kmh if sm_prev is not None else kmh
                setattr(self, f"s_{tid}", kmh_s)
                out[tid]=min(self.max_kmh, kmh_s)
                self.samples.append(out[tid])
            state.prev_center[tid]=(cx,cy,t)

        # periodic auto-calibration toward human speed distribution
        if time.time() - self.last_adapt > 2.5:
            self._adapt_scale()
            self.last_adapt = time.time()
        return out

# ---------- Control metric ----------
def compute_control(state, tracks, ball):
    centers={tid:((b[0]+b[2])/2,(b[1]+b[3])/2) for tid,b in tracks.items()}
    if not centers: return {"team0":50.0,"team1":50.0}
    if ball:
        bx=(ball[0]+ball[2])/2; by=(ball[1]+ball[3])/2
    else:
        bx=np.mean([c[0] for c in centers.values()]); by=np.mean([c[1] for c in centers.values()])
    s=[0.0,0.0]
    for tid,(cx,cy) in centers.items():
        d=max(1.0, ((cx-bx)**2+(cy-by)**2)**0.5)
        w=math.exp(-d/120.0)
        team=state.team_of.get(tid,0)
        s[team]+=w
    tot=s[0]+s[1] or 1.0
    return {"team0":100*s[0]/tot, "team1":100*s[1]/tot}

# ---------- Draw ----------
def draw_overlay(img, tracks, speeds, ball, team_of, control):
    for tid,b in tracks.items():
        x1,y1,x2,y2=map(int,b)
        team=team_of.get(tid,0)
        color=(0,255,0) if team==0 else (255,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        sp=speeds.get(tid,0.0)
        cv2.putText(img, f"{tid} {sp:.1f} km/h", (x1,max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cx=int((x1+x2)/2); cy=int(y2); cv2.circle(img,(cx,cy),8,color,2)
    if ball:
        x1,y1,x2,y2=map(int,ball)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.circle(img,(int((x1+x2)/2),int((y1+y2)/2)),6,(0,0,255),-1)
    panel=np.zeros((60,260,3),dtype=np.uint8)
    cv2.putText(panel,f"Team 1 Control: {control.get('team0',0):5.1f}%",(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
    cv2.putText(panel,f"Team 2 Control: {control.get('team1',0):5.1f}%",(8,48),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
    h,w=img.shape[:2]; img[h-60:h, w-260:w]=cv2.addWeighted(img[h-60:h, w-260:w],0.3,panel,0.7,0)

# ---- hooks for metrics pipeline (used by /live2) ----
def estimate_pitch_bounds(mask):
    ys, xs = np.where(mask>0)
    if xs.size==0: return (0,0,mask.shape[1]-1,mask.shape[0]-1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ---- hooks for metrics pipeline (used by /live2) ----
def estimate_pitch_bounds(mask):
    ys, xs = np.where(mask>0)
    if xs.size==0: return (0,0,mask.shape[1]-1,mask.shape[0]-1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ---- helper used by /live_pro ----
def estimate_pitch_bounds(mask):
    ys, xs = np.where(mask>0)
    if xs.size==0:
        return (0,0,mask.shape[1]-1,mask.shape[0]-1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

