import io, re, sys, pathlib

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
orig = txt

# --- A) Ensure numpy import once (top-level) ---------------------------------
if re.search(r'^\s*import\s+numpy\s+as\s+np\b', txt, flags=re.M) is None:
    # place after the first standard import block
    m = re.search(r'^(from\s+\w[^\n]+\n|import\s+\w[^\n]*\n)+', txt, flags=re.M)
    ins_at = m.end() if m else 0
    txt = txt[:ins_at] + "import numpy as np\n" + txt[ins_at:]

# --- B) Normalize StaticFiles: remove duplicates, mount once near app --------
# remove any repeated app.mount("/web", StaticFiles(...)) lines
txt = re.sub(r'(?m)^\s*app\.mount\(\s*[\'"]\/web[\'"]\s*,\s*StaticFiles\([^\)]*\)\s*,\s*name\s*=\s*[\'"]web[\'"]\s*\)\s*$', "", txt)

# ensure a single guarded mount after app exists
def ensure_staticfiles_once(s: str) -> str:
    if "from fastapi.staticfiles import StaticFiles" not in s and \
       "from starlette.staticfiles import StaticFiles" not in s:
        # prefer FastAPI’s StaticFiles
        s = re.sub(r'(from\s+fastapi\s+import\s+[^\n]+\n)',
                   r'\1from fastapi.staticfiles import StaticFiles\n',
                   s, count=1)
        if "from fastapi.staticfiles import StaticFiles" not in s:
            # if no fastapi import line found, just inject near top
            s = "from fastapi.staticfiles import StaticFiles\n" + s

    # find app = FastAPI(...) and insert mount after it
    m = re.search(r'^\s*app\s*=\s*FastAPI\s*\(.*?\)\s*$', s, flags=re.M)
    if m:
        i = m.end()
        mount = '\n# Serve minimal UI once\ntry:\n    app.mount("/web", StaticFiles(directory="web", html=True), name="web")\nexcept Exception:\n    pass\n'
        s = s[:i] + mount + s[i:]
    return s

txt = ensure_staticfiles_once(txt)

# --- C) Replace the BROKEN /live_pro block with a complete one ---------------
# Find the FIRST occurrence of @app.get("/live_pro") and replace its def body
pat_start = re.search(r'(?m)^\s*@app\.get\(["\']/live_pro["\']\)\s*$', txt)
if pat_start:
    # start at the next "def live_pro("
    mdef = re.search(r'(?m)^\s*def\s+live_pro\s*\(', txt[pat_start.end():])
    if mdef:
        s_idx = pat_start.end() + mdef.start()
        # find the end of this block: next top-level decorator or try/except or EOF
        cut = re.search(r'(?m)^\s*@app\.get\(|^\s*try:\s*$|^\s*# =+|^\s*# ---', txt[s_idx:])
        e_idx = s_idx + (cut.start() if cut else len(txt[s_idx:]))
        # replacement implementation (complete & self-contained)
        repl = r"""
@app.get("/live_pro")
def live_pro(
    src: str,
    resize_w: int = 1280,
    skip: int = 1,
    exclude_top_pct: float | None = None,
    min_overlap: float | None = None,
    conf_min: float | None = None
):
    # lazy import so startup doesn't break on cv2/ultralytics absence
    import cv2, time
    from ultralytics import YOLO
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedEstimator, estimate_m_per_px,
        compute_control, estimate_pitch_bounds
    )
    from .pitch import build_pitch_mask, box_kept_by_mask
    from .numbering import SquadNumberer
    from .metrics_core import MetricsState, nearest_to_ball, in_final_third
    try:
        from .learn_state import load_state, save_state
    except Exception:
        def load_state(): return {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
        def save_state(**kwargs): pass

    cfg = load_state()
    if exclude_top_pct is None: exclude_top_pct = float(cfg.get("exclude_top_pct", 0.25))
    if min_overlap is None:     min_overlap     = float(cfg.get("min_overlap", 0.60))
    if conf_min is None:        conf_min        = float(cfg.get("conf_min", 0.60))

    # YOLO only for ball/person (fallback if your own detector isn't good on ball)
    _det_model = None
    def _ensure_det():
        nonlocal _det_model
        if _det_model is None:
            _det_model = YOLO("yolov8n.pt")  # auto-download

    def yolo_person_and_ball(frame, conf=0.35):
        _ensure_det()
        r = _det_model.predict(source=frame, conf=conf, verbose=False)
        boxes=[]; balls=[]
        names = _det_model.model.names
        for b in r[0].boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            label = names.get(c, "")
            if c==0:   # person
                boxes.append([x1,y1,x2,y2,cf])
            elif label in ("sports ball","sportsball","ball"):
                balls.append([x1,y1,x2,y2,cf])
        ball = max(balls, key=lambda t:t[4]) if balls else None
        return boxes, ball

    from fastapi.responses import StreamingResponse

    def gen():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {src}")

        st   = init_tracker_state()
        metr = MetricsState()
        nums = SquadNumberer()

        # boot frame
        ok, frame = cap.read()
        if not ok:
            cap.release(); raise RuntimeError("No frames.")
        if resize_w:
            h0,w0 = frame.shape[:2]
            frame = cv2.resize(frame, (resize_w, int(resize_w*h0/w0)))

        H,W = frame.shape[:2]
        excl = (0,0,W, int(H*exclude_top_pct))
        st.pitch_mask = build_pitch_mask(frame, exclude_rect=excl)
        bounds = estimate_pitch_bounds(st.pitch_mask)
        st.m_per_px  = estimate_m_per_px(st.pitch_mask)
        speed        = SpeedEstimator(m_per_px=st.m_per_px)

        i = 0
        last_ball_xy = None
        while True:
            if i>0:
                ok, frame = cap.read()
                if not ok: break
                if resize_w:
                    h,w = frame.shape[:2]
                    frame = cv2.resize(frame, (resize_w, int(resize_w*h/w)))

            if i % max(1, skip) == 0:
                # players
                dets = run_player_detector(frame, conf=conf_min)
                dets = [d for d in dets if box_kept_by_mask(d[:4], st.pitch_mask, min_overlap=min_overlap)]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                tids, tracks = run_tracker(st, dets, fps=fps)
                assign_teams(st, frame, tracks)

                # numbers bookkeeping
                for tid in tracks.keys():
                    team = st.team_of.get(tid, 0)
                    nums.touch(team, tid)
                nums.gc()

                # ball (fallback to YOLO if detect_ball returns None)
                ball = detect_ball(frame)
                ball_xy = None
                if ball:
                    x1,y1,x2,y2 = ball
                    ball_xy = ((x1+x2)/2, (y1+y2)/2)
                if ball_xy is None:
                    boxes, ball_det = yolo_person_and_ball(frame, conf=0.35)
                    if ball_det:
                        x1,y1,x2,y2,_ = ball_det
                        ball_xy = ((x1+x2)/2, (y1+y2)/2)
                if ball_xy is None and last_ball_xy is not None:
                    ball_xy = last_ball_xy
                if ball_xy is None:
                    ball_xy = (W/2, H/2)
                last_ball_xy = ball_xy
                bx, by = ball_xy

                # simple ownership by nearest
                owner_tid, dist = (None, None)
                if tracks:
                    owner_tid, dist = min(
                        ((tid, (( ( (b[0]+b[2])*0.5 - bx )**2 + ( (b[1]+b[3])*0.5 - by )**2 )**0.5))
                         for tid,b in tracks.items()),
                        key=lambda t:t[1],
                        default=(None, None)
                    )

                # metrics
                if owner_tid is not None:
                    team = st.team_of.get(owner_tid, 0)
                    metr.update_possession(team)
                    sq = nums.get(team, owner_tid)
                    inside, xnorm = in_final_third((bx,by), bounds)
                    metr.mark_touch(team, sq, in_final_third=inside)

                # speeds & control
                speeds = speed.update(st, tracks)
                control = compute_control(st, tracks, [bx-5,by-5,bx+5,by+5])

                # overlay: numbers + speed + ball + possession
                for tid, box in tracks.items():
                    team = st.team_of.get(tid,0)
                    sq = nums.get(team, tid)
                    x1,y1,x2,y2 = map(int, box)
                    color = (0,255,0) if team==0 else (255,255,0)
                    tag = ("A" if team==0 else "B") + str(sq)
                    cv2.putText(frame, f"{tag} {speeds.get(tid,0.0):.1f} km/h",
                                (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cx=int((x1+x2)/2); cy=int(y2); cv2.circle(frame,(cx,cy),8,color,2)

                cv2.circle(frame, (int(bx),int(by)), 6, (0,0,255), -1)

                tot = max(1e-3, sum(metr.possession))
                pA = 100*metr.possession[0]/tot; pB = 100*metr.possession[1]/tot
                panel = (np.zeros((60,260,3), dtype=np.uint8))
                cv2.putText(panel, f"Team A Control: {pA:5.1f}%", (8,22), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                cv2.putText(panel, f"Team B Control: {pB:5.1f}%", (8,48), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                h,w=frame.shape[:2]; frame[h-60:h, w-260:w] = cv2.addWeighted(frame[h-60:h, w-260:w],0.3,panel,0.7,0)

            i += 1
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok: 
                continue
            yield (b"--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n" + jpg.tobytes() + b"\\r\\n")

        cap.release()
        try:
            save_state(
                m_per_px=float(st.m_per_px),
                exclude_top_pct=float(exclude_top_pct),
                min_overlap=float(min_overlap),
                conf_min=float(conf_min)
            )
        except Exception:
            pass

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
""".strip("\n")
        txt = txt[:pat_start.start()] + repl + "\n" + txt[e_idx:]

# --- D) Optional: remove 2nd duplicate /live2 block to keep file lean --------
# (keep the first; remove later duplicates)
occ = list(re.finditer(r'(?m)^\s*@app\.get\(["\']/live2["\']\)\s*$', txt))
if len(occ) > 1:
    # remove everything from second occurrence to just before the next top-level marker
    s2 = occ[1].start()
    cut = re.search(r'(?m)^\s*@app\.get\(|^\s*try:\s*$|^\s*# =+|^\s*# ---|^\s*# ===', txt[occ[1].end():])
    e2 = occ[1].end() + (cut.start() if cut else 0)
    txt = txt[:s2] + txt[e2:]

io.open(P, "w", encoding="utf-8", newline="\n").write(txt)
print("server.py surgically repaired.")
