import io, re

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()

# ---------- A) Ensure app = FastAPI() exists before any @app.get ----------
lines = txt.splitlines()

def has_app_declaration(s):
    return re.search(r'^\s*app\s*=\s*FastAPI\s*\(', s, flags=re.M) is not None

def first_decorator_idx(lines):
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("@app."):
            return i
    return None

need_app_header = (not has_app_declaration(txt))
first_deco = first_decorator_idx(lines)

header = (
    "from fastapi import FastAPI\n"
    "app = FastAPI()\n"
)

if need_app_header:
    # If there is any import for FastAPI later, leave it; duplicate imports are fine.
    insert_at = 0
    if first_deco is not None:
        insert_at = 0
    # Prepend header
    lines = [header] + lines

txt = "\n".join(lines) + "\n"


# ---------- B) Replace the whole LIVE_PRO block with a clean stub ----------
# Find markers if present, else look for @app.get("/live_pro")
start_idx = None
end_idx = None
L = txt.splitlines()

for i, ln in enumerate(L):
    if "LIVE_PRO" in ln and "all-in-one" in ln:
        start_idx = i
        break

if start_idx is None:
    # fallback: find the decorator line
    for i, ln in enumerate(L):
        if ln.strip().startswith('@app.get("/live_pro")'):
            # try to find a "try:" above to include the whole block
            start_idx = i
            for j in range(i-1, -1, -1):
                if L[j].strip().endswith("try:"):
                    start_idx = j
                    break
            break

if start_idx is not None:
    # find end: next top-level section marker or EOF
    end_idx = len(L)-1
    for j in range(start_idx+1, len(L)):
        if L[j].startswith("# ===================") or L[j].startswith("# ================================================================="):
            end_idx = j-1
            break

    # replace with a minimal, correct block that compiles
    clean_block = r'''
# =================== LIVE_PRO (rescued minimal stub) ===================
try:
    from fastapi.responses import StreamingResponse
    import cv2, numpy as np, time
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedEstimator, estimate_m_per_px,
        compute_control, draw_overlay, estimate_pitch_bounds
    )
    from .pitch import build_pitch_mask, box_kept_by_mask
    from ultralytics import YOLO
    from .numbering import SquadNumberer
    from .metrics_core import MetricsState, nearest_to_ball, in_final_third

    try:
        from .learn_state import load_state, save_state
    except Exception:
        def load_state():
            return {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
        def save_state(**kwargs):
            pass

    LIVE_METRICS = {"ok": False, "snapshot": {}}

    @app.get("/metrics")
    def metrics():
        return LIVE_METRICS

    _det_model = None
    def _ensure_det():
        global _det_model
        if _det_model is None:
            _det_model = YOLO("yolov8n.pt")

    def yolo_person_and_ball(frame, conf=0.35):
        _ensure_det()
        r = _det_model.predict(source=frame, conf=conf, verbose=False)
        boxes, balls = [], []
        names = getattr(_det_model.model, "names", {})
        for b in r[0].boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            label = names.get(c, "")
            if c == 0:
                boxes.append([x1,y1,x2,y2,cf])
            elif label in ("sports ball", "sportsball", "ball"):
                balls.append([x1,y1,x2,y2,cf])
        ball = max(balls, key=lambda t:t[4]) if balls else None
        return boxes, ball

    @app.get("/live_pro")
    def live_pro(
        src: str,
        resize_w: int = 1280,
        skip: int = 1,
        exclude_top_pct: float | None = None,
        min_overlap: float | None = None,
        conf_min: float | None = None
    ):
        # Minimal stub to keep server bootable; replace with your full logic later.
        return {"ok": True, "message": "live_pro stub running"}
except Exception:
    # keep import/boot resilient even if optional deps fail
    pass
# =======================================================================
'''.strip("\n")

    newL = L[:start_idx] + [clean_block] + L[end_idx+1:]
    txt = "\n".join(newL) + "\n"

# ---------- C) Write back ----------
io.open(P, "w", encoding="utf-8", newline="\n").write(txt)
print("Rescued: ensured app exists early and rebuilt LIVE_PRO as a minimal stub.")
