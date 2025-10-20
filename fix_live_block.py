import io, re

p = r".\server\server.py"
src = io.open(p, "r", encoding="utf-8").read()

# Replace the whole /live block (including its surrounding try/except) with a clean version.
pat = re.compile(
    r"""(?sx)
    ^try:\s*
    \s*from\ fastapi\.responses\ import\ StreamingResponse
    .*?
    ^except\ Exception\ as\ _e:\s*\n
    \s*#\ keep\ API\ alive\ even\ if\ OpenCV/live_core\ not\ present\s*\n
    \s*pass\s*$
    """,
    re.M,
)

new = r'''
try:
    from fastapi.responses import StreamingResponse
    import cv2
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedSmoother,
        compute_control, draw_overlay
    )

    @app.get("/live")
    def live(src: str, resize_w: int = 1280, skip: int = 1):
        def gen():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")
            st = init_tracker_state()
            smooth = SpeedSmoother()
            i = 0
            last_ball = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if resize_w:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (resize_w, int(resize_w * h / w)))

                if i % max(1, skip) == 0:
                    dets = run_player_detector(frame)
                    tids, tracks = run_tracker(st, dets)
                    ball = detect_ball(frame)
                    assign_teams(st, frame, tracks)
                    speeds = smooth.update(st, tracks)
                    control = compute_control(st, tracks, ball)
                    draw_overlay(frame, tracks, speeds, ball, st.team_of, control)

                i += 1
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            cap.release()

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
except Exception as _e:
    # keep API alive even if OpenCV/live_core not present
    pass
'''.strip() + "\n"

out, n = pat.subn(new, src)
if n == 0:
    print("No /live block matched; aborting to avoid damaging file.")
else:
    io.open(p, "w", encoding="utf-8", newline="\n").write(out)
    print(f"Replaced /live block (matches: {n})")
