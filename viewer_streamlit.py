import json, argparse, time
import numpy as np
import streamlit as st

# --- simple pitch drawing (105x68 default) ---
PITCH_W, PITCH_H = 105.0, 68.0

def load_tracks(path):
    with open(path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    # Expect a flat list like your smooth.py output
    tracks = tj.get("tracks") if isinstance(tj, dict) else tj
    if tracks is None: tracks = tj  # fallback if root is a list
    return tracks

def group_by_time(tracks):
    # t in seconds; we’ll build frames by rounded t*10 (0.1s steps)
    buckets = {}
    for r in tracks:
        t = float(r.get("t", 0.0))
        k = int(round(t*10))
        buckets.setdefault(k, []).append(r)
    return buckets

def to_metric(r):
    # Prefer meters if present, otherwise px scaled to pitch
    if "x_m" in r and "y_m" in r:
        return float(r["x_m"]), float(r["y_m"]), True
    if "x_px" in r and "y_px" in r:
        # naive px->m fallback if pitch not available; just normalize
        return float(r["x_px"]) / 10.0, float(r["y_px"]) / 10.0, False
    return None, None, False

def main():
    st.set_page_config(page_title="Scoutiqo Analyzer View", layout="wide")
    st.title("⚽ Scoutiqo — Analyzer View (tracks.json)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="Path to tracks.json")
    args, _ = ap.parse_known_args()

    tracks = load_tracks(args.tracks)
    if not tracks:
        st.error("No tracks found.")
        return

    buckets = group_by_time(tracks)
    keys = sorted(buckets.keys())
    t0 = keys[0]/10.0
    t1 = keys[-1]/10.0

    colL, colR = st.columns([2,1])
    with colR:
        speed = st.slider("Playback speed (x)", 0.25, 4.0, 1.0, 0.25)
        show_ids = st.checkbox("Show player IDs", True)
        show_ball = st.checkbox("Show ball", True)
        tpos = st.slider("Jump to time (s)", t0, t1, t0, 0.1)

    # canvas as an SVG via st.pyplot for simplicity
    canvas = colL.empty()

    # run once through timeline starting at tpos
    start_idx = min(range(len(keys)), key=lambda i: abs(keys[i]/10.0 - tpos))
    for i in range(start_idx, len(keys)):
        frame_t = keys[i]/10.0
        frame = buckets[keys[i]]
        # separate players & ball
        players, ball = [], []
        for r in frame:
            x, y, _ = to_metric(r)
            if x is None: continue
            if r.get("type") == "ball":
                ball.append((x, y))
            else:
                players.append((x, y, r.get("id")))

        # draw a simple pitch
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,6))
        ax = plt.gca()
        ax.set_xlim(0, PITCH_W); ax.set_ylim(0, PITCH_H)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"t = {frame_t:.1f}s")
        # pitch lines
        ax.add_patch(plt.Rectangle((0,0), PITCH_W, PITCH_H, fill=False))
        ax.axvline(PITCH_W/2, linestyle="--", linewidth=1)

        # plot players
        if players:
            xs = [p[0] for p in players]; ys=[p[1] for p in players]
            ax.scatter(xs, ys, s=40)
            if show_ids:
                for (x,y,pid) in players:
                    ax.text(x+0.5, y, f"{pid}", fontsize=8)

        # plot ball
        if show_ball and ball:
            bx=[b[0] for b in ball]; by=[b[1] for b in ball]
            ax.scatter(bx, by, s=60, marker="o")

        canvas.pyplot(fig, clear_figure=True)
        plt.close(fig)
        time.sleep(max(0.01, 0.1/speed))

if __name__ == "__main__":
    main()
