import json, time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

PITCH_W, PITCH_H = 105.0, 68.0

def group_by_time(tracks):
    buckets = {}
    for r in tracks:
        try:
            t = float(r.get("t", 0.0))
        except Exception:
            continue
        k = int(round(t * 10))  # 0.1s bins
        buckets.setdefault(k, []).append(r)
    return buckets

def to_metric(r):
    # Prefer meters if present, otherwise naive px→m fallback
    if "x_m" in r and "y_m" in r:
        return float(r["x_m"]), float(r["y_m"]), True
    if "x_px" in r and "y_px" in r:
        return float(r["x_px"]) / 10.0, float(r["y_px"]) / 10.0, False
    return None, None, False

def main():
    st.set_page_config(page_title="ScoutIQO — Analyzer View", layout="wide")
    st.title("⚽ ScoutIQO — Analyzer View")

    src = st.sidebar.file_uploader("Upload tracks.json", type=["json"])
    url = st.sidebar.text_input("...or paste JSON URL")

    if src:
        tj = json.load(src)
    elif url:
        import requests
        tj = requests.get(url, timeout=10).json()
    else:
        st.info("Upload a tracks.json or paste a URL.")
        return

    tracks = tj.get("tracks", tj if isinstance(tj, list) else [])
    if not tracks:
        st.error("No tracks found in JSON.")
        return

    buckets = group_by_time(tracks)
    keys = sorted(buckets.keys())
    if not keys:
        st.error("No time buckets found.")
        return

    t0 = keys[0] / 10.0
    t1 = keys[-1] / 10.0

    colL, colR = st.columns([2, 1])
    with colR:
        speed = st.slider("Playback speed (×)", 0.25, 4.0, 1.0, 0.25)
        show_ids = st.checkbox("Show player IDs", True)
        show_ball = st.checkbox("Show ball", True)
        tpos = st.slider("Jump to time (s)", float(t0), float(t1), float(t0), 0.1)

    plot_slot = colL.empty()
    fig, ax = plt.subplots(figsize=(10, 6))

    # find nearest start bucket
    start_idx = min(range(len(keys)), key=lambda i: abs(keys[i] / 10.0 - tpos))

    for i in range(start_idx, len(keys)):
        frame_t = keys[i] / 10.0
        frame = buckets[keys[i]]

        players, ball = [], []
        for r in frame:
            x, y, _ = to_metric(r)
            if x is None:
                continue
            if r.get("type") == "ball":
                ball.append((x, y))
            else:
                players.append((x, y, r.get("id")))

        ax.clear()
        ax.set_xlim(0, PITCH_W); ax.set_ylim(0, PITCH_H)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"t = {frame_t:.1f}s")

        # pitch
        ax.add_patch(plt.Rectangle((0, 0), PITCH_W, PITCH_H, fill=False, linewidth=1))
        ax.axvline(PITCH_W / 2, linestyle="--", linewidth=1)

        # players
        if players:
            xs = [p[0] for p in players]; ys = [p[1] for p in players]
            ax.scatter(xs, ys, s=40)
            if show_ids:
                for (x, y, pid) in players:
                    ax.text(x + 0.5, y, f"{pid}", fontsize=8)

        # ball
        if show_ball and ball:
            bx = [b[0] for b in ball]; by = [b[1] for b in ball]
            ax.scatter(bx, by, s=60, marker="o")

        plot_slot.pyplot(fig, clear_figure=False)
        time.sleep(max(0.01, 0.1 / speed))

if __name__ == "__main__":
    main()
