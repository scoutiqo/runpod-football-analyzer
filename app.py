import os, time, json, urllib.parse, requests, io
import streamlit as st

RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_ID", "br8xmojkyvacw7")  # set your endpoint id
RUNPOD_API_KEY  = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_RUN_URL  = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run"
RUNPOD_STAT_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/"

st.set_page_config(layout="wide", page_title="Scoutiqo – Analyzer")

st.title("⚽ Scoutiqo – End-User Analyzer")

# --- helpers ---
def dbx_to_direct(url: str) -> str:
    if "dropbox.com" in url:
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        q["dl"] = ["1"]
        u = u._replace(netloc="www.dropbox.com", query=urllib.parse.urlencode(q, doseq=True))
        return urllib.parse.urlunparse(u)
    return url

def start_job(player_id: str, video_url: str) -> str:
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    payload = {"input": {"player_id": player_id, "video_url": video_url}}
    r = requests.post(RUNPOD_RUN_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def poll_job(req_id: str):
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    while True:
        r = requests.get(RUNPOD_STAT_URL + req_id, headers=headers, timeout=60)
        r.raise_for_status()
        j = r.json()
        status = j.get("status")
        logs = "\n".join(j.get("streamLogs", [])[-8:])
        yield status, j, logs
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)

def load_tracks_from_url(url: str):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    # supports either {"tracks":[...]} or [...]
    return data.get("tracks", data)

# --- sidebar inputs ---
with st.sidebar:
    st.header("Input")
    player_id = st.text_input("Player ID (label)", "player_001")
    url_mode = st.radio("Video source", ["Paste URL", "Upload file"])
    video_url = ""
    upload_bytes = None

    if url_mode == "Paste URL":
        url_in = st.text_input("Video URL (Dropbox/HTTP)", "")
        if url_in:
            video_url = dbx_to_direct(url_in)
            st.caption("Using direct link: " + video_url[:90] + ("..." if len(video_url)>90 else ""))
    else:
        f = st.file_uploader("Upload MP4", type=["mp4", "mov", "mkv"])
        if f:
            upload_bytes = f.read()

    run_btn = st.button("Analyze")

colL, colR = st.columns([2,1])

# --- action ---
if run_btn:
    if not RUNPOD_API_KEY:
        st.error("Server is not configured. Missing RUNPOD_API_KEY.")
    elif url_mode == "Paste URL" and not video_url:
        st.error("Please paste a video URL.")
    elif url_mode == "Upload file" and not upload_bytes:
        st.error("Please upload a video.")
    else:
        # If uploading file, stream to a temp file host the pod can reach.
        # For MVP, require URL mode (simpler). You can integrate Supabase/Tus later.
        if url_mode == "Upload file":
            st.error("For this MVP, use a URL (Dropbox direct link).")
        else:
            with st.status("Submitting job…", expanded=True) as s:
                try:
                    req_id = start_job(player_id, video_url)
                except Exception as e:
                    st.exception(e)
                    st.stop()
                st.write("Job ID: ", req_id)

            with st.status("Processing on GPU…", expanded=True) as s:
                tracks_url = None
                last_logs = ""
                for status, raw, logs in poll_job(req_id):
                    if logs and logs != last_logs:
                        st.code(logs)
                        last_logs = logs
                    if status == "COMPLETED":
                        out = raw.get("output", {})
                        arts = out.get("artifacts", {})
                        tracks_url = arts.get("tracks.json")
                        if not tracks_url:
                            st.warning("No tracks.json returned. Full output shown below.")
                            st.json(out)
                        break
                    if status == "FAILED":
                        st.error("Run failed")
                        st.json(raw)
                        st.stop()

            if tracks_url:
                st.success("Analysis complete. Loading tracks…")
                try:
                    tracks = load_tracks_from_url(tracks_url)
                except Exception as e:
                    st.exception(e)
                    st.stop()

                # ---- simple analyzer view ----
                st.subheader("Pitch View")
                import matplotlib.pyplot as plt
                PITCH_W, PITCH_H = 105.0, 68.0

                # group by time (0.1s buckets)
                buckets = {}
                for r in tracks:
                    t = float(r.get("t", 0.0))
                    k = int(round(t*10))
                    buckets.setdefault(k, []).append(r)
                keys = sorted(buckets.keys())
                if not keys:
                    st.warning("No track points.")
                    st.stop()

                spd = st.slider("Playback speed (x)", 0.25, 3.0, 1.0, 0.25)
                show_ids = st.checkbox("Show player IDs", True)
                show_ball = st.checkbox("Show ball", True)
                canvas = st.empty()

                for i in range(len(keys)):
                    frame = buckets[keys[i]]
                    players, ball = [], []
                    for r in frame:
                        if "x_m" in r and "y_m" in r:
                            x, y = float(r["x_m"]), float(r["y_m"])
                        else:
                            x, y = float(r.get("x_px", 0))/10.0, float(r.get("y_px", 0))/10.0
                        if r.get("type") == "ball": ball.append((x,y))
                        else: players.append((x,y,r.get("id")))

                    fig = plt.figure(figsize=(10,6))
                    ax = plt.gca()
                    ax.set_xlim(0, PITCH_W); ax.set_ylim(0, PITCH_H); ax.invert_yaxis(); ax.set_aspect("equal")
                    ax.add_patch(plt.Rectangle((0,0), PITCH_W, PITCH_H, fill=False))
                    ax.axvline(PITCH_W/2, linestyle="--", linewidth=1)
                    if players:
                        xs=[p[0] for p in players]; ys=[p[1] for p in players]
                        ax.scatter(xs, ys, s=40)
                        if show_ids:
                            for (x,y,pid) in players: ax.text(x+0.5, y, f"{pid}", fontsize=8)
                    if show_ball and ball:
                        bx=[b[0] for b in ball]; by=[b[1] for b in ball]
                        ax.scatter(bx, by, s=60, marker="o")
                    canvas.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                    time.sleep(max(0.01, 0.1/spd))
