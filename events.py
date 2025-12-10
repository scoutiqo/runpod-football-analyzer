# events.py
import numpy as np

def infer_possession(tracks, ball_track, t, window=0.2):
    """Nearest player to ball within SHORT window -> team possession."""
    # tracks: list of {id, team, x_m, y_m, t}
    bp = [b for b in ball_track if abs(b["t"] - t) <= window]
    if not bp:
        return None
    bx,by = bp[len(bp)//2]["x_m"], bp[len(bp)//2]["y_m"]
    cand = [(np.hypot(p["x_m"]-bx, p["y_m"]-by), p) for p in tracks if "x_m" in p]
    if not cand: return None
    _, nearest = min(cand, key=lambda z: z[0])
    return nearest.get("team")

def detect_passes(ball_series, players_by_t, dist_thresh=3.0, gap_s=0.5):
    """
    ball_series: [{t, x_m, y_m, vx, vy}]
    players_by_t: dict[t] -> [player states]
    returns: list of events {type:'pass', t_start, t_end, from_id, to_id, length_m}
    """
    events=[]
    # Simple heuristic: ball accelerates away from A, then decelerates near B
    # TODO refine with velocity dot products + ownership change
    return events

def cut_clip(video_path, t_center, out_path, pre=4.0, post=4.0):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    t1 = max(0, t_center - pre)
    t2 = min(clip.duration, t_center + post)
    clip.subclip(t1, t2).write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)
