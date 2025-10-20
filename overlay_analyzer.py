import os, cv2, numpy as np, requests
from urllib.parse import urlparse

FILES_ROOT = os.path.abspath("./files")  # server serves /files from here

def url_to_local(url: str) -> str:
    # ONLY map when it's our /files/* path
    if not url.startswith("/files/"):
        raise ValueError(f"Cannot map non-local url to path: {url}")
    path = urlparse(url).path
    return os.path.join(FILES_ROOT, path[len("/files/"):].replace("/", os.sep))

def local_overlay_path(job_id: str, seg_index: int) -> str:
    out_dir = os.path.join(FILES_ROOT, "jobs", job_id)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"overlay_seg_{seg_index:03}.mp4")

def ensure_local_copy(seg_url: str, job_id: str, seg_index: int) -> str:
    """
    1) If seg_url is /files/* and exists locally -> use it.
    2) Else download seg_url (http/https or /files/* missing) to ./files/jobs/<job>/seg_XXX_dl.mp4
    """
    if seg_url.startswith("/files/"):
        candidate = url_to_local(seg_url)
        if os.path.exists(candidate):
            return candidate
    # Download fallback
    out_dir = os.path.join(FILES_ROOT, "jobs", job_id)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"seg_{seg_index:03}_dl.mp4")
    src = seg_url if seg_url.startswith("http") else f"http://127.0.0.1:8081{seg_url}"
    with requests.get(src, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1<<20):
                if chunk: f.write(chunk)
    return tmp

def _open_cap_relaxed(path: str):
    """Open a video path; if OpenCV fails, remux with ffmpeg, then transcode if needed."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, path
    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    base, _ = os.path.splitext(path)
    remux = base + "_remux.mp4"
    os.system(f'"{ffmpeg}" -y -loglevel error -i "{path}" -c copy "{remux}"')
    cap = cv2.VideoCapture(remux)
    if cap.isOpened():
        return cap, remux
    trans = base + "_x264.mp4"
    os.system(f'"{ffmpeg}" -y -loglevel error -i "{path}" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -an "{trans}"')
    cap = cv2.VideoCapture(trans)
    if cap.isOpened():
        return cap, trans
    raise RuntimeError(f"Cannot open video even after remux/transcode: {path}")

class CentroidTracker:
    def __init__(self, max_dist=60, max_age=20):
        self.next_id = 1
        self.tracks = {}   # id -> {"centroid":(x,y), "age":0, "trail":[(x,y),...]}
        self.max_dist = max_dist
        self.max_age  = max_age
    def update(self, boxes):
        centers = [ (x+w//2, y+h//2) for (x,y,w,h) in boxes ]
        used=set()
        for tid,t in list(self.tracks.items()):
            besti=-1; bestd=1e9; bestc=None
            for i,c in enumerate(centers):
                if i in used: continue
                d=(t["centroid"][0]-c[0])**2+(t["centroid"][1]-c[1])**2
                if d<bestd: bestd, besti, bestc = d, i, c
            if bestc is not None and bestd**0.5<=self.max_dist:
                used.add(besti); t["centroid"]=bestc; t["age"]=0
                t["trail"].append(bestc)
                if len(t["trail"])>25: t["trail"].pop(0)
            else:
                t["age"]+=1
                if t["age"]>self.max_age: del self.tracks[tid]
        for i,c in enumerate(centers):
            if i not in used:
                self.tracks[self.next_id]={"centroid":c,"age":0,"trail":[c]}
                self.next_id+=1
        id_map={}
        for (x,y,w,h) in boxes:
            c=(x+w//2,y+h//2); bestid=None; bestd=1e9
            for tid,t in self.tracks.items():
                d=(t["centroid"][0]-c[0])**2+(t["centroid"][1]-c[1])**2
                if d<bestd: bestd, bestid = d, tid
            id_map[bestid]=(x,y,w,h)
        return id_map

def draw_overlay(src_path: str, dst_path: str):
    cap, used_path = _open_cap_relaxed(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w,h))
    if not out.isOpened(): raise RuntimeError(f"Cannot open writer {dst_path}")

    mog=cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=30, detectShadows=False)
    tracker=CentroidTracker(max_dist=max(w,h)//20, max_age=10)
    last_centers=[]; idx=0

    while True:
        ok,frame=cap.read()
        if not ok: break
        mask=mog.apply(frame)
        mask=cv2.medianBlur(mask,5)
        mask=cv2.threshold(mask,200,255,cv2.THRESH_BINARY)[1]
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), iterations=1)
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)

        cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in cnts:
            x,y,wc,hc=cv2.boundingRect(c); area=wc*hc
            if area<150 or area>(w*h*0.15): continue
            if hc<12 or wc<6 or hc/wc<1.1: continue
            boxes.append((x,y,wc,hc))

        id_map=tracker.update(boxes)

        ball=None; best=0
        for (x,y,wc,hc) in boxes:
            c=(x+wc//2,y+hc//2)
            spd=max( ((px-c[0])**2+(py-c[1])**2)**0.5 for (px,py) in last_centers ) if last_centers else 0
            score=(max(1, 60 - (wc*hc)**0.5)) + spd
            if score>best: best, ball = score, c
        last_centers=[ (x+w//2,y+h//2) for (x,y,w,h) in boxes ]

        overlay=frame.copy()
        for tid,(x,y,wc,hc) in id_map.items():
            cv2.rectangle(overlay,(x,y),(x+wc,y+hc),(0,255,0),2)
            cv2.putText(overlay,f"ID {tid}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
            trail = tracker.tracks.get(tid,{}).get("trail",[])
            for i in range(1,len(trail)):
                a=trail[i-1]; b=trail[i]
                cv2.line(overlay,a,b,(0,200,255),2)
        if ball is not None:
            cv2.circle(overlay, ball, 7, (0,80,255), -1)
            cv2.putText(overlay,"BALL?",(ball[0]+8,ball[1]-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,150,255),1,cv2.LINE_AA)

        cv2.putText(overlay,f"Frame {idx}",(12,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1,cv2.LINE_AA)
        out.write(overlay); idx+=1

    out.release(); cap.release()

def make_overlay_for_segment(seg_url: str, job_id: str, seg_index: int) -> str:
    src_local = ensure_local_copy(seg_url, job_id, seg_index)
    dst_local = local_overlay_path(job_id, seg_index)
    draw_overlay(src_local, dst_local)
    rel = os.path.relpath(dst_local, FILES_ROOT).replace(os.sep, "/")
    return f"/files/{rel}"
