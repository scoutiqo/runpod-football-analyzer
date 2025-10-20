# server/metrics_core.py
import math, time, json
from collections import defaultdict, deque
import numpy as np
from filterpy.kalman import KalmanFilter

class BallKF:
    def __init__(self):
        self.kf=None

    def _init(self, x, y, dt):
        kf=KalmanFilter(dim_x=4, dim_z=2)
        kf.F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
        kf.H=np.array([[1,0,0,0],[0,1,0,0]],float)
        kf.P*=200; kf.R*=8; kf.Q*=0.02
        kf.x=np.array([[x],[y],[0.0],[0.0]],float)
        self.kf=kf

    def step(self, det_xy, fps=25.0):
        dt=1.0/max(1.0,fps)
        if self.kf is None and det_xy is not None:
            self._init(det_xy[0], det_xy[1], dt)
            return det_xy
        if self.kf is None:
            return det_xy
        self.kf.F[0,2]=self.kf.F[1,3]=dt
        self.kf.predict()
        if det_xy is not None:
            self.kf.update(np.array(det_xy))
        return float(self.kf.x[0]), float(self.kf.x[1])

class MetricsState:
    def __init__(self):
        self.started=time.time()
        self.pos=[0.0,0.0]
        self.owner=None      # (team, sq)
        self.owner_since=time.time()
        self.events=[]
        self.player=defaultdict(lambda: defaultdict(float))
        self.form_hist=deque(maxlen=30)
        self.ballkf=BallKF()

    def _lab(self, team, sq): return ("A" if team==0 else "B")+str(int(sq))

    def update_owner(self, team, sq, dist=None):
        now=time.time()
        if self.owner is None:
            self.owner=(team, sq); self.owner_since=now; return
        if team==self.owner[0]:
            self.pos[team]+=now-self.owner_since; self.owner_since=now
            if sq!=self.owner[1]:
                # same-team change -> pass
                self.player[self._lab(*self.owner)]["passes"]+=1
                self.player[self._lab(*self.owner)]["passes_complete"]+=1
                self.events.append({"t":now,"type":"pass","from":self._lab(*self.owner),"to":self._lab(team,sq)})
                self.owner=(team,sq)
        else:
            # different team -> tackle/turnover
            self.player[self._lab(team,sq)]["tackles"]+=1
            self.events.append({"t":now,"type":"tackle","who":self._lab(team,sq)})
            self.owner=(team,sq); self.owner_since=now

    def touch(self, team, sq, in_final=False):
        k=self._lab(team,sq)
        self.player[k]["touches"]+=1
        if in_final: self.player[k]["final_third_touches"]+=1

    def header(self, team, sq):
        k=self._lab(team,sq); self.player[k]["headers"]+=1
        self.events.append({"t":time.time(),"type":"header","who":k})

    def snapshot(self):
        tot=max(1e-3, sum(self.pos))
        pA=100*self.pos[0]/tot; pB=100*self.pos[1]/tot
        return {
            "uptime": round(time.time()-self.started,1),
            "possession": {"teamA": pA, "teamB": pB},
            "players": self.player,
            "events_tail": self.events[-50:],
            "formation": self._formation()
        }

    def _formation(self):
        if not self.form_hist: return {"A":"?","B":"?"}
        A=np.array([r[0] for r in self.form_hist if len(r)==2 and len(r[0])>0], dtype=object)
        B=np.array([r[1] for r in self.form_hist if len(r)==2 and len(r[1])>0], dtype=object)
        def fmt(arr):
            if arr.size==0: return "?"
            pts=np.vstack(arr)
            qs=np.quantile(pts[:,1], [0.33,0.66])
            lines=[np.sum(pts[:,1]<=qs[0]), np.sum((pts[:,1]>qs[0])&(pts[:,1]<=qs[1])), np.sum(pts[:,1]>qs[1])]
            return f"{int(lines[0])}-{int(lines[1])}-{int(lines[2])}"
        return {"A":fmt(A), "B":fmt(B)}

def nearest_to_ball(ball_xy, tracks):
    if ball_xy is None or not tracks: return None, None
    bx,by=ball_xy; best=None; bestd=1e9
    for tid,(x1,y1,x2,y2) in tracks.items():
        cx=(x1+x2)/2; cy=(y1+y2)/2
        d=(cx-bx)**2+(cy-by)**2
        if d<bestd: bestd=d; best=tid
    return best, math.sqrt(bestd)

def in_final_third(xy, bounds):
    if xy is None: return False
    x1,y1,x2,y2=bounds
    rx=(xy[0]-x1)/max(1,(x2-x1))
    return rx>2/3
