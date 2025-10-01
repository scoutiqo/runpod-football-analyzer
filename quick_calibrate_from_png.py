import json, argparse, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

STD_W, STD_H = 105.0, 68.0  # meters

def dlt_homography(src, dst):
    A=[]
    for (x,y),(X,Y) in zip(src,dst):
        A.append([x,y,1,0,0,0,-X*x,-X*y,-X])
        A.append([0,0,0,x,y,1,-Y*x,-Y*y,-Y])
    A=np.asarray(A,float)
    _,_,Vt=np.linalg.svd(A)
    h=Vt[-1,:]/Vt[-1,-1]
    return h.reshape(3,3)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image', required=True)  # runs/frame0.png
    ap.add_argument('--out', default='runs/json/homography.json')
    args=ap.parse_args()

    img=imread(args.image)
    fig,ax=plt.subplots(figsize=(12,7))
    ax.imshow(img)
    ax.set_title('Click 4 pitch corners: TL, TR, BR, BL. Close window when done.')
    pts=plt.ginput(4, timeout=0)
    plt.close(fig)

    if len(pts)!=4: raise SystemExit('Need exactly 4 clicks.')
    src=np.array(pts,float)
    dst=np.array([[0,0],[STD_W,0],[STD_W,STD_H],[0,STD_H]],float)
    H=dlt_homography(src,dst)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump({'H':H.tolist(),'src':src.tolist(),'dst':dst.tolist(),'pitch_m':[STD_W,STD_H]}, f, indent=2)
    print(f'[DONE] wrote {args.out}')
if __name__=='__main__': main()
