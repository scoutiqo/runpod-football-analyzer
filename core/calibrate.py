import cv2
import numpy as np
from ultralytics import YOLO
import os

# REAL WORLD COORDINATES (Meters) - Origin Top-Left
REAL_WORLD_POINTS = {
    "TL_Corner": [0, 0],
    "TR_Corner": [105, 0],
    "BR_Corner": [105, 68],
    "BL_Corner": [0, 68],
    "Center_Spot": [52.5, 34],
    "Center_Circle_Top": [52.5, 34 - 9.15],
    "Center_Circle_Bottom": [52.5, 34 + 9.15],
    "Penalty_Spot_Left": [11, 34],
    "Penalty_Spot_Right": [105 - 11, 34],
    "Box_TL_Left": [16.5, 13.84],
    "Box_BL_Left": [16.5, 54.16],
    "Box_TR_Right": [105 - 16.5, 13.84],
    "Box_BR_Right": [105 - 16.5, 54.16]
}

KEYPOINT_NAMES = [
    "TL_Corner", "TR_Corner", "BR_Corner", "BL_Corner",
    "Center_Circle_Top", "Center_Circle_Bottom", "Center_Spot",
    "Penalty_Spot_Left", "Penalty_Spot_Right",
    "Box_TL_Left", "Box_BL_Left", "Box_TR_Right", "Box_BR_Right"
]

class DynamicCalibrator:
    def __init__(self, model_path="models/pitch_calibration_v1.pt"):
        self.model = None
        if os.path.exists(model_path):
            print(f"   📐 Loading AI Surveyor: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("   ⚠️ Calibration Model missing. Using Default.")
            
        self.homography = None
        self.default_homography = None

    def get_default_homography(self, w, h):
        if self.default_homography is not None:
            return self.default_homography
            
        # Fallback: Assume standard TV Broadcast view (Trapezoid)
        # Screen Points (Trapezoid)
        src = np.array([
            [w * 0.1, h * 0.2],  # Top Left (Far)
            [w * 0.9, h * 0.2],  # Top Right (Far)
            [w * 0.95, h * 0.95], # Bottom Right (Near)
            [w * 0.05, h * 0.95]  # Bottom Left (Near)
        ], dtype=np.float32)
        
        # World Points (Rectangle 105x68)
        dst = np.array([
            [0, 0],     # TL
            [105, 0],   # TR
            [105, 68],  # BR
            [0, 68]     # BL
        ], dtype=np.float32)
        
        self.default_homography = cv2.getPerspectiveTransform(src, dst)
        return self.default_homography

    def calibrate(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Try AI Calibration
        if self.model:
            try:
                results = self.model(frame, verbose=False)[0]
                if results.keypoints.has_visible:
                    kpts = results.keypoints.xy[0].cpu().numpy()
                    confs = results.keypoints.conf[0].cpu().numpy() if results.keypoints.conf is not None else [1.0]*13
                    
                    src_pts = []
                    dst_pts = []
                    
                    for i, (x, y) in enumerate(kpts):
                        if confs[i] > 0.5: 
                            name = KEYPOINT_NAMES[i]
                            if name in REAL_WORLD_POINTS:
                                src_pts.append([x, y])
                                dst_pts.append(REAL_WORLD_POINTS[name])
                                
                    if len(src_pts) >= 4:
                        self.homography, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts))
                        return self.homography
            except: pass

        # 2. Return Last Known Good Matrix
        if self.homography is not None:
            return self.homography
            
        # 3. Return Default (Safety Net)
        return self.get_default_homography(w, h)

    def transform(self, x, y):
        # Guaranteed to have a matrix (Default or AI)
        H = self.homography if self.homography is not None else self.default_homography
        
        if H is None: return -1.0, -1.0 # Should happen only on init before first frame
        
        pt = np.array([[[x, y]]], dtype=np.float32)
        try:
            dst = cv2.perspectiveTransform(pt, H)
            tx, ty = float(dst[0][0][0]), float(dst[0][0][1])
            # Clamp to pitch dims
            return max(0, min(105, tx)), max(0, min(68, ty))
        except:
            return -1.0, -1.0
