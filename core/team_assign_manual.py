import cv2
import numpy as np
import webcolors

class ManualTeamAssigner:
    def __init__(self, team_a_hex, team_b_hex):
        # Convert HEX to LAB (Perceptual Color Space)
        self.color_a = self._hex_to_lab(team_a_hex)
        self.color_b = self._hex_to_lab(team_b_hex)
        self.final_labels = {}

    def _hex_to_lab(self, hex_str):
        # Expand 3-digit hex (e.g., #F00 -> #FF0000) if necessary, though input usually standard
        rgb = webcolors.hex_to_rgb(hex_str)
        # Create 1x1 pixel image to convert color space
        bgr_pixel = np.array([[[rgb.blue, rgb.green, rgb.red]]], dtype=np.uint8)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        return lab_pixel[0][0].astype(np.float32)

    def _torso_patch(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h: return None
        
        bw, bh = x2-x1, y2-y1
        # Strict chest crop
        tx1 = x1 + int(0.35 * bw)
        tx2 = x1 + int(0.65 * bw)
        ty1 = y1 + int(0.20 * bh)
        ty2 = y1 + int(0.50 * bh)
        return frame[ty1:ty2, tx1:tx2]

    def get_team(self, frame, bbox):
        patch = self._torso_patch(frame, *bbox)
        if patch is None or patch.size == 0: return 'unknown'
        
        # Get player's median color in LAB
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        player_color = np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)
        
        # Calculate Distance to User Samples
        dist_a = np.linalg.norm(player_color - self.color_a)
        dist_b = np.linalg.norm(player_color - self.color_b)
        
        # Logic: Assign to closest
        # REJECTION: If too far from BOTH (e.g. > 50 units), it's a Ref/Coach
        REJECTION_THRESHOLD = 40.0 
        
        if dist_a < dist_b:
            return 'A' if dist_a < REJECTION_THRESHOLD else 'unknown'
        else:
            return 'B' if dist_b < REJECTION_THRESHOLD else 'unknown'
