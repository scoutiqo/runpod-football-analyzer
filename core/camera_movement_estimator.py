import cv2
import numpy as np
import os
import sys

class CameraMovementEstimator:
    def __init__(self, frame):
        # Standard Optical Flow Parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7
        )
        
        h, w = frame.shape[:2]
        self.features_mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.features_mask[0:int(h*0.1), :] = 255
        self.features_mask[int(h*0.9):h, :] = 255
        self.features_mask[:, 0:int(w*0.05)] = 255
        self.features_mask[:, int(w*0.95):w] = 255
        
        self.minimum_distance = 5
        
        # State for Streaming
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.old_features = cv2.goodFeaturesToTrack(self.old_gray, mask=self.features_mask, **self.feature_params)
        self.frame_count = 0

    def get_camera_movement(self, frames):
        # Legacy method for small lists
        return [self.process_frame(f) for f in frames]

    def process_frame(self, frame_curr):
        """
        Calculates [x, y] movement relative to the PREVIOUS frame.
        Updates internal state automatically.
        """
        frame_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        
        if self.old_features is None:
            self.old_features = cv2.goodFeaturesToTrack(self.old_gray, mask=self.features_mask, **self.feature_params)
            self.old_gray = frame_gray.copy()
            return [0, 0]

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.old_features, None, **self.lk_params)
        
        movement = [0, 0]
        
        if new_features is not None and status is not None:
            good_new = new_features[status==1]
            good_old = self.old_features[status==1]
            
            max_dist = 0
            cam_x_sum, cam_y_sum = 0, 0
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                new_pt, old_pt = new.ravel(), old.ravel()
                dist = np.linalg.norm(new_pt - old_pt)
                if dist > max_dist: max_dist = dist
                
                cam_x_sum += (old_pt[0] - new_pt[0])
                cam_y_sum += (old_pt[1] - new_pt[1])
            
            if max_dist > self.minimum_distance and len(good_new) > 0:
                movement = [cam_x_sum / len(good_new), cam_y_sum / len(good_new)]
                self.old_features = good_new.reshape(-1, 1, 2)
        
        self.old_gray = frame_gray.copy()
        
        # Refresh features periodically to avoid drift
        if self.frame_count % 30 == 0:
             features = cv2.goodFeaturesToTrack(frame_gray, mask=self.features_mask, **self.feature_params)
             if features is not None: self.old_features = features

        return movement
