import cv2
import numpy as np
import json

class ViewTransformer:
    def __init__(self, config=None):
        # Target: Standard Football Pitch (105m x 68m)
        self.target_vertices = np.array([
            [0, 68],   # Top-Left (0,0 in visual usually bottom-left in math, let's standard top-left logic)
            [105, 68], # Top-Right
            [105, 0],  # Bottom-Right
            [0, 0]     # Bottom-Left
        ], dtype=np.float32)
        
        self.pixel_vertices = None
        self.perspective_transform = None
        self.config = config

    def fit(self, w, h):
        """
        Calculates the Homography Matrix based on the input config.
        """
        # 1. If config is a list of points (Manual Mask from Frontend)
        if self.config and isinstance(self.config, str) and '[' in self.config:
            try:
                # Parse string representation of list
                points = json.loads(self.config)
                if len(points) == 4:
                    self.pixel_vertices = np.array(points, dtype=np.float32)
            except:
                pass
        
        # 2. If config is 'wide' or invalid, use Default Broadcast Approximation
        if self.pixel_vertices is None:
            print("   ⚠️ No specific mask found. Using 'Broadcast View' approximation.")
            # Approximate trapezoid for a standard center-field camera
            # Top: wider, Bottom: narrower (perspective effect inverse)
            # Actually in image: Top edge is far (narrower), Bottom edge is near (wider)
            
            # Image Coords: (0,0) is Top-Left.
            # We assume the pitch fills the frame roughly like a TV broadcast
            
            # Top-Left (Corner flag far left)
            tl = [w * 0.1, h * 0.2]
            # Top-Right (Corner flag far right)
            tr = [w * 0.9, h * 0.2]
            # Bottom-Right (Corner flag near right)
            br = [w * 0.95, h * 0.95]
            # Bottom-Left (Corner flag near left)
            bl = [w * 0.05, h * 0.95]
            
            self.pixel_vertices = np.array([tl, tr, br, bl], dtype=np.float32)

        # Calculate Matrix
        self.perspective_transform = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, x, y):
        """
        Converts Pixel (x,y) -> Meter (x,y)
        """
        if self.perspective_transform is None: 
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.perspective_transform)
        
        tx = transformed[0][0][0]
        ty = transformed[0][0][1]
        
        # Clamp to pitch dimensions to avoid wild errors
        # tx = max(0, min(105, tx))
        # ty = max(0, min(68, ty))
        
        return tx, ty
