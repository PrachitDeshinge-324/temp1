import numpy as np
from typing import Dict, List, Tuple
import torch

class PoseDetector:
    def __init__(self, model, device='cpu', confidence=0.45):
        self.model = model
        self.device = device
        self.confidence = confidence
        
        # COCO keypoint skeleton connections
        self.skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                        [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]]
    
    def detect(self, image):
        """Detect poses in an image"""
        results = self.model.predict(image, conf=self.confidence, 
                                     device=self.device, verbose=False)
        keypoints = []
        
        for result in results:
            if result.keypoints is None:
                continue
            kpts = result.keypoints.xy.cpu().numpy()[0]
            keypoints.append(kpts)
            
        return keypoints
    
    def smooth_keypoints(self, history, current):
        """Apply temporal smoothing to keypoints"""
        if not history or len(history) < 2:
            return current
        
        weights = np.linspace(0.5, 1.0, len(history) + 1)
        weights = weights / np.sum(weights)
        all_keypoints = history + [current]
        result = np.zeros_like(current)
        
        for i in range(len(current)):
            valid_pts = []
            for kp in all_keypoints:
                if i < len(kp) and kp[i].size >= 2 and kp[i][0] > 0 and kp[i][1] > 0:
                    valid_pts.append(kp[i])
            
            if valid_pts:
                valid_weights = weights[-len(valid_pts):]
                valid_weights = valid_weights / np.sum(valid_weights)
                result[i] = np.average(valid_pts, axis=0, weights=valid_weights)
            else:
                result[i] = current[i]
        
        return result