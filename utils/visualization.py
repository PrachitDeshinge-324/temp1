import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple

class Visualizer:
    def __init__(self):
        np.random.seed(42)
        self.track_colors = np.random.randint(0, 255, size=(100, 3)).tolist()
        self.joint_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                             (255, 255, 0), (0, 255, 255)]
        self.skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                        [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]]
        
    def get_color(self, track_id):
        """Get consistent color for a track ID"""
        return self.track_colors[int(track_id) % len(self.track_colors)]
    
    def draw_bbox(self, frame, bbox, track_id, color):
        """Draw bounding box with track ID"""
        x1, y1, x2, y2 = map(int, bbox)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter on frame"""
        cv.putText(frame, f"FPS: {int(fps)}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
    
    def draw_detection(self, frame, detection, buffer_ratio=0.1):
        """Draw bounding box and track ID"""
        x1, y1, x2, y2 = map(int, detection.bbox)
        track_id = detection.track_id
        
        color = self.get_color(track_id)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Calculate buffered box
        w, h = x2 - x1, y2 - y1
        x1_buf = max(0, x1 - int(w * buffer_ratio))
        y1_buf = max(0, y1 - int(h * buffer_ratio))
        x2_buf = min(frame.shape[1], x2 + int(w * buffer_ratio))
        y2_buf = min(frame.shape[0], y2 + int(h * buffer_ratio))
        
        return (x1_buf, y1_buf, x2_buf, y2_buf), color
    
    def draw_identity(self, frame, detection, identity, confidence, color):
        """Draw identity and confidence"""
        x1, y1 = map(int, detection.bbox[:2])
        id_text = f"{identity}: {confidence:.2f}"
        cv.putText(frame, id_text, (x1, y1 - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def draw_keypoints(self, frame, keypoints, offset_x, offset_y, color):
        """Draw skeleton and keypoints"""
        for i, point in enumerate(keypoints):
            px, py = int(point[0]), int(point[1])
            if px > 0 and py > 0:
                joint_color = self.joint_colors[i % len(self.joint_colors)]
                cv.circle(frame, (px + offset_x, py + offset_y), 3, joint_color, -1)
        
        for pair in self.skeleton:
            p1, p2 = pair[0]-1, pair[1]-1
            if p1 >= len(keypoints) or p2 >= len(keypoints):
                continue
            if (keypoints[p1][0] > 0 and keypoints[p1][1] > 0 and 
                keypoints[p2][0] > 0 and keypoints[p2][1] > 0):
                cv.line(frame, 
                        (int(keypoints[p1][0]) + offset_x, int(keypoints[p1][1]) + offset_y), 
                        (int(keypoints[p2][0]) + offset_x, int(keypoints[p2][1]) + offset_y),
                        color, 1)