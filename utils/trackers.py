import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    track_id: Optional[int] = None
    
class BaseTracker:
    """Base class for object trackers"""
    def __init__(self):
        pass
        
    def update(self, frame, detections):
        """Update tracker with new detections"""
        raise NotImplementedError

class YOLOTracker:
    """Wrapper for YOLO's built-in tracking"""
    def __init__(self, model, device='cpu', min_confidence=0.45, classes=[0]):
        self.model = model
        self.device = device
        self.min_confidence = min_confidence
        self.classes = classes
        self.id_mapping = {}
        self.next_track_id = 1
        
    def update(self, frame):
        """Track objects in frame using YOLO's built-in tracker"""
        results = self.model.track(frame, persist=True, conf=self.min_confidence, 
                                  classes=self.classes, device=self.device, verbose=False)
        
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()
                for i, (box, conf, yolo_id) in enumerate(zip(boxes, confs, track_ids)):
                    if yolo_id not in self.id_mapping:
                        self.id_mapping[yolo_id] = self.next_track_id
                        self.next_track_id += 1
                    track_id = self.id_mapping[yolo_id]
                    
                    detections.append(Detection(
                        bbox=box,
                        confidence=conf,
                        track_id=track_id
                    ))
        
        return detections

class DeepSORTTracker:
    """DeepSORT tracking implementation"""
    def __init__(self, device='cpu'):
        self.device = device
        self.tracker = self._init_deepsort()
        print("DeepSORT tracker initialized")
        
    def _init_deepsort(self):
        return DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.2,
            nn_budget=100,
            override_track_class=None,
            embedder="torchreid",
            half=True,
            bgr=True,
            embedder_gpu=self.device=='cuda',
            embedder_model_name="osnet_ain_x1_0",
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
    def update(self, frame, detections):
        """Update DeepSORT with new detections"""
        boxes = []
        confidences = []
        
        # print(f"Processing {len(detections)} detections with DeepSORT")
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # DeepSORT requires boxes in format [x, y, width, height]
            w, h = x2 - x1, y2 - y1
            boxes.append([x1, y1, w, h])
            confidences.append(det.confidence)
        
        # Create detection objects in DeepSORT format
        detection_list = []
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            detection_list.append((box, conf))
        
        # Update the tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Left, Top, Right, Bottom format
            
            results.append(Detection(
                bbox=np.array(ltrb),
                confidence=1.0,  # Using fixed confidence for tracked objects
                track_id=track_id
            ))
        
        # print(f"DeepSORT returned {len(results)} tracked objects")
        return results