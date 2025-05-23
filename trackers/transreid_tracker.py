"""
TransReID-based person tracking implementation
"""
import numpy as np
from typing import List, Dict, Any, Optional

from utils.trackers import Detection
from utils.transreid_model import load_transreid_model

class TransReIDTracker:
    """TransReID-based person tracker with robust re-identification"""
    def __init__(self, device='cpu', reid_model_path='model/transreid_vitbase.pth', iou_threshold=0.5, conf_threshold=0.5, max_age=30):
        self.device = device
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_age = max_age  # Maximum frames a track can be inactive before deletion
        self.tracks = {}
        self.next_id = 1  # Ensure we start at 1, never 0 or negative
        self.frame_count = 0
        
        # Load TransReID model
        try:
            # Use the TransReIDModel implementation
            self.reid_model = load_transreid_model(reid_model_path, device)
            print(f"TransReID model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading TransReID model: {e}")
            self.reid_model = None
        
        # Feature database for tracking
        self.feature_db = {}  # track_id -> feature vector
    
    def extract_feature(self, image_crop):
        """Extract feature vector from person crop using TransReID"""
        if self.reid_model is None or image_crop.size == 0:
            return None
            
        try:
            # Use the extract_features method from TransReIDModel
            features = self.reid_model.extract_features(image_crop)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return max(0.0, min(iou, 1.0))

    def update(self, frame, detections: List[Detection]) -> List[Detection]:
        """Update tracking with new detections"""
        self.frame_count += 1
        
        # Process empty detection case
        if not detections:
            # Update track ages and remove stale tracks
            tracks_to_remove = []
            for track_id, track_info in self.tracks.items():
                track_info['age'] += 1
                if track_info['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                if track_id in self.feature_db:
                    del self.feature_db[track_id]
                    
            return []
            
        # Extract features for new detections
        detection_features = []
        valid_detections = []
        
        for det in detections:
            # Additional validation to ensure detection is valid
            if not hasattr(det, 'bbox') or det.bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                continue
                
            # Clip coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Get person crop
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            # Extract feature vector
            feature = self.extract_feature(person_crop)
            if feature is not None:
                detection_features.append(feature)
                valid_detections.append(det)
        
        # Match detections with existing tracks based on features and IoU
        if not self.tracks:
            # Initialize tracks if none exist
            for i, (det, feat) in enumerate(zip(valid_detections, detection_features)):
                # Ensure track ID is positive
                new_id = max(1, self.next_id)
                self.tracks[new_id] = {
                    'bbox': det.bbox,
                    'feature': feat,
                    'age': 0
                }
                self.feature_db[new_id] = feat
                det.track_id = new_id
                self.next_id = new_id + 1  # Ensure ID increments properly
        else:
            # Match detections to existing tracks
            unmatched_detections = list(range(len(valid_detections)))
            unmatched_tracks = list(self.tracks.keys())
            
            # Calculate feature similarity for all detection-track pairs
            similarity_matrix = np.zeros((len(valid_detections), len(self.tracks)))
            
            for i, feat in enumerate(detection_features):
                for j, track_id in enumerate(self.tracks):
                    if track_id in self.feature_db:
                        similarity = np.dot(feat, self.feature_db[track_id])
                        similarity_matrix[i, j] = similarity
            
            # Calculate IoU for spatial consistency
            iou_matrix = np.zeros((len(valid_detections), len(self.tracks)))
            
            for i, det in enumerate(valid_detections):
                for j, track_id in enumerate(self.tracks):
                    iou_matrix[i, j] = self.calculate_iou(det.bbox, self.tracks[track_id]['bbox'])
            
            # Combined matching score (weighted combination of feature similarity and IoU)
            combined_matrix = 0.7 * similarity_matrix + 0.3 * iou_matrix
            
            # Match detections to tracks
            matches = []
            
            # While there are possible matches
            while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
                # Find highest combined score
                max_score = 0
                best_match = (-1, -1)
                
                for i in unmatched_detections:
                    for j, track_idx in enumerate(unmatched_tracks):
                        j_idx = list(self.tracks.keys()).index(track_idx)
                        score = combined_matrix[i, j_idx]
                        if score > max_score:
                            max_score = score
                            best_match = (i, track_idx)
                
                # If best match is good enough, add it
                if max_score > 0.5:  # Threshold for match quality
                    i, track_id = best_match
                    matches.append((i, track_id))
                    unmatched_detections.remove(i)
                    unmatched_tracks.remove(track_id)
                else:
                    break
            
            # Update matched tracks
            for det_idx, track_id in matches:
                det = valid_detections[det_idx]
                feat = detection_features[det_idx]
                
                # Ensure track ID is valid (positive)
                if track_id <= 0:
                    track_id = max(1, self.next_id)
                    self.next_id = track_id + 1
                
                # Update track information
                self.tracks[track_id].update({
                    'bbox': det.bbox,
                    'feature': feat,
                    'age': 0
                })
                
                # Update feature database with exponential moving average
                alpha = 0.7  # Weight for existing feature
                self.feature_db[track_id] = alpha * self.feature_db[track_id] + (1-alpha) * feat
                
                # Normalize updated feature - with safety check
                norm = np.linalg.norm(self.feature_db[track_id])
                if norm > 0:  # Avoid division by zero
                    self.feature_db[track_id] = self.feature_db[track_id] / norm
                
                # Set detection track_id
                det.track_id = track_id
            
            # Handle unmatched detections (create new tracks)
            for det_idx in unmatched_detections:
                det = valid_detections[det_idx]
                feat = detection_features[det_idx]
                
                # Ensure track ID is positive
                new_id = max(1, self.next_id)
                self.tracks[new_id] = {
                    'bbox': det.bbox,
                    'feature': feat,
                    'age': 0
                }
                self.feature_db[new_id] = feat
                det.track_id = new_id
                self.next_id = new_id + 1  # Ensure ID increments properly
            
            # Update unmatched tracks (increase age)
            for track_id in unmatched_tracks:
                self.tracks[track_id]['age'] += 1
            
            # Remove stale tracks
            tracks_to_remove = []
            for track_id, track_info in self.tracks.items():
                if track_info['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                if track_id in self.feature_db:
                    del self.feature_db[track_id]
        
        # Final validation - ensure all track IDs are positive
        valid_detections_with_valid_ids = []
        for det in valid_detections:
            if hasattr(det, 'track_id'):
                if det.track_id > 0:  # Only include positive track IDs
                    valid_detections_with_valid_ids.append(det)
                else:
                    # Fix invalid track ID
                    new_id = max(1, self.next_id)
                    det.track_id = new_id
                    self.next_id = new_id + 1
                    valid_detections_with_valid_ids.append(det)
        
        return valid_detections_with_valid_ids
