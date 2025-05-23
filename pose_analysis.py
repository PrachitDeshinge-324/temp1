"""
Pose Analysis Module

Handles pose detection, keypoint processing, and gait feature extraction.
All skeleton and pose analysis functionality consolidated here.
"""

import numpy as np
from collections import defaultdict
from utils.pose_detector import PoseDetector
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator

class PoseAnalyzer:
    """Comprehensive pose analysis and gait feature extraction"""
    
    def __init__(self, model_pose, pose_device='cpu', history_length=5):
        self.pose_detector = PoseDetector(model_pose, device=pose_device)
        self.gait_analyzer = GaitFeatureExtractor()
        self.gait_validator = GaitValidator()
        
        # Keypoints history for temporal smoothing
        self.keypoints_history = defaultdict(list)
        self.history_length = history_length
        
        # Person identification (optional)
        self.person_identifier = None
    
    def process_pose(self, frame, detection, frame_count):
        """Process pose for a single person detection"""
        track_id = detection.track_id
        
        if track_id <= 0:  # Skip invalid track IDs
            return None
            
        # Extract person crop with buffer
        x1, y1, x2, y2 = map(int, detection.bbox)
        buffer_x = int((x2 - x1) * 0.05)
        buffer_y = int((y2 - y1) * 0.05)
        
        x1_buf = max(0, x1 - buffer_x)
        y1_buf = max(0, y1 - buffer_y)
        x2_buf = min(frame.shape[1], x2 + buffer_x)
        y2_buf = min(frame.shape[0], y2 + buffer_y)
        
        person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
        
        if person_crop.size == 0:
            return None
        
        # Detect poses in the crop
        keypoints_list = self.pose_detector.detect(person_crop)
        
        processed_keypoints = []
        for keypoints in keypoints_list:
            # Update keypoints history for smoothing
            if len(self.keypoints_history[track_id]) >= self.history_length:
                self.keypoints_history[track_id].pop(0)
            self.keypoints_history[track_id].append(keypoints.copy())
            
            # Apply temporal smoothing to keypoints
            smoothed_keypoints = self.pose_detector.smooth_keypoints(
                self.keypoints_history[track_id][:-1], keypoints)
            
            # Update gait analyzer with new keypoints
            self.gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
            
            # Perform identification if requested
            identity = None
            confidence = 0.0
            if self.person_identifier and frame_count % 15 == 0:
                feature_vector = self.gait_analyzer.get_feature_vector(track_id)
                if feature_vector is not None:
                    identity, confidence = self.person_identifier.identify_person(feature_vector)
            
            processed_keypoints.append({
                'keypoints': smoothed_keypoints,
                'crop_offset': (x1_buf, y1_buf),
                'identity': identity,
                'confidence': confidence
            })
        
        return processed_keypoints
    
    def get_gait_features(self, track_id):
        """Get extracted gait features for a track"""
        return self.gait_analyzer.get_features(track_id)
    
    def get_feature_vector(self, track_id):
        """Get feature vector for person identification"""
        return self.gait_analyzer.get_feature_vector(track_id)
    
    def export_features_csv(self, csv_path):
        """Export all gait features to CSV"""
        return self.gait_analyzer.export_features_csv(csv_path)
    
    def get_track_history(self):
        """Get complete tracking history"""
        return self.gait_analyzer.track_history
    
    def validate_gait_sequence(self, track_id):
        """Validate if gait sequence is complete and reliable"""
        features = self.get_gait_features(track_id)
        if features is None:
            return False
        return self.gait_validator.validate_sequence(features)
    
    def calculate_view_invariant_features(self, track_id):
        """Calculate view-invariant gait features"""
        return self.gait_analyzer.calculate_view_invariant_features(track_id)
    
    def set_person_identifier(self, identifier):
        """Set person identifier for gait-based recognition"""
        self.person_identifier = identifier
    
    def get_normalized_keypoints(self, track_id):
        """Get normalized keypoints for a track"""
        if track_id in self.gait_analyzer.track_history:
            history = self.gait_analyzer.track_history[track_id]
            normalized_points = []
            for frame_idx, keypoints in history:
                normalized = self.gait_analyzer.normalize_keypoints(keypoints)
                normalized_points.append((frame_idx, normalized))
            return normalized_points
        return None
    
    def get_gait_cycle_info(self, track_id):
        """Get gait cycle information for a track"""
        # This would extract gait cycle timing, step frequency, etc.
        features = self.get_gait_features(track_id)
        if features is None:
            return None
            
        # Extract gait cycle specific metrics
        cycle_info = {
            'step_frequency': features.get('step_frequency', 0),
            'stride_length': features.get('avg_stride_length', 0),
            'gait_symmetry': features.get('gait_symmetry_ratio', 0),
            'stance_phase_ratio': features.get('stance_phase_ratio', 0)
        }
        
        return cycle_info
    
    def get_postural_features(self, track_id):
        """Get postural and body proportion features"""
        features = self.get_gait_features(track_id)
        if features is None:
            return None
            
        postural_info = {
            'body_height_ratio': features.get('avg_body_height', 0),
            'torso_leg_ratio': features.get('avg_torso_to_leg_ratio', 0),
            'shoulder_width': features.get('avg_shoulder_width', 0),
            'hip_width': features.get('avg_hip_width', 0)
        }
        
        return postural_info
