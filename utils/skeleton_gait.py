import numpy as np
import pandas as pd
import os
from collections import defaultdict

class GaitFeatureExtractor:
    def __init__(self):
        # Initialize data structures to track people
        self.track_history = {}  # Mapping track_id -> [(frame_idx, keypoints), ...]
        self.features_cache = {}  # Cache for computed features per track_id
        self.invariant_features_cache = {}  # Cache for view-invariant features
        
    def update_track(self, track_id, keypoints, frame_idx):
        """Update tracking history for a specific person ID"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # Store frame index and keypoints
        self.track_history[track_id].append((frame_idx, keypoints))
        
        # Reset features cache for this track as we have new data
        if track_id in self.features_cache:
            del self.features_cache[track_id]
        if track_id in self.invariant_features_cache:
            del self.invariant_features_cache[track_id]
    
    def get_features(self, track_id):
        """Extract gait features from a track's history"""
        # Return cached features if available
        if track_id in self.features_cache:
            return self.features_cache[track_id]
            
        if track_id not in self.track_history:
            return None
            
        history = self.track_history[track_id]
        if len(history) < 10:  # Need minimum frames for stable features
            return None
        
        # Extract features
        features = {}
        
        # Calculate limb lengths
        limb_lengths = self._calculate_limb_lengths(track_id)
        features.update(limb_lengths)
        
        # Calculate joint angles
        joint_angles = self._calculate_joint_angles(track_id)
        features.update(joint_angles)
        
        # Calculate movement features
        movement_features = self._calculate_movement_features(track_id)
        features.update(movement_features)
        
        # Cache computed features
        self.features_cache[track_id] = features
        return features
    
    def _calculate_limb_lengths(self, track_id):
        """Calculate limb length features"""
        features = {}
        history = self.track_history[track_id]
        keypoints_list = [kpts for _, kpts in history]
        
        # Define limb connections (adjust indices based on your keypoint format)
        limbs = {
            "neck_to_right_shoulder": (1, 2),
            "neck_to_left_shoulder": (1, 5),
            "right_shoulder_to_right_elbow": (2, 3),
            "right_elbow_to_right_wrist": (3, 4),
            "left_shoulder_to_left_elbow": (5, 6),
            "left_elbow_to_left_wrist": (6, 7),
            "right_hip_to_right_knee": (8, 9),
            "right_knee_to_right_ankle": (9, 10),
            "left_hip_to_left_knee": (11, 12),
            "left_knee_to_left_ankle": (12, 13)
        }
        
        # Calculate average length for each limb
        for limb_name, (idx1, idx2) in limbs.items():
            lengths = []
            for kpts in keypoints_list:
                if len(kpts) > max(idx1, idx2):
                    p1, p2 = kpts[idx1], kpts[idx2]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        length = np.linalg.norm(p1 - p2)
                        lengths.append(length)
            
            if lengths:
                features[f"avg_{limb_name}_length"] = np.mean(lengths)
                features[f"std_{limb_name}_length"] = np.std(lengths)
        
        return features
    
    def _calculate_joint_angles(self, track_id):
        """Calculate joint angle features"""
        features = {}
        history = self.track_history[track_id]
        keypoints_list = [kpts for _, kpts in history]
        
        # Define joint angles (triplets of keypoint indices)
        joints = {
            "right_elbow": (2, 3, 4),  # shoulder, elbow, wrist
            "left_elbow": (5, 6, 7),
            "right_knee": (8, 9, 10),  # hip, knee, ankle
            "left_knee": (11, 12, 13)
        }
        
        # Calculate average angle for each joint
        for joint_name, (idx1, idx2, idx3) in joints.items():
            angles = []
            for kpts in keypoints_list:
                if len(kpts) > max(idx1, idx2, idx3):
                    p1, p2, p3 = kpts[idx1], kpts[idx2], kpts[idx3]
                    if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
                        angle = self._calculate_angle(p1, p2, p3)
                        if angle is not None:
                            angles.append(angle)
            
            if angles:
                features[f"avg_{joint_name}_angle"] = np.mean(angles)
                features[f"std_{joint_name}_angle"] = np.std(angles)
        
        return features
    
    def _calculate_movement_features(self, track_id):
        """Calculate movement-based features"""
        features = {}
        history = self.track_history[track_id]
        frames, keypoints_list = zip(*history)
        
        # Calculate step frequency and rhythm using ankle keypoints
        ankle_positions = []
        valid_frames = []
        
        for frame_idx, kpts in zip(frames, keypoints_list):
            if len(kpts) > 13:  # Check if ankles exist (adjust index as needed)
                left_ankle = kpts[13]
                right_ankle = kpts[10]
                if not (np.isnan(left_ankle).any() or np.isnan(right_ankle).any()):
                    ankle_positions.append((left_ankle, right_ankle))
                    valid_frames.append(frame_idx)
        
        if len(ankle_positions) > 10:  # Need enough samples for meaningful analysis
            # Calculate stride metrics
            left_strides = []
            right_strides = []
            stride_times = []
            
            for i in range(1, len(ankle_positions)):
                prev_left, prev_right = ankle_positions[i-1]
                curr_left, curr_right = ankle_positions[i]
                time_diff = valid_frames[i] - valid_frames[i-1]
                
                # Distance between current and previous positions
                left_dist = np.linalg.norm(curr_left - prev_left)
                right_dist = np.linalg.norm(curr_right - prev_right)
                
                # Only count significant movements as strides
                if left_dist > 5:  # Threshold for stride detection (adjust as needed)
                    left_strides.append(left_dist)
                    stride_times.append(time_diff)
                if right_dist > 5:
                    right_strides.append(right_dist)
                    stride_times.append(time_diff)
            
            if left_strides and right_strides and stride_times:
                # Basic stride metrics
                features['mean_left_stride_length'] = np.mean(left_strides)
                features['mean_right_stride_length'] = np.mean(right_strides)
                features['std_left_stride_length'] = np.std(left_strides)
                features['std_right_stride_length'] = np.std(right_strides)
                
                # Gait symmetry (ratio of left to right stride)
                features['gait_symmetry'] = np.mean(left_strides) / np.mean(right_strides)
                
                # Average stride speed
                features['mean_stride_speed'] = (np.mean(left_strides) + np.mean(right_strides)) / (2 * np.mean(stride_times))
                
        return features
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints to make them invariant to camera view and distance.
        
        Approaches:
        1. Body-centered coordinate system (origin at mid-hip)
        2. Scale normalization based on torso length
        3. Rotation normalization based on shoulder orientation
        """
        if keypoints is None or len(keypoints) < 15:  # Ensure we have enough keypoints
            return None
            
        normalized_kpts = keypoints.copy()
        
        # 1. Translate to body-centered coordinate system
        # Use mid-hip point as origin (keypoint indices depend on your skeleton format)
        # Assuming mid-hip can be calculated from left hip (11) and right hip (8)
        left_hip_idx, right_hip_idx = 11, 8  # Adjust based on your keypoint format
        if (left_hip_idx < len(keypoints) and right_hip_idx < len(keypoints) and 
                not np.isnan(keypoints[left_hip_idx]).any() and 
                not np.isnan(keypoints[right_hip_idx]).any()):
            mid_hip = (keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2
            for i in range(len(normalized_kpts)):
                if not np.isnan(normalized_kpts[i]).any():
                    normalized_kpts[i] = normalized_kpts[i] - mid_hip
        else:
            # If hip points aren't available, try other reference points
            # For example, use keypoint with index 1 (often neck/nose)
            if 1 < len(keypoints) and not np.isnan(keypoints[1]).any():
                reference_point = keypoints[1]
                for i in range(len(normalized_kpts)):
                    if not np.isnan(normalized_kpts[i]).any():
                        normalized_kpts[i] = normalized_kpts[i] - reference_point
            else:
                return None  # Can't normalize without reference point
        
        # 2. Scale normalization
        # Use distance between neck and mid-hip as reference scale
        neck_idx = 1  # Adjust based on your skeleton format
        if (neck_idx < len(keypoints) and left_hip_idx < len(keypoints) and 
                right_hip_idx < len(keypoints)):
            if (not np.isnan(keypoints[neck_idx]).any() and 
                    not np.isnan(keypoints[left_hip_idx]).any() and 
                    not np.isnan(keypoints[right_hip_idx]).any()):
                mid_hip = (keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2
                torso_length = np.linalg.norm(keypoints[neck_idx] - mid_hip)
                if torso_length > 0:
                    for i in range(len(normalized_kpts)):
                        if not np.isnan(normalized_kpts[i]).any():
                            normalized_kpts[i] = normalized_kpts[i] / torso_length
        
        # 3. Rotation normalization (align shoulders to be horizontal)
        left_shoulder_idx, right_shoulder_idx = 5, 2  # Adjust based on your keypoint format
        if (left_shoulder_idx < len(keypoints) and right_shoulder_idx < len(keypoints) and
                not np.isnan(keypoints[left_shoulder_idx]).any() and 
                not np.isnan(keypoints[right_shoulder_idx]).any()):
            # Vector between shoulders
            shoulder_vector = keypoints[right_shoulder_idx] - keypoints[left_shoulder_idx]
            
            # Calculate rotation angle to make shoulders horizontal
            angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
            
            # Create rotation matrix
            cos_angle, sin_angle = np.cos(-angle), np.sin(-angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])
            
            # Apply rotation to all keypoints
            for i in range(len(normalized_kpts)):
                if not np.isnan(normalized_kpts[i]).any():
                    normalized_kpts[i] = rotation_matrix @ normalized_kpts[i]
        
        return normalized_kpts

    def calculate_view_invariant_features(self, track_id):
        """
        Calculate features that are invariant to camera view.
        """
        # Return cached features if available
        if track_id in self.invariant_features_cache:
            return self.invariant_features_cache[track_id]
            
        if track_id not in self.track_history:
            return None
        
        features = {}
        history = self.track_history[track_id]
        
        if len(history) < 10:  # Require minimum frames for stable features
            return None
        
        # Extract normalized keypoints
        keypoints_list = [kpts for _, kpts in history]
        normalized_kpts_list = [self.normalize_keypoints(kpts) for kpts in keypoints_list]
        
        # Skip if normalization failed
        if any(x is None for x in normalized_kpts_list) or len(normalized_kpts_list) == 0:
            return None
        
        # 1. Joint angles (invariant to translation and scale)
        angles_left_arm = []
        angles_right_arm = []
        angles_left_leg = []
        angles_right_leg = []
        
        for n_kpts in normalized_kpts_list:
            # Left arm angle (shoulder-elbow-wrist)
            if 5 < len(n_kpts) and 6 < len(n_kpts) and 7 < len(n_kpts):
                left_shoulder = n_kpts[5]
                left_elbow = n_kpts[6]
                left_wrist = n_kpts[7]
                if not (np.isnan(left_shoulder).any() or np.isnan(left_elbow).any() or np.isnan(left_wrist).any()):
                    angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
                    if angle is not None:
                        angles_left_arm.append(angle)
                
            # Right arm angle (shoulder-elbow-wrist)
            if 2 < len(n_kpts) and 3 < len(n_kpts) and 4 < len(n_kpts):
                right_shoulder = n_kpts[2]
                right_elbow = n_kpts[3]
                right_wrist = n_kpts[4]
                if not (np.isnan(right_shoulder).any() or np.isnan(right_elbow).any() or np.isnan(right_wrist).any()):
                    angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
                    if angle is not None:
                        angles_right_arm.append(angle)
                    
            # Left leg angle (hip-knee-ankle)
            if 11 < len(n_kpts) and 12 < len(n_kpts) and 13 < len(n_kpts):
                left_hip = n_kpts[11]
                left_knee = n_kpts[12]
                left_ankle = n_kpts[13]
                if not (np.isnan(left_hip).any() or np.isnan(left_knee).any() or np.isnan(left_ankle).any()):
                    angle = self._calculate_angle(left_hip, left_knee, left_ankle)
                    if angle is not None:
                        angles_left_leg.append(angle)
                    
            # Right leg angle (hip-knee-ankle)
            if 8 < len(n_kpts) and 9 < len(n_kpts) and 10 < len(n_kpts):
                right_hip = n_kpts[8]
                right_knee = n_kpts[9]
                right_ankle = n_kpts[10]
                if not (np.isnan(right_hip).any() or np.isnan(right_knee).any() or np.isnan(right_ankle).any()):
                    angle = self._calculate_angle(right_hip, right_knee, right_ankle)
                    if angle is not None:
                        angles_right_leg.append(angle)
        
        # Calculate statistics for joint angles
        if angles_left_arm:
            features['inv_left_arm_angle_mean'] = np.mean(angles_left_arm)
            features['inv_left_arm_angle_std'] = np.std(angles_left_arm)
            
        if angles_right_arm:
            features['inv_right_arm_angle_mean'] = np.mean(angles_right_arm)
            features['inv_right_arm_angle_std'] = np.std(angles_right_arm)
            
        if angles_left_leg:
            features['inv_left_leg_angle_mean'] = np.mean(angles_left_leg)
            features['inv_left_leg_angle_std'] = np.std(angles_left_leg)
            
        if angles_right_leg:
            features['inv_right_leg_angle_mean'] = np.mean(angles_right_leg)
            features['inv_right_leg_angle_std'] = np.std(angles_right_leg)
        
        # 2. Limb length ratios (invariant to scale)
        limb_ratios = self._calculate_limb_ratios(normalized_kpts_list)
        features.update(limb_ratios)
        
        # 3. Temporal features - gait cycle analysis
        temporal_features = self._calculate_temporal_features(normalized_kpts_list)
        features.update(temporal_features)
        
        # 4. Postural features
        postural_features = self._calculate_postural_features(normalized_kpts_list)
        features.update(postural_features)
        
        # Cache and return features
        self.invariant_features_cache[track_id] = features
        return features
    
    def _calculate_limb_ratios(self, normalized_kpts_list):
        """Calculate limb length ratios that are invariant to scale"""
        features = {}
        
        # Define limbs to measure
        limbs = {
            'torso': (1, 8),  # neck to hip
            'upper_arm_right': (2, 3),  # shoulder to elbow
            'lower_arm_right': (3, 4),  # elbow to wrist
            'upper_arm_left': (5, 6),
            'lower_arm_left': (6, 7),
            'upper_leg_right': (8, 9),  # hip to knee
            'lower_leg_right': (9, 10),  # knee to ankle
            'upper_leg_left': (11, 12),
            'lower_leg_left': (12, 13)
        }
        
        # Calculate average lengths
        avg_lengths = defaultdict(list)
        
        for n_kpts in normalized_kpts_list:
            for limb_name, (idx1, idx2) in limbs.items():
                if idx1 < len(n_kpts) and idx2 < len(n_kpts):
                    p1, p2 = n_kpts[idx1], n_kpts[idx2]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        length = np.linalg.norm(p1 - p2)
                        avg_lengths[limb_name].append(length)
        
        # Calculate mean lengths
        mean_lengths = {}
        for limb_name, lengths in avg_lengths.items():
            if lengths:
                mean_lengths[limb_name] = np.mean(lengths)
        
        # Calculate ratios (if we have torso as reference)
        if 'torso' in mean_lengths and mean_lengths['torso'] > 0:
            for limb_name, length in mean_lengths.items():
                if limb_name != 'torso':
                    features[f'inv_ratio_{limb_name}_to_torso'] = length / mean_lengths['torso']
        
        # Arm symmetry (left vs right)
        if 'upper_arm_right' in mean_lengths and 'upper_arm_left' in mean_lengths:
            features['inv_arm_symmetry_upper'] = mean_lengths['upper_arm_left'] / mean_lengths['upper_arm_right']
            
        if 'lower_arm_right' in mean_lengths and 'lower_arm_left' in mean_lengths:
            features['inv_arm_symmetry_lower'] = mean_lengths['lower_arm_left'] / mean_lengths['lower_arm_right']
            
        # Leg symmetry (left vs right)
        if 'upper_leg_right' in mean_lengths and 'upper_leg_left' in mean_lengths:
            features['inv_leg_symmetry_upper'] = mean_lengths['upper_leg_left'] / mean_lengths['upper_leg_right']
            
        if 'lower_leg_right' in mean_lengths and 'lower_leg_left' in mean_lengths:
            features['inv_leg_symmetry_lower'] = mean_lengths['lower_leg_left'] / mean_lengths['lower_leg_right']
        
        return features
    
    def _calculate_temporal_features(self, normalized_kpts_list):
        """Calculate temporal features from normalized keypoints"""
        features = {}
        
        # Need enough frames for temporal analysis
        if len(normalized_kpts_list) < 10:
            return features
        
        # Track ankle positions for gait cycle analysis
        left_ankle_positions = []
        right_ankle_positions = []
        
        for n_kpts in normalized_kpts_list:
            if 13 < len(n_kpts) and 10 < len(n_kpts):
                left_ankle = n_kpts[13]
                right_ankle = n_kpts[10]
                if not (np.isnan(left_ankle).any() or np.isnan(right_ankle).any()):
                    left_ankle_positions.append(left_ankle)
                    right_ankle_positions.append(right_ankle)
        
        # Need enough ankle positions
        if len(left_ankle_positions) < 10 or len(right_ankle_positions) < 10:
            return features
            
        # Calculate stride length (maximum distance between consecutive positions)
        left_strides = [np.linalg.norm(left_ankle_positions[i] - left_ankle_positions[i-1]) 
                      for i in range(1, len(left_ankle_positions))]
        right_strides = [np.linalg.norm(right_ankle_positions[i] - right_ankle_positions[i-1]) 
                       for i in range(1, len(right_ankle_positions))]
        
        if left_strides and right_strides:
            features['inv_mean_stride_length'] = (np.mean(left_strides) + np.mean(right_strides)) / 2
            features['inv_stride_length_std'] = (np.std(left_strides) + np.std(right_strides)) / 2
            features['inv_stride_symmetry'] = np.mean(left_strides) / np.mean(right_strides) if np.mean(right_strides) > 0 else 1.0
        
        # Calculate periodic patterns in ankle movement
        if len(left_ankle_positions) > 20:
            # Use x-coordinate (horizontal movement) for step detection
            left_x = [pos[0] for pos in left_ankle_positions]
            right_x = [pos[0] for pos in right_ankle_positions]
            
            # Simple frequency analysis (detect peaks)
            left_peaks = self._detect_peaks(left_x)
            right_peaks = self._detect_peaks(right_x)
            
            if left_peaks and right_peaks:
                # Average step cycle duration (in frames)
                features['inv_left_step_cycle'] = np.mean(np.diff(left_peaks))
                features['inv_right_step_cycle'] = np.mean(np.diff(right_peaks))
                features['inv_step_cycle_ratio'] = features['inv_left_step_cycle'] / features['inv_right_step_cycle'] if features['inv_right_step_cycle'] > 0 else 1.0
        
        return features
    
    def _calculate_postural_features(self, normalized_kpts_list):
        """Calculate postural features from normalized keypoints"""
        features = {}
        
        # Calculate posture angles
        torso_angles = []  # Angle of torso with vertical
        shoulder_angles = []  # Angle of shoulders with horizontal
        hip_angles = []  # Angle of hips with horizontal
        
        # For torso twist calculation, we need paired angles from the same frame
        paired_angles = []  # Store (shoulder_angle, hip_angle) pairs
        
        for n_kpts in normalized_kpts_list:
            # Torso angle (neck to mid-hip)
            neck_idx, left_hip_idx, right_hip_idx = 1, 11, 8
            if (neck_idx < len(n_kpts) and left_hip_idx < len(n_kpts) and right_hip_idx < len(n_kpts) and
                    not np.isnan(n_kpts[neck_idx]).any() and
                    not np.isnan(n_kpts[left_hip_idx]).any() and
                    not np.isnan(n_kpts[right_hip_idx]).any()):
                mid_hip = (n_kpts[left_hip_idx] + n_kpts[right_hip_idx]) / 2
                vertical = np.array([0.0, 1.0])  # Vertical direction
                torso = n_kpts[neck_idx] - mid_hip
                if np.linalg.norm(torso) > 0:
                    torso = torso / np.linalg.norm(torso)
                    torso_angle = np.arccos(np.clip(np.dot(torso, vertical), -1.0, 1.0))
                    torso_angles.append(torso_angle)
            
            # Track both shoulder and hip angles from the same frame for torso twist
            shoulder_angle = None
            hip_angle = None
            
            # Shoulder angle (left to right shoulder)
            left_shoulder_idx, right_shoulder_idx = 5, 2
            if (left_shoulder_idx < len(n_kpts) and right_shoulder_idx < len(n_kpts) and
                    not np.isnan(n_kpts[left_shoulder_idx]).any() and
                    not np.isnan(n_kpts[right_shoulder_idx]).any()):
                shoulder_vector = n_kpts[right_shoulder_idx] - n_kpts[left_shoulder_idx]
                horizontal = np.array([1.0, 0.0])  # Horizontal direction
                if np.linalg.norm(shoulder_vector) > 0:
                    shoulder_vector = shoulder_vector / np.linalg.norm(shoulder_vector)
                    shoulder_angle = np.arccos(np.clip(np.dot(shoulder_vector, horizontal), -1.0, 1.0))
                    shoulder_angles.append(shoulder_angle)
            
            # Hip angle (left to right hip)
            left_hip_idx, right_hip_idx = 11, 8
            if (left_hip_idx < len(n_kpts) and right_hip_idx < len(n_kpts) and
                    not np.isnan(n_kpts[left_hip_idx]).any() and
                    not np.isnan(n_kpts[right_hip_idx]).any()):
                hip_vector = n_kpts[right_hip_idx] - n_kpts[left_hip_idx]
                horizontal = np.array([1.0, 0.0])  # Horizontal direction
                if np.linalg.norm(hip_vector) > 0:
                    hip_vector = hip_vector / np.linalg.norm(hip_vector)
                    hip_angle = np.arccos(np.clip(np.dot(hip_vector, horizontal), -1.0, 1.0))
                    hip_angles.append(hip_angle)
            
            # If we have both shoulder and hip angles for this frame, store them as a pair
            if shoulder_angle is not None and hip_angle is not None:
                paired_angles.append((shoulder_angle, hip_angle))
        
        # Calculate statistics
        if torso_angles:
            features['inv_torso_angle_mean'] = np.mean(torso_angles)
            features['inv_torso_angle_std'] = np.std(torso_angles)
        
        if shoulder_angles:
            features['inv_shoulder_angle_mean'] = np.mean(shoulder_angles)
            features['inv_shoulder_angle_std'] = np.std(shoulder_angles)
        
        if hip_angles:
            features['inv_hip_angle_mean'] = np.mean(hip_angles)
            features['inv_hip_angle_std'] = np.std(hip_angles)
                
        # Calculate relative angle between shoulders and hips (torso twist)
        # Only using frames where both measurements are available
        if paired_angles:
            shoulder_hip_diffs = [abs(s_angle - h_angle) for s_angle, h_angle in paired_angles]
            features['inv_torso_twist_mean'] = np.mean(shoulder_hip_diffs)
            features['inv_torso_twist_std'] = np.std(shoulder_hip_diffs)
        
        return features
    
    def _detect_peaks(self, signal, min_height=None, min_distance=3):
        """Simple peak detection for gait cycle analysis"""
        if min_height is None:
            min_height = np.mean(signal)
            
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Found a peak
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (in radians)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Calculate angle (in radians)
            dot_product = np.dot(v1, v2)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            return angle
        
        return None

    def get_feature_vector(self, track_id):
        """Get a feature vector (flat numpy array) for a track"""
        features = self.get_features(track_id)
        invariant_features = self.calculate_view_invariant_features(track_id)
        
        # Combine regular and invariant features
        all_features = {}
        if features is not None:
            all_features.update(features)
        if invariant_features is not None:
            all_features.update(invariant_features)
            
        if not all_features:
            return None
            
        # Convert to numpy array
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector
    
    def export_features_csv(self, csv_path):
        """Export features to CSV file"""
        results = []
        
        for track_id in self.track_history:
            features = self.get_features(track_id)
            invariant_features = self.calculate_view_invariant_features(track_id)
            
            if features or invariant_features:
                row = {'track_id': track_id}
                
                # Add regular features
                if features:
                    for k, v in features.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            row[k] = v
                
                # Add invariant features
                if invariant_features:
                    for k, v in invariant_features.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            row[k] = v
                
                results.append(row)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(f"Exported {len(results)} gait feature records with {len(df.columns) - 1} features")