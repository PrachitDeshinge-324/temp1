# filepath: /Users/prachit/self/Working/Person_Temp/utils/tracker.py

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class Detection:
    """Class to hold detection information"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    features: Optional[torch.Tensor] = None


@dataclass
class Track:
    """Class to represent a tracked object"""
    track_id: int
    bbox: np.ndarray
    features: Optional[torch.Tensor]
    kalman_filter: Any  # 'KalmanFilter' instance
    hits: int = 1
    time_since_update: int = 0
    state: str = 'tentative'  # 'tentative', 'confirmed', 'deleted'
    age: int = 1
    confidence: float = 1.0  # Track confidence score
    velocity: np.ndarray = None  # Track velocity
    feature_history: List = None  # History of features for robust matching
    
    def __post_init__(self):
        """Initialize optional attributes"""
        if self.velocity is None:
            self.velocity = np.zeros(4)  # [vx, vy, vw, vh]
        if self.feature_history is None:
            self.feature_history = []
            if self.features is not None:
                self.feature_history.append(self.features)
    
    def __eq__(self, other):
        """Equality based on track_id only"""
        if isinstance(other, Track):
            return self.track_id == other.track_id
        return False
    
    def __hash__(self):
        """Hash based on track_id"""
        return hash(self.track_id)
    
    def update_features(self, new_features, max_history=5):
        """Update feature representation with new detection features"""
        if new_features is None:
            return
            
        # Add new features to history
        self.feature_history.append(new_features)
        if len(self.feature_history) > max_history:
            self.feature_history.pop(0)
            
        # Update the main feature vector using a weighted average
        # Give more weight to recent features
        weights = np.linspace(0.5, 1.0, len(self.feature_history))
        weights = weights / weights.sum()
        
        weighted_features = torch.zeros_like(new_features)
        for i, feat in enumerate(self.feature_history):
            weighted_features += weights[i] * feat
            
        self.features = weighted_features


class KalmanFilter:
    """
    A Kalman filter for tracking bounding boxes in image space.
    
    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains
    the bounding box center position (x, y), width w, height h,
    and their respective velocities.
    """
    
    def __init__(self):
        ndim, dt = 4, 1.0
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. Tuned for better stability and reduced ID switches
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """Create track from unassociated measurement.
        
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y) is
            the center position, w the width, and h the height of the bounding
            box.
            
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements,
                       only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance matrix over the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in format
            (x, y, w, h) where (x, y) is the bounding box center position,
            w the width, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
            
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('Invalid distance metric')


def bbox_to_xywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [center_x, center_y, width, height]"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    return np.array([center_x, center_y, w, h])


def xywh_to_bbox(xywh):
    """Convert [center_x, center_y, width, height] to [x1, y1, x2, y2]"""
    center_x, center_y, w, h = xywh
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2
    return np.array([x1, y1, x2, y2])


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


class ByteTracker:
    """
    Enhanced ByteTracker implementation designed to handle crowded scenes
    with improved appearance matching and motion consistency.
    """
    
    def __init__(self, 
                 max_disappeared=60,
                 max_age=80,
                 min_hits=3,  # Restore to 3 for faster confirmation
                 iou_threshold=0.5,  # Legacy parameter
                 low_iou_threshold=0.25,
                 high_score_threshold=0.6,
                 feature_threshold=0.6,  # Relaxed for ID reuse
                 appearance_weight=0.7,  # Weight for appearance in matching
                 motion_weight=0.3,      # Weight for motion in matching
                 feature_history_size=5, # Number of feature samples to keep
                 **kwargs):              # For backward compatibility
        """
        Initialize ByteTracker with parameters optimized for crowded scenes
        """
        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.high_score_threshold = high_score_threshold
        self.feature_threshold = feature_threshold
        
        # Matching weights
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.feature_history_size = feature_history_size
        
        # Track management
        self.next_id = 1
        self.tracks = []
        self.kalman_filter = KalmanFilter()
        self.frame_count = 0
        
        # Enhanced track buffering for recovery
        self.deleted_tracks = {}  # Track ID -> Track mapping
        self.recently_deleted = []  # List of recently deleted tracks (for quick recovery)
        self.track_buffer = {}  # Buffer for temporarily lost tracks
        self.reid_buffer_size = 100  # Number of frames to keep track in ReID buffer
    
    def _get_velocity(self, prev_box, curr_box):
        """Calculate velocity between two bounding boxes"""
        prev_center_x = (prev_box[0] + prev_box[2]) / 2
        prev_center_y = (prev_box[1] + prev_box[3]) / 2
        prev_width = prev_box[2] - prev_box[0]
        prev_height = prev_box[3] - prev_box[1]
        
        curr_center_x = (curr_box[0] + curr_box[2]) / 2
        curr_center_y = (curr_box[1] + curr_box[3]) / 2
        curr_width = curr_box[2] - curr_box[0]
        curr_height = curr_box[3] - curr_box[1]
        
        velocity = np.array([
            curr_center_x - prev_center_x,
            curr_center_y - prev_center_y,
            curr_width - prev_width,
            curr_height - prev_height
        ])
        
        return velocity
    
    def _compute_appearance_cost_matrix(self, detections, tracks):
        """Compute appearance similarity cost matrix between detections and tracks"""
        cost_matrix = np.ones((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            if det.features is None:
                continue
            det_feat = det.features / (det.features.norm() + 1e-6)
            for j, track in enumerate(tracks):
                if track.features is None:
                    continue
                track_feat = track.features / (track.features.norm() + 1e-6)
                # Allow appearance match regardless of IoU, but penalize low IoU
                iou = iou_batch(det.bbox, track.bbox)[0, 0]
                penalty = 0.2 if iou < 0.1 else 0.0
                similarity = torch.cosine_similarity(
                    det_feat.unsqueeze(0),
                    track_feat.unsqueeze(0)
                ).item()
                cost_matrix[i, j] = 1.0 - similarity + penalty
        
        return cost_matrix
    
    def _compute_motion_cost_matrix(self, det_boxes, tracks):
        """Compute motion consistency cost matrix using Kalman predictions and IoU"""
        cost_matrix = np.zeros((len(det_boxes), len(tracks)))
        
        for j, track in enumerate(tracks):
            if track.time_since_update > 0:
                # For unmatched tracks, use Kalman prediction
                pred_xywh = track.kalman_filter.mean[:4]
                pred_bbox = xywh_to_bbox(pred_xywh)
                track_boxes = np.array([pred_bbox])
            else:
                # For matched tracks, use the current bbox
                track_boxes = np.array([track.bbox])
            
            # Calculate IoU-based cost
            iou_matrix = iou_batch(np.array(det_boxes), track_boxes)
            
            # Convert IoU to cost (1 - IoU)
            for i in range(len(det_boxes)):
                cost_matrix[i, j] = 1.0 - iou_matrix[i, 0]
        
        return cost_matrix
    
    def _fuse_motion(self, det_boxes, tracks, appearance_cost=None):
        """Fuse appearance and motion information for matching"""
        if appearance_cost is None:
            # If no appearance cost provided, just use motion
            return self._compute_motion_cost_matrix(det_boxes, tracks)
        
        motion_cost = self._compute_motion_cost_matrix(det_boxes, tracks)
        
        # Combine costs with weights
        combined_cost = (
            self.appearance_weight * appearance_cost + 
            self.motion_weight * motion_cost
        )
        
        return combined_cost
    
    def _cascaded_matching(self, high_dets, tracks):
        """
        Perform cascaded matching using appearance and motion cues,
        optimized for crowded scenes
        """
        if not tracks or not high_dets:
            return [], list(range(len(high_dets))), list(range(len(tracks)))
        
        # Get detection bounding boxes and features
        det_boxes = [det.bbox for det in high_dets]
        
        # Step 1: Appearance matching (if features available)
        appearance_cost = self._compute_appearance_cost_matrix(high_dets, tracks)
        
        # Step 2: Fuse with motion information
        fused_cost = self._fuse_motion(det_boxes, tracks, appearance_cost)
        
        # Apply high IoU gating (limit max cost)
        motion_cost = self._compute_motion_cost_matrix(det_boxes, tracks)
        for i in range(len(det_boxes)):
            for j in range(len(tracks)):
                # If IoU is too low, set cost to infinity
                if motion_cost[i, j] > (1.0 - self.high_iou_threshold):
                    fused_cost[i, j] = float('inf')
        
        # Step 3: Perform assignment
        if fused_cost.size > 0:
            # Check if all values are infinite
            if np.isinf(fused_cost).all():
                # All costs are infinite, no valid matches possible
                return [], list(range(len(high_dets))), list(range(len(tracks)))
            
            try:
                # Make sure there are finite values in the cost matrix
                det_indices, trk_indices = linear_sum_assignment(fused_cost)
                
                # Filter matches that exceed cost threshold
                valid_matches = []
                for d, t in zip(det_indices, trk_indices):
                    if not np.isinf(fused_cost[d, t]):
                        valid_matches.append((d, t))
                
                if valid_matches:
                    det_indices, trk_indices = zip(*valid_matches)
                else:
                    det_indices, trk_indices = [], []
            except ValueError:
                # Handle 'cost matrix is infeasible' error gracefully
                print("Warning: Cost matrix is infeasible. Skipping this matching step.")
                det_indices, trk_indices = [], []
        else:
            det_indices, trk_indices = [], []
        
        # Get unmatched detections and tracks
        unmatched_dets = [d for d in range(len(high_dets)) if d not in det_indices]
        unmatched_tracks = [t for t in range(len(tracks)) if t not in trk_indices]
        
        matches = list(zip(det_indices, trk_indices))
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _second_stage_matching(self, low_dets, unmatched_tracks):
        """Match low confidence detections with remaining tracks"""
        if not unmatched_tracks or not low_dets:
            return [], list(range(len(low_dets)))
        
        # Get track objects for unmatched tracks
        unmatch_track_objects = [self.tracks[i] for i in unmatched_tracks]
        
        # Get detection bounding boxes
        det_boxes = [det.bbox for det in low_dets]
        
        # Compute IoU-based cost matrix
        cost_matrix = self._compute_motion_cost_matrix(det_boxes, unmatch_track_objects)
        
        # Apply low IoU gating
        for i in range(len(det_boxes)):
            for j in range(len(unmatch_track_objects)):
                if cost_matrix[i, j] > (1.0 - self.low_iou_threshold):
                    cost_matrix[i, j] = float('inf')
        
        # Perform assignment
        if cost_matrix.size > 0:
            # Check if all costs are infinite
            if np.isinf(cost_matrix).all():
                return [], list(range(len(low_dets)))
                
            try:
                det_indices, trk_indices = linear_sum_assignment(cost_matrix)
                
                # Filter matches that exceed cost threshold
                valid_matches = []
                for d, t in zip(det_indices, trk_indices):
                    if not np.isinf(cost_matrix[d, t]):
                        valid_matches.append((d, t))
                
                if valid_matches:
                    det_indices, trk_indices = zip(*valid_matches)
                else:
                    det_indices, trk_indices = [], []
            except ValueError:
                # Handle assignment error gracefully
                print("Warning: Second-stage cost matrix is infeasible. Skipping.")
                det_indices, trk_indices = [], []
        else:
            det_indices, trk_indices = [], []
        
        # Convert track indices back to original track indices
        matches = [(d, unmatched_tracks[t]) for d, t in zip(det_indices, trk_indices)]
        
        # Get unmatched detections
        unmatched_dets = [d for d in range(len(low_dets)) if d not in det_indices]
        
        return matches, unmatched_dets
    
    def _recover_lost_tracks(self, detections):
        """
        Try to recover recently lost tracks using appearance similarity
        This helps with occlusion handling in crowded scenes
        """
        if not self.recently_deleted or not detections:
            return []
        
        recovered_matches = []
        
        # Consider only high-confidence detections for track recovery
        high_dets = [det for det in detections if det.confidence >= self.high_score_threshold]
        if not high_dets:
            return []
        
        # For each deleted track, try to find a match
        for track in self.recently_deleted[:]:  # Use a copy to allow removal during iteration
            if track.features is None or track.time_since_update > 30:  # Only consider recent deletions
                continue
                
            best_match = -1
            best_similarity = self.feature_threshold
            
            for i, det in enumerate(high_dets):
                if det.features is None:
                    continue
                
                similarity = torch.cosine_similarity(
                    det.features.unsqueeze(0),
                    track.features.unsqueeze(0)
                ).item()
                
                # Also check motion consistency
                is_consistent = True
                if track.velocity is not None:
                    pred_center_x = track.bbox[0] + (track.bbox[2] - track.bbox[0]) / 2 + track.velocity[0]
                    pred_center_y = track.bbox[1] + (track.bbox[3] - track.bbox[1]) / 2 + track.velocity[1]
                    
                    det_center_x = (det.bbox[0] + det.bbox[2]) / 2
                    det_center_y = (det.bbox[1] + det.bbox[3]) / 2
                    
                    dist = np.sqrt((pred_center_x - det_center_x)**2 + (pred_center_y - det_center_y)**2)
                    
                    # If prediction is too far, consider motion inconsistent
                    if dist > (track.bbox[2] - track.bbox[0]) * 2:  # threshold relative to object size
                        is_consistent = False
                
                if similarity > best_similarity and is_consistent:
                    best_match = i
                    best_similarity = similarity
            
            if best_match >= 0:
                # We've found a match for this deleted track
                recovered_matches.append((track.track_id, best_match))
                self.recently_deleted.remove(track)
        
        return recovered_matches
    
    def update(self, detections: List[Detection]):
        """
        Enhanced update method for crowded scenes with better ID consistency
        """
        self.frame_count += 1
        results = []
        if not detections:
            self._update_unmatched_tracks()
            self._delete_old_tracks()
            for track in self.tracks:
                if track.state == 'confirmed':
                    results.append((track.track_id, track.bbox))
            return results
        self._predict()
        high_dets = []
        low_dets = []
        high_to_all_idx = {}
        low_to_all_idx = {}
        for i, det in enumerate(detections):
            if det.confidence >= self.high_score_threshold:
                high_to_all_idx[len(high_dets)] = i
                high_dets.append(det)
            else:
                low_to_all_idx[len(low_dets)] = i
                low_dets.append(det)
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        matches_a = []
        unmatched_dets_a = list(range(len(high_dets)))
        unmatched_tracks_a = list(range(len(self.tracks)))
        if high_dets and self.tracks:
            confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.state == 'confirmed']
            unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.state == 'tentative']
            if confirmed_tracks:
                try:
                    matches_a, unmatched_dets_a, unmatched_tracks_a = self._cascaded_matching(
                        high_dets, [self.tracks[i] for i in confirmed_tracks])
                    matches.extend([(high_to_all_idx[d], confirmed_tracks[t]) for d, t in matches_a])
                    unmatched_tracks = [t for t in unmatched_tracks if t in confirmed_tracks and confirmed_tracks.index(t) in unmatched_tracks_a or t not in confirmed_tracks]
                except Exception as e:
                    print(f"Warning: Error in first stage matching: {e}")
                    matches_a = []
                    unmatched_dets_a = list(range(len(high_dets)))
                    unmatched_tracks_a = list(range(len(confirmed_tracks)))  # Fix: Return indices, not track objects
            if unconfirmed_tracks and unmatched_dets_a:
                try:
                    # Only use remaining unmatched detections
                    remaining_high_dets = [high_dets[d] for d in unmatched_dets_a]
                    remaining_high_to_all = {i: high_to_all_idx[unmatched_dets_a[i]] for i in range(len(unmatched_dets_a))}
                    if remaining_high_dets:
                        matches_b, _, _ = self._cascaded_matching(
                            remaining_high_dets, [self.tracks[i] for i in unconfirmed_tracks])
                        matches.extend([(remaining_high_to_all[d], unconfirmed_tracks[t]) for d, t in matches_b])
                        matched_unconfirmed = [unconfirmed_tracks[t] for _, t in matches_b]
                        unmatched_tracks = [t for t in unmatched_tracks if t not in matched_unconfirmed]
                except Exception as e:
                    print(f"Warning: Error in unconfirmed track matching: {e}")
            if low_dets and unmatched_tracks:
                try:
                    matches_c, _ = self._second_stage_matching(low_dets, unmatched_tracks)
                    matches.extend([(low_to_all_idx[d], t) for d, t in matches_c])
                    matched_low = [t for _, t in matches_c]
                    unmatched_tracks = [t for t in unmatched_tracks if t not in matched_low]
                except Exception as e:
                    print(f"Warning: Error in second stage matching: {e}")
        for det_idx, track_idx in matches:
            self._update_matched_track(self.tracks[track_idx], detections[det_idx])
        recovered_matches = self._recover_lost_tracks(detections)
        for track_id, det_idx in recovered_matches:
            self._recreate_track(detections[det_idx], track_id)
        for i, det in enumerate(high_dets):
            if not any(high_to_all_idx.get(d) == i for d, _ in matches):
                self._create_new_track(det)
        self._update_unmatched_tracks()
        self._delete_old_tracks()
        for track in self.tracks:
            if track.state == 'confirmed':
                results.append((track.track_id, track.bbox))
        return results
    
    def _predict(self):
        """Predict next state for all tracks using Kalman filter"""
        for track in self.tracks:
            mean, cov = self.kalman_filter.predict(track.kalman_filter.mean, track.kalman_filter.covariance)
            track.kalman_filter.mean = mean
            track.kalman_filter.covariance = cov
            track.age += 1
            track.time_since_update += 1
    
    def _update_matched_track(self, track, detection):
        """Update track with new detection"""
        # Calculate velocity from previous position
        prev_bbox = track.bbox.copy()
        
        # Update Kalman filter
        bbox_xywh = bbox_to_xywh(detection.bbox)
        mean, cov = self.kalman_filter.update(
            track.kalman_filter.mean, track.kalman_filter.covariance, bbox_xywh)
        track.kalman_filter.mean = mean
        track.kalman_filter.covariance = cov
        
        # Update bounding box
        track.bbox = detection.bbox
        
        # Update velocity
        track.velocity = self._get_velocity(prev_bbox, detection.bbox)
        
        # Update features with history
        track.update_features(detection.features, self.feature_history_size)
        
        # Update confidence and state
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0
        
        # Confirm track if it has enough hits
        if track.state == 'tentative' and track.hits >= self.min_hits:
            track.state = 'confirmed'
    
    def _update_unmatched_tracks(self):
        """Update unmatched tracks"""
        for track in self.tracks:
            if track.time_since_update > 0:
                # Use Kalman prediction for unmatched tracks
                track.bbox = xywh_to_bbox(track.kalman_filter.mean[:4])
                
                # Penalize tracks that have been unmatched for several frames
                if track.time_since_update > 3:
                    track.confidence *= 0.7
                else:
                    track.confidence *= 0.95
    
    def _create_new_track(self, detection):
        """Create a new track from a detection"""
        bbox_xywh = bbox_to_xywh(detection.bbox)
        mean, cov = self.kalman_filter.initiate(bbox_xywh)
        
        # Try to match with recently deleted tracks for better ID consistency
        matched_id = None
        best_similarity = self.feature_threshold
        
        if detection.features is not None:
            # First check recently deleted tracks (faster recovery)
            for track in self.recently_deleted:
                if track.features is None:
                    continue
                
                similarity = torch.cosine_similarity(
                    detection.features.unsqueeze(0),
                    track.features.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    matched_id = track.track_id
                    best_similarity = similarity
            
            # Then check older deleted tracks
            if matched_id is None:
                for track_id, track in list(self.deleted_tracks.items()):
                    if track.features is None:
                        continue
                    
                    similarity = torch.cosine_similarity(
                        detection.features.unsqueeze(0),
                        track.features.unsqueeze(0)
                    ).item()
                    
                    if similarity > best_similarity:
                        matched_id = track_id
                        best_similarity = similarity
        
        # Create new track or reuse ID
        track_id = matched_id if matched_id is not None else self.next_id
        
        track = Track(
            track_id=track_id,
            bbox=detection.bbox,
            features=detection.features,
            kalman_filter=type('KF', (), {
                'mean': mean,
                'covariance': cov
            })(),
            state='tentative',
            confidence=detection.confidence
        )
        
        self.tracks.append(track)
        
        if matched_id is None:
            self.next_id += 1
    
    def _recreate_track(self, detection, track_id):
        """Re-create a track with a specific ID (for recovery)"""
        bbox_xywh = bbox_to_xywh(detection.bbox)
        mean, cov = self.kalman_filter.initiate(bbox_xywh)
        
        track = Track(
            track_id=track_id,
            bbox=detection.bbox,
            features=detection.features,
            kalman_filter=type('KF', (), {
                'mean': mean,
                'covariance': cov
            })(),
            # Start with higher hit count to confirm faster
            hits=self.min_hits,
            time_since_update=0,
            state='confirmed',
            confidence=detection.confidence
        )
        
        self.tracks.append(track)
    
    def _delete_old_tracks(self):
        """
        Delete tracks that are too old or unmatched for too long
        And update recently_deleted buffer
        """
        tracks_to_delete = []
        
        for track in self.tracks:
            delete_criteria = (
                (track.state == 'tentative' and track.time_since_update >= self.min_hits) or
                (track.time_since_update > self.max_disappeared) or
                (track.age > self.max_age)
            )
            
            if delete_criteria:
                tracks_to_delete.append(track)
        
        # Move deleted tracks to buffers
        for track in tracks_to_delete:
            if track in self.tracks:
                self.tracks.remove(track)
                
                # Add to recently_deleted for fast recovery
                self.recently_deleted.append(track)
                if len(self.recently_deleted) > 30:  # Keep only 30 most recent
                    oldest = self.recently_deleted.pop(0)
                    
                    # Store in deleted_tracks for long-term ID consistency
                    self.deleted_tracks[oldest.track_id] = oldest
                    
                    # Limit size of deleted_tracks
                    if len(self.deleted_tracks) > self.reid_buffer_size:
                        # Remove oldest or lowest confidence track
                        oldest_id = next(iter(self.deleted_tracks))
                        del self.deleted_tracks[oldest_id]
    
    def get_stats(self):
        """Get tracking statistics"""
        confirmed = len([t for t in self.tracks if t.state == 'confirmed'])
        tentative = len([t for t in self.tracks if t.state == 'tentative'])
        
        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks),
            'confirmed_tracks': confirmed,
            'tentative_tracks': tentative,
            'next_id': self.next_id,
            'active_persons': confirmed,
            'total_known_persons': self.next_id - 1,  # For compatibility with existing code
            'recovery_count': len(self.recently_deleted),  # Track recovery stats
            'total_tracks_created': self.next_id - 1  # Total number of unique tracks created
        }


# Import scipy at the top if available, fallback to numpy
try:
    import scipy.linalg
    import scipy
except ImportError:
    print("Warning: scipy not available, using numpy for linear algebra operations")
    import numpy as np
    class MockScipy:
        linalg = np.linalg
    scipy = MockScipy()

# For backward compatibility, set KalmanTracker = ByteTracker
# KalmanTracker = ByteTracker

# For best results, use PersonTrackerKalman from utils.transreid
# (Do not import KalmanTracker from this file)

# Recommend using TransReID-based tracker for best robustness
# See utils/transreid.py for PersonTrackerKalman

# Example usage (do not use ByteTracker directly for best results):
# from utils.transreid import TransReIDModel, PersonTrackerKalman
# transreid_model = TransReIDModel(weights_path="path/to/transreid_weights.pth", device="cuda")
# tracker = PersonTrackerKalman(transreid_model, similarity_threshold=0.7, max_disappeared=30)
# person_ids = tracker.assign_ids(person_crops, bboxes, frame_number)