import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from PIL import Image
import os
from functools import lru_cache
import cv2
import numpy as np
from utils.helper import get_best_device

# Import scipy for assignment algorithm
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("Warning: scipy not available. Using greedy assignment fallback.")
    linear_sum_assignment = None

def interpolate_pos_embed_rectangular(model, state_dict):
    """
    Interpolate position embeddings from checkpoint to model when grids don't match,
    properly handling rectangular (non-square) grid layouts.
    """
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # Get grid sizes from model
        model_grid_size = (model.patch_embed.num_y, model.patch_embed.num_x)
        model_num_patches = model_grid_size[0] * model_grid_size[1]
        
        # Get grid size from checkpoint
        checkpoint_num_patches = pos_embed_checkpoint.shape[1] - 1  # minus cls token
        
        # Estimate checkpoint grid size (try to get close to original aspect ratio)
        # For TransReID, most common is 24x10 grid (240 patches) or 24×9 (216 patches)
        if checkpoint_num_patches in [210, 240, 216]:
            if checkpoint_num_patches == 210:
                checkpoint_grid_size = (21, 10)  # 21×10=210 patches
            elif checkpoint_num_patches == 240:
                checkpoint_grid_size = (24, 10)  # 24×10=240 patches
            elif checkpoint_num_patches == 216:
                checkpoint_grid_size = (24, 9)   # 24×9=216 patches
        else:
            # For other sizes, try to approximate proportionally
            ratio = model_grid_size[0] / model_grid_size[1]  # height/width ratio
            checkpoint_h = int((checkpoint_num_patches * ratio) ** 0.5)
            checkpoint_w = checkpoint_num_patches // checkpoint_h
            while checkpoint_h * checkpoint_w != checkpoint_num_patches:
                checkpoint_h -= 1
                checkpoint_w = checkpoint_num_patches // checkpoint_h
            checkpoint_grid_size = (checkpoint_h, checkpoint_w)
            
        print(f"Interpolating position embeddings:")
        print(f"  - Checkpoint grid: {checkpoint_grid_size[0]}×{checkpoint_grid_size[1]} ({checkpoint_num_patches} patches)")
        print(f"  - Model grid: {model_grid_size[0]}×{model_grid_size[1]} ({model_num_patches} patches)")
        
        # Handle class token and reshape
        cls_pos_embed = pos_embed_checkpoint[:, 0:1, :]
        pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
        
        # Reshape into grid
        pos_embed_checkpoint = pos_embed_checkpoint.reshape(
            1, checkpoint_grid_size[0], checkpoint_grid_size[1], embedding_size
        ).permute(0, 3, 1, 2)
        
        # Interpolate to new size
        import torch.nn.functional as F
        pos_embed_new = F.interpolate(
            pos_embed_checkpoint, 
            size=model_grid_size, 
            mode='bicubic', 
            align_corners=False
        )
        
        # Reshape back
        pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(
            1, model_grid_size[0] * model_grid_size[1], embedding_size
        )
        
        # Attach class token
        new_pos_embed = torch.cat((cls_pos_embed, pos_embed_new), dim=1)
        state_dict['pos_embed'] = new_pos_embed
        
    return state_dict

class TransReIDModel:
    def __init__(self, weights_path, device=None):
        """
        Initialize TransReID model for person re-identification
        
        Args:
            weights_path: Path to pre-trained weights
            device: Device to run the model on
        """
        self.device = device if device is not None else get_best_device()
        
        # TransReID feature dimension (commonly 768 for ViT-base)
        self.feature_dim = 768
        
        # Initialize model
        self.model = self._load_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, weights_path):
        if not os.path.exists(weights_path):
            print(f"Warning: TransReID weights file not found at {weights_path}")
            raise FileNotFoundError(f"TransReID weights file not found at {weights_path}")

        try:
            print(f"Loading TransReID weights from {weights_path}...")
            model = self._create_transreid_model()
            # Load the state dict and remove 'base.' prefix if present
            state_dict = torch.load(weights_path, map_location='cpu')
            # Add to utils/transreid_model.py after model loading
            model_params = sum(p.numel() for p in model.parameters())
            print(f"TransReID model has {model_params:,} parameters")

            # Print model structure
            print("TransReID Model Architecture:")
            print(model.__class__.__name__)
            for name, module in model.named_children():
                print(f"  - {name}: {module.__class__.__name__}")
            
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'base.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('base.'):
                    new_state_dict[k[5:]] = v
                else:
                    new_state_dict[k] = v
            
            # Apply position embedding interpolation
            new_state_dict = interpolate_pos_embed_rectangular(model, new_state_dict)
            
            # Load state dict
            model.load_state_dict(new_state_dict, strict=False)
            print("TransReID weights loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise
    
    def _create_transreid_model(self):
        model = vit_base_patch16_224_TransReID(
            img_size=(256, 128),   # or your dataset's image size
            stride_size=16,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            local_feature=False,
            sie_xishu=1.5
        )
        return model
    
    def extract_features(self, person_crop):
        """
        Extract feature embeddings from a person crop
        
        Args:
            person_crop: Cropped image of a person (BGR format)
        
        Returns:
            Feature embedding tensor
        """
        # Convert BGR to RGB
        if person_crop.size == 0:
            return torch.zeros(self.feature_dim).to(self.device)
            
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Preprocess
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Normalize feature vector
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        
        return normalized_features.squeeze()

class PersonTracker:
    """
    Person tracker using TransReID for re-identification
    """
    def __init__(self, transreid_model, similarity_threshold=0.7, max_disappeared=30):
        """
        Initialize person tracker
        
        Args:
            transreid_model: TransReIDModel instance
            similarity_threshold: Cosine similarity threshold for matching
            max_disappeared: Maximum frames a person can be missing before deletion
        """
        self.transreid_model = transreid_model
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        
        # Track database
        self.known_persons = {}  # person_id -> {"features": tensor, "disappeared": int}
        self.next_id = 1
        
    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors"""
        return torch.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
    
    def assign_ids(self, person_crops, frame_number):
        """
        Assign unique IDs to detected persons
        
        Args:
            person_crops: List of cropped person images
            frame_number: Current frame number
            
        Returns:
            List of person IDs corresponding to each crop
        """
        person_ids = []
        
        # Extract features for all crops
        features_list = []
        for crop in person_crops:
            if crop.size > 0:
                features = self.transreid_model.extract_features(crop)
                features_list.append(features)
            else:
                features_list.append(None)
        
        # Update disappeared counter for all known persons
        for person_id in list(self.known_persons.keys()):
            self.known_persons[person_id]["disappeared"] += 1
            
            # Remove persons who have been missing too long
            if self.known_persons[person_id]["disappeared"] > self.max_disappeared:
                del self.known_persons[person_id]
        
        # Match each detection to known persons or assign new ID
        for i, features in enumerate(features_list):
            if features is None:
                person_ids.append(-1)  # Invalid detection
                continue
                
            best_match_id = None
            best_similarity = 0.0
            
            # Compare with all known persons
            for person_id, person_data in self.known_persons.items():
                similarity = self.cosine_similarity(features, person_data["features"])
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = person_id
            
            if best_match_id is not None:
                # Match found - update features and reset disappeared counter
                self.known_persons[best_match_id]["features"] = features
                self.known_persons[best_match_id]["disappeared"] = 0
                person_ids.append(best_match_id)
            else:
                # No match found - assign new ID
                new_id = self.next_id
                self.next_id += 1
                self.known_persons[new_id] = {
                    "features": features,
                    "disappeared": 0
                }
                person_ids.append(new_id)
        
        return person_ids
    
    def get_stats(self):
        """Get tracking statistics"""
        return {
            "total_known_persons": len(self.known_persons),
            "next_id": self.next_id,
            "active_persons": len([p for p in self.known_persons.values() if p["disappeared"] == 0])
        }

class PersonTrackerKalman:
    """
    Enhanced person tracker using Kalman filter and TransReID for robust tracking
    This is a standalone implementation that doesn't depend on ByteTracker
    """
    def __init__(self, transreid_model, similarity_threshold=0.7, max_disappeared=30):
        """
        Initialize enhanced person tracker
        
        Args:
            transreid_model: TransReIDModel instance for feature extraction
            similarity_threshold: Feature similarity threshold
            max_disappeared: Maximum frames a person can be missing
        """
        import numpy as np
        from dataclasses import dataclass
        from typing import Optional, List
        import torch
        
        # Define Detection class locally to avoid dependency on tracker.py
        @dataclass
        class Detection:
            bbox: np.ndarray
            confidence: float
            features: Optional[torch.Tensor] = None
        
        self.Detection = Detection
        self.transreid_model = transreid_model
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        self.next_id = 1
        
        # Track storage
        self.active_tracks = {}  # track_id -> track_info
        self.track_features = {}  # track_id -> feature_history
        self.track_positions = {}  # track_id -> position_history
        self.track_ages = {}  # track_id -> age_since_last_seen
        self.frame_count = 0
        
        # Tracking parameters optimized for crowded scenes
        self.max_feature_history = 5
        self.position_weight = 0.2  # Reduced weight on position for crowded scenes
        self.feature_weight = 0.8   # Increased weight on features for better ID consistency
        self.min_hits_for_confirmation = 3  # Require 3 hits before confirming track
        self.iou_threshold = 0.3  # IoU threshold for position matching
        
        # Advanced features for ID consistency
        self.lost_tracks = {}  # Recently lost tracks for recovery
        self.max_lost_track_age = 30  # Frames to keep lost tracks
        self.feature_bank_size = 10  # Number of recent features to keep per track
        
        # Debugging and statistics
        self.debug_mode = False
        self.recovery_count = 0  # Track successful recoveries
        self.new_track_count = 0  # Track new track creations
        
    def enable_debug(self, enable=True):
        """Enable or disable debug mode for detailed logging"""
        self.debug_mode = enable
        if enable:
            print("Debug mode enabled for PersonTrackerKalman")
        
    def get_tracking_summary(self):
        """Get a comprehensive tracking summary"""
        stats = self.get_stats()
        
        recovery_rate = (self.recovery_count / max(1, self.new_track_count)) * 100
        
        summary = f"""
=== TransReID Tracking Summary ===
Frame: {self.frame_count}
Active Persons: {stats['active_persons']}
Confirmed Persons: {stats['confirmed_persons']}
Lost Persons: {stats['lost_persons']}
Total Tracks Created: {stats['total_tracks_created']}
Successful Recoveries: {stats['recovery_count']}
Recovery Rate: {recovery_rate:.1f}%
ID Consistency: {'High' if recovery_rate > 20 else 'Medium' if recovery_rate > 10 else 'Improving'}
"""
        return summary
    
    def assign_ids(self, person_crops, bboxes, frame_number):
        """
        Assign unique IDs to detected persons using robust feature matching
        
        Args:
            person_crops: List of cropped person images
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            frame_number: Current frame number
            
        Returns:
            List of person IDs corresponding to each crop/bbox
        """
        import numpy as np
        import torch
        
        if len(person_crops) != len(bboxes):
            raise ValueError(f"Number of crops ({len(person_crops)}) must match number of bboxes ({len(bboxes)})")
        
        self.frame_count = frame_number
        current_detections = []
        
        # Extract features for all detections
        for i, (crop, bbox) in enumerate(zip(person_crops, bboxes)):
            if len(bbox) != 4:
                raise ValueError(f"Bbox at index {i} must have 4 coordinates [x1, y1, x2, y2], got {len(bbox)}")
            
            bbox_array = np.array(bbox, dtype=np.float32)
            
            # Extract features if crop is valid
            features = None
            if crop is not None and hasattr(crop, 'size'):
                try:
                    # Check if it's a PIL image with size attribute
                    if hasattr(crop.size, '__len__') and len(crop.size) == 2:
                        width, height = crop.size
                        if width > 0 and height > 0:
                            features = self.transreid_model.extract_features(crop) if self.transreid_model else None
                    elif hasattr(crop, 'shape') and len(crop.shape) >= 2:
                        # Handle numpy arrays
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            features = self.transreid_model.extract_features(crop) if self.transreid_model else None
                    
                    # Normalize features if extracted
                    if features is not None:
                        features = features / (features.norm() + 1e-8)
                except Exception as e:
                    print(f"Warning: Feature extraction failed for crop {i}: {e}")
                    features = None
            
            # For testing purposes, if no TransReID model, create dummy features
            if features is None and self.transreid_model is None:
                # Create consistent dummy features based on bbox center for testing
                center_x = (bbox_array[0] + bbox_array[2]) / 2
                center_y = (bbox_array[1] + bbox_array[3]) / 2
                # Use rough position bins to create consistent features for similar positions
                bin_x = int(center_x // 100) * 100  # 100-pixel bins for more consistent matching
                bin_y = int(center_y // 100) * 100
                bbox_hash = hash((bin_x, bin_y))
                torch.manual_seed(abs(bbox_hash) % 2**31)
                features = torch.randn(768)
                features = features / features.norm()
            
            detection = self.Detection(
                bbox=bbox_array,
                confidence=1.0,
                features=features
            )
            current_detections.append(detection)
        
        # Match detections with existing tracks
        person_ids = self._match_detections_to_tracks(current_detections)
        
        # Update track ages
        self._update_track_ages()
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return person_ids
    
    def _match_detections_to_tracks(self, detections):
        """Match current detections to existing tracks using feature similarity and position"""
        import numpy as np
        import torch
        from scipy.optimize import linear_sum_assignment
        
        person_ids = [-1] * len(detections)
        
        if not self.active_tracks:
            # No existing tracks, create new ones
            for i, detection in enumerate(detections):
                if detection.features is not None:
                    track_id = self._create_new_track(detection)
                    person_ids[i] = track_id
                else:
                    person_ids[i] = -1
            return person_ids
        
        # Compute similarity matrix between detections and tracks
        track_ids = list(self.active_tracks.keys())
        if not track_ids:
            # Create new tracks for all detections
            for i, detection in enumerate(detections):
                if detection.features is not None:
                    track_id = self._create_new_track(detection)
                    person_ids[i] = track_id
                else:
                    person_ids[i] = -1
            return person_ids
        
        # Build cost matrix
        cost_matrix = np.ones((len(detections), len(track_ids))) * 1e6
        
        for i, detection in enumerate(detections):
            if detection.features is None:
                continue
                
            for j, track_id in enumerate(track_ids):
                # Feature similarity
                feature_sim = self._compute_feature_similarity(detection.features, track_id)
                
                # Position similarity (IoU)
                position_sim = self._compute_position_similarity(detection.bbox, track_id)
                
                # Combined similarity
                combined_sim = (self.feature_weight * feature_sim + 
                               self.position_weight * position_sim)
                
                # Convert similarity to cost (lower is better)
                cost_matrix[i, j] = 1.0 - combined_sim
        
        # Apply threshold - set high cost for poor matches
        cost_matrix[cost_matrix > (1.0 - self.similarity_threshold)] = 1e6
        
        # Solve assignment problem
        if len(detections) > 0 and len(track_ids) > 0:
            try:
                if linear_sum_assignment is not None:
                    det_indices, track_indices = linear_sum_assignment(cost_matrix)
                else:
                    # Fallback greedy assignment
                    det_indices, track_indices = self._greedy_assignment(cost_matrix)
                
                # Process matches
                matched_detections = set()
                for det_idx, track_idx in zip(det_indices, track_indices):
                    if cost_matrix[det_idx, track_idx] < 1e6:  # Valid match
                        track_id = track_ids[track_idx]
                        person_ids[det_idx] = track_id
                        self._update_track(track_id, detections[det_idx])
                        matched_detections.add(det_idx)
                        if self.debug_mode:
                            print(f"Matched detection {det_idx} to existing track {track_id}")
                
                # Create new tracks for unmatched detections
                for i, detection in enumerate(detections):
                    if i not in matched_detections and detection.features is not None:
                        # First try to recover from lost tracks
                        recovered_track_id = self._try_recover_lost_track(detection)
                        if recovered_track_id is not None:
                            person_ids[i] = recovered_track_id
                            if self.debug_mode:
                                print(f"Detection {i} recovered lost track {recovered_track_id}")
                        else:
                            track_id = self._create_new_track(detection)
                            person_ids[i] = track_id
                        
            except Exception as e:
                print(f"Warning: Assignment failed: {e}")
                # Fallback: create new tracks for all detections with features
                for i, detection in enumerate(detections):
                    if detection.features is not None:
                        track_id = self._create_new_track(detection)
                        person_ids[i] = track_id
        
        return person_ids
    
    def _greedy_assignment(self, cost_matrix):
        """Fallback greedy assignment when scipy is not available"""
        import numpy as np
        
        det_indices = []
        track_indices = []
        used_tracks = set()
        
        # For each detection, find the best available track
        for det_idx in range(cost_matrix.shape[0]):
            best_track_idx = -1
            best_cost = 1e6
            
            for track_idx in range(cost_matrix.shape[1]):
                if track_idx not in used_tracks and cost_matrix[det_idx, track_idx] < best_cost:
                    best_cost = cost_matrix[det_idx, track_idx]
                    best_track_idx = track_idx
            
            if best_track_idx >= 0 and best_cost < 1e6:
                det_indices.append(det_idx)
                track_indices.append(best_track_idx)
                used_tracks.add(best_track_idx)
        
        return det_indices, track_indices
    
    def _compute_feature_similarity(self, detection_features, track_id):
        """Compute feature similarity between detection and track"""
        import torch
        
        if track_id not in self.track_features or not self.track_features[track_id]:
            return 0.0
        
        # Use weighted average of recent features for robust matching
        track_feature_history = self.track_features[track_id]
        
        # Compute similarity with each stored feature and use weighted average
        similarities = []
        weights = []
        
        for i, track_features in enumerate(track_feature_history):
            similarity = torch.cosine_similarity(
                detection_features.unsqueeze(0),
                track_features.unsqueeze(0)
            ).item()
            similarities.append(similarity)
            # Give more weight to recent features
            weights.append(1.0 + 0.1 * i)  # More recent = higher weight
        
        if similarities:
            # Weighted average with bias toward recent features
            weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
            # Also consider the maximum similarity for robust matching
            max_sim = max(similarities)
            # Combine weighted average and max (favor consistency but allow recovery)
            return 0.7 * weighted_sim + 0.3 * max_sim
        
        return 0.0
    
    def _compute_position_similarity(self, detection_bbox, track_id):
        """Compute position similarity (IoU) between detection and track"""
        if track_id not in self.track_positions or not self.track_positions[track_id]:
            return 0.0
        
        # Use the most recent position with motion prediction if available
        recent_bbox = self.track_positions[track_id][-1]
        
        # Simple motion prediction: if we have 2+ positions, predict next position
        if len(self.track_positions[track_id]) >= 2:
            prev_bbox = self.track_positions[track_id][-2]
            # Predict position based on velocity
            dx = recent_bbox[0] - prev_bbox[0]  # x1 change
            dy = recent_bbox[1] - prev_bbox[1]  # y1 change
            dw = (recent_bbox[2] - recent_bbox[0]) - (prev_bbox[2] - prev_bbox[0])  # width change
            dh = (recent_bbox[3] - recent_bbox[1]) - (prev_bbox[3] - prev_bbox[1])  # height change
            
            # Predicted bbox
            predicted_bbox = [
                recent_bbox[0] + dx,
                recent_bbox[1] + dy,
                recent_bbox[2] + dx + dw,
                recent_bbox[3] + dy + dh
            ]
            
            # Use both recent and predicted positions (weight toward predicted)
            recent_iou = self._calculate_iou(detection_bbox, recent_bbox)
            predicted_iou = self._calculate_iou(detection_bbox, predicted_bbox)
            
            return 0.4 * recent_iou + 0.6 * predicted_iou
        else:
            return self._calculate_iou(detection_bbox, recent_bbox)
    
    def _create_new_track(self, detection):
        """Create a new track from detection"""
        track_id = self.next_id
        self.next_id += 1
        self.new_track_count += 1
        
        # Initialize track data
        self.active_tracks[track_id] = {
            'created_frame': self.frame_count,
            'last_seen_frame': self.frame_count,
            'hit_count': 1
        }
        
        self.track_features[track_id] = [detection.features.clone()]
        self.track_positions[track_id] = [detection.bbox.copy()]
        self.track_ages[track_id] = 0
        
        if self.debug_mode:
            print(f"Created new track ID {track_id} at frame {self.frame_count}")
        
        return track_id
    
    def _update_track(self, track_id, detection):
        """Update existing track with new detection"""
        # Update track info
        self.active_tracks[track_id]['last_seen_frame'] = self.frame_count
        self.active_tracks[track_id]['hit_count'] += 1
        
        # Update feature history
        self.track_features[track_id].append(detection.features.clone())
        if len(self.track_features[track_id]) > self.max_feature_history:
            self.track_features[track_id].pop(0)
        
        # Update position history
        self.track_positions[track_id].append(detection.bbox.copy())
        if len(self.track_positions[track_id]) > self.max_feature_history:
            self.track_positions[track_id].pop(0)
        
        # Reset age
        self.track_ages[track_id] = 0
    
    def _update_track_ages(self):
        """Update age for all tracks"""
        for track_id in self.active_tracks:
            self.track_ages[track_id] = self.frame_count - self.active_tracks[track_id]['last_seen_frame']
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen for too long"""
        tracks_to_remove = []
        tracks_to_move_to_lost = []
        
        for track_id, age in self.track_ages.items():
            if age > self.max_disappeared:
                tracks_to_remove.append(track_id)
            elif age > 2:  # Move to lost tracks buffer after just 2 frames of being unseen
                tracks_to_move_to_lost.append(track_id)
        
        # Move tracks to lost tracks buffer for potential recovery
        for track_id in tracks_to_move_to_lost:
            if track_id not in self.lost_tracks and track_id in self.active_tracks:
                self.lost_tracks[track_id] = {
                    'features': self.track_features[track_id][-1].clone() if self.track_features[track_id] else None,
                    'last_position': self.track_positions[track_id][-1].copy() if self.track_positions[track_id] else None,
                    'lost_frame': self.frame_count,
                    'track_info': self.active_tracks[track_id].copy()
                }
                if self.debug_mode:
                    print(f"Moved track ID {track_id} to lost tracks buffer (age: {age})")
        
        # Remove old tracks from active tracking
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.track_features:
                del self.track_features[track_id]
            if track_id in self.track_positions:
                del self.track_positions[track_id]
            if track_id in self.track_ages:
                del self.track_ages[track_id]
            if self.debug_mode:
                print(f"Removed track ID {track_id} from active tracks")
        
        # Clean up old lost tracks
        lost_tracks_to_remove = []
        for track_id, lost_info in self.lost_tracks.items():
            if self.frame_count - lost_info['lost_frame'] > self.max_lost_track_age:
                lost_tracks_to_remove.append(track_id)
        
        for track_id in lost_tracks_to_remove:
            if self.debug_mode:
                print(f"Permanently removed lost track ID {track_id}")
            del self.lost_tracks[track_id]

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _try_recover_lost_track(self, detection):
        """Try to recover a lost track that matches the detection"""
        import torch
        
        if not self.lost_tracks or detection.features is None:
            return None
        
        best_track_id = None
        best_similarity = self.similarity_threshold * 0.7  # Lower threshold for recovery
        
        for track_id, lost_info in self.lost_tracks.items():
            if lost_info['features'] is None:
                continue
            
            # Feature similarity
            feature_sim = torch.cosine_similarity(
                detection.features.unsqueeze(0),
                lost_info['features'].unsqueeze(0)
            ).item()
            
            # Position similarity (if available)
            position_sim = 0.0
            if lost_info['last_position'] is not None:
                position_sim = self._calculate_iou(detection.bbox, lost_info['last_position'])
            
            # Combined similarity (emphasize features for recovery)
            combined_sim = 0.9 * feature_sim + 0.1 * position_sim
            
            if self.debug_mode:
                print(f"Recovery check for track {track_id}: feature_sim={feature_sim:.3f}, combined_sim={combined_sim:.3f}")
            
            if combined_sim > best_similarity:
                best_similarity = combined_sim
                best_track_id = track_id
        
        if best_track_id is not None:
            # Recover the track
            lost_info = self.lost_tracks[best_track_id]
            
            # Restore track to active tracks
            self.active_tracks[best_track_id] = lost_info['track_info'].copy()
            self.active_tracks[best_track_id]['last_seen_frame'] = self.frame_count
            
            # Initialize with recovered features and new detection
            self.track_features[best_track_id] = [lost_info['features'].clone(), detection.features.clone()]
            self.track_positions[best_track_id] = [detection.bbox.copy()]
            self.track_ages[best_track_id] = 0
            
            # Remove from lost tracks
            del self.lost_tracks[best_track_id]
            
            self.recovery_count += 1
            if self.debug_mode:
                print(f"✅ Recovered track ID {best_track_id} (similarity: {best_similarity:.3f})")
            return best_track_id
        
        return None
    
    def get_stats(self):
        """Get tracking statistics"""
        active_count = len(self.active_tracks)
        lost_count = len(self.lost_tracks)
        total_created = self.next_id - 1
        
        # Calculate confirmed tracks (tracks with enough hits)
        confirmed_count = sum(1 for track_info in self.active_tracks.values() 
                            if track_info.get('hit_count', 0) >= self.min_hits_for_confirmation)
        
        return {
            "total_known_persons": total_created,
            "next_id": self.next_id,
            "active_persons": active_count,
            "confirmed_persons": confirmed_count,
            "lost_persons": lost_count,
            "frame_count": self.frame_count,
            "total_tracks_created": total_created,
            "recovery_count": self.recovery_count,
            "new_track_count": self.new_track_count
        }

# add cached factory
@lru_cache(maxsize=None)
def load_transreid_model(weights_path: str, device: str):
    """
    Return a cached TransReIDModel instance for given weights and device.
    """
    return TransReIDModel(weights_path, device)