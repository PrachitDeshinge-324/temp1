import numpy as np
import torch
from collections import defaultdict
import cv2

class ReIDEnhancedTracker:
    """
    Enhances YOLO tracker results with TransReID identity verification.
    Acts as a post-processor for the built-in tracker.
    """
    def __init__(self, 
                 transreid_model, 
                 similarity_threshold=0.7, 
                 feature_history_size=10,
                 reid_memory_frames=30):
        """
        Args:
            transreid_model: TransReID model instance
            similarity_threshold: Threshold for feature similarity (0.0-1.0)
            feature_history_size: Number of feature vectors to store per track
            reid_memory_frames: How many frames to remember tracks for re-identification
        """
        self.transreid_model = transreid_model
        self.similarity_threshold = similarity_threshold
        self.feature_history_size = feature_history_size
        self.reid_memory_frames = reid_memory_frames
        
        # Initialize storage
        self.track_features = {}           # track_id -> list of feature vectors
        self.last_seen = {}                # track_id -> frame_idx when last seen
        self.next_id = 1                   # Start IDs from 1
        self.frame_idx = 0                 # Current frame index
        self.id_mapping = {}               # Maps YOLO IDs to corrected IDs
        self.first_frame = True            # Flag for first frame processing
        
    def update(self, frame, boxes, person_ids, confidences):
        """
        Process YOLO tracking results and enhance IDs with ReID
        
        Args:
            frame: The current video frame
            boxes: Detected bounding boxes (x1, y1, x2, y2)
            person_ids: Person IDs from YOLO tracker
            confidences: Detection confidences
            
        Returns:
            Corrected person IDs
        """
        self.frame_idx += 1
        corrected_ids = person_ids.copy()
        
        # Track which IDs we've already used in this frame to prevent duplicates
        used_ids_in_frame = set()
        
        # For the very first frame, use YOLO's IDs directly or assign new sequential IDs
        # This ensures that we start with unique IDs for each person
        if self.first_frame:
            self.first_frame = False
            
            # Extract crops and features for initialization
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                # Ensure box is within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:  # Valid box
                    # Assign a new unique ID
                    new_id = self.next_id
                    self.next_id += 1
                    corrected_ids[i] = new_id
                    used_ids_in_frame.add(new_id)
                    
                    # Extract and store features
                    crop = frame[y1:y2, x1:x2]
                    feature = self.transreid_model.extract_features(crop)
                    self.track_features[new_id] = [feature]
                    self.last_seen[new_id] = self.frame_idx
                    self.id_mapping[int(person_ids[i])] = new_id
                    
            # Return the newly assigned IDs for the first frame
            return corrected_ids
        
        # For subsequent frames, perform normal processing
        # Extract crops from frame
        crops = []
        valid_indices = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Ensure box is within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:  # Valid box
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
                valid_indices.append(i)
        
        if not crops:
            return person_ids
        
        # Extract features
        features = []
        for crop in crops:
            feature = self.transreid_model.extract_features(crop)
            features.append(feature)
        
        # First pass: find the best matches for each detection and score them
        match_candidates = []
        for idx, i in enumerate(valid_indices):
            if idx >= len(features):
                continue
                
            feature = features[idx]
            yolo_id = int(person_ids[i])
            
            # If this is a known ID that we've mapped before
            if yolo_id in self.id_mapping:
                reid_id = self.id_mapping[yolo_id]
                match_candidates.append({
                    'idx': idx,
                    'valid_idx': i,
                    'reid_id': reid_id,
                    'score': 1.0,  # Perfect match for existing mapped ID
                    'feature': feature,
                    'yolo_id': yolo_id
                })
            else:
                # Compare with existing tracks
                best_match_id = None
                best_similarity = 0.0
                
                for track_id, track_features in self.track_features.items():
                    # Skip if track hasn't been seen recently
                    if self.frame_idx - self.last_seen.get(track_id, 0) > self.reid_memory_frames:
                        continue
                    
                    # Calculate similarity with this track
                    similarities = [self._compute_similarity(feature, tf) for tf in track_features]
                    if not similarities:
                        continue
                    
                    # Use both max and average similarity
                    max_sim = max(similarities)
                    avg_sim = sum(similarities) / len(similarities)
                    similarity = 0.7 * max_sim + 0.3 * avg_sim
                    
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = track_id
                
                if best_match_id is not None:
                    # Found a match with existing track
                    match_candidates.append({
                        'idx': idx,
                        'valid_idx': i,
                        'reid_id': best_match_id,
                        'score': best_similarity,
                        'feature': feature,
                        'yolo_id': yolo_id
                    })
                else:
                    # No match found - will create new ID
                    match_candidates.append({
                        'idx': idx,
                        'valid_idx': i,
                        'reid_id': None,
                        'score': 0.0,
                        'feature': feature,
                        'yolo_id': yolo_id
                    })
        
        # Sort match candidates by score (highest first)
        match_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Second pass: assign IDs ensuring no duplicates in this frame
        for match in match_candidates:
            i = match['valid_idx']
            feature = match['feature']
            yolo_id = match['yolo_id']
            
            if match['reid_id'] is not None and match['reid_id'] not in used_ids_in_frame:
                # Use the matched ID since it's not used yet
                reid_id = match['reid_id']
                corrected_ids[i] = reid_id
                used_ids_in_frame.add(reid_id)
                
                # Update mapping
                self.id_mapping[yolo_id] = reid_id
                
                # Update feature history
                if reid_id not in self.track_features:
                    self.track_features[reid_id] = [feature]
                else:
                    self.track_features[reid_id].append(feature)
                    if len(self.track_features[reid_id]) > self.feature_history_size:
                        self.track_features[reid_id] = self.track_features[reid_id][-self.feature_history_size:]
                
                # Update last seen
                self.last_seen[reid_id] = self.frame_idx
            else:
                # Create a new ID (either no match or matched ID already used)
                new_id = self.next_id
                self.next_id += 1
                corrected_ids[i] = new_id
                used_ids_in_frame.add(new_id)
                
                # Update mapping and store features
                self.id_mapping[yolo_id] = new_id
                self.track_features[new_id] = [feature]
                self.last_seen[new_id] = self.frame_idx
        
        # Cleanup old tracks to save memory
        if self.frame_idx % 100 == 0:
            self._cleanup_old_tracks()
            
        return corrected_ids
    
    def _compute_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors"""
        return torch.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0)).item()
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen in a while"""
        current_tracks = {}
        current_last_seen = {}
        current_mappings = {}
        
        for track_id, last_frame in self.last_seen.items():
            if self.frame_idx - last_frame <= self.reid_memory_frames * 2:
                current_tracks[track_id] = self.track_features.get(track_id, [])
                current_last_seen[track_id] = last_frame
                
        # Update mappings
        for yolo_id, reid_id in self.id_mapping.items():
            if reid_id in current_tracks:
                current_mappings[yolo_id] = reid_id
                
        self.track_features = current_tracks
        self.last_seen = current_last_seen
        self.id_mapping = current_mappings