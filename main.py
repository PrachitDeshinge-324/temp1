from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import logging
from collections import defaultdict
import argparse
import os
import json
import torchvision.transforms as T
from PIL import Image
import sys
import pandas as pd

# Import custom modules
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator
from utils.trackers import YOLOTracker, Detection
from utils.pose_detector import PoseDetector
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor
from utils.id_merger import IDMerger  # New import for ID merger
from utils.transreid_model import TransReIDModel, load_transreid_model  # New import for TransReID

# Suppress Ultralytics YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

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
            # Use the new TransReIDModel implementation
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

    def update(self, frame, detections):
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
       
def parse_args():
    parser = argparse.ArgumentParser(description="Gait-based Person Identification")
    parser.add_argument("--video", type=str, default="../Person_New/input/3c.mp4",
                       help="Path to input video")
    parser.add_argument("--start_frame", type=int, default=150,
                       help="Starting frame number")
    parser.add_argument("--end_frame", type=int, default=2000,
                       help="Ending frame number")
    parser.add_argument("--output_features", type=str, default="industrial_gait_features.csv",
                       help="Path to save extracted features")
    parser.add_argument("--model", type=str, default="gait_validation_results/gait_classifier_model.pkl",
                       help="Path to pre-trained gait classifier model")
    parser.add_argument("--identify", action="store_true",
                       help="Perform real-time identification")
    parser.add_argument("--output_video", type=str, default="",
                       help="Path to save output video (if provided, no window will be shown)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no window display)")
    parser.add_argument("--buffer_size", type=float, default=0.05,
                       help="Buffer size ratio around detected person (default: 0.1)")
    parser.add_argument("--save_bbox_info", action="store_true", default=False,
                       help="Save bounding box information to JSON")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory to save all output files")
    parser.add_argument("--use_transreid", action="store_true", default=True,
                       help="Use TransReID for person tracking (default: True)")
    parser.add_argument("--transreid_model", type=str, default="model/transreid_vitbase.pth",
                       help="Path to TransReID model weights")
    parser.add_argument("--tracking_iou", type=float, default=0.5,
                       help="IoU threshold for tracking association")
    parser.add_argument("--tracking_age", type=int, default=30,
                       help="Maximum age for tracks before deletion")
    # New argument for ID merging
    parser.add_argument("--merge_ids", action="store_true", default=False,
                       help="Run ID merger to merge incorrectly split tracking IDs")
    return parser.parse_args()

def export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, feature_order_path, flat_npy_path):
    """Export all collected data to files with improved NaN handling"""
    # Filter out tracks with invalid IDs
    valid_track_ids = [track_id for track_id in gait_analyzer.track_history.keys() if track_id > 0]
    
    # Print statistics about valid and invalid tracks
    print(f"Total tracks: {len(gait_analyzer.track_history)}")
    print(f"Valid tracks (ID > 0): {len(valid_track_ids)}")
    print(f"Invalid tracks: {len(gait_analyzer.track_history) - len(valid_track_ids)}")
    
    # Create a filtered track history with only valid IDs
    filtered_track_history = {
        track_id: gait_analyzer.track_history[track_id] 
        for track_id in valid_track_ids
    }
    
    # Store the original track history
    original_track_history = gait_analyzer.track_history
    
    # Temporarily replace with filtered version for export
    gait_analyzer.track_history = filtered_track_history
    
    # Export features to CSV (now using only valid track IDs)
    gait_analyzer.export_features_csv(features_path)
    print(f"Exported gait features to {features_path}")
    
    # Save bounding box info if requested - also filter out invalid IDs
    if args.save_bbox_info:
        filtered_bbox_info = {
            track_id: bbox_info[track_id]
            for track_id in bbox_info
            if track_id > 0
        }
        with open(bbox_json_path, 'w') as f:
            json.dump(filtered_bbox_info, f)
        print(f"Saved bounding box information to {bbox_json_path}")
    
    # Gather all unique feature keys across all tracks
    all_feature_keys = set()
    for track_id in filtered_track_history:
        # Get both original features and view-invariant features
        features = gait_analyzer.get_features(track_id)
        invariant_features = gait_analyzer.calculate_view_invariant_features(track_id)
        
        if features is not None:
            all_feature_keys.update([k for k, v in features.items() 
                              if isinstance(v, (int, float, np.floating, np.integer))])
        if invariant_features is not None:
            all_feature_keys.update([k for k, v in invariant_features.items() 
                              if isinstance(v, (int, float, np.floating, np.integer))])
    
    # Sort features for consistency
    feature_keys = sorted(all_feature_keys)
    
    # Save the feature order to text file
    with open(feature_order_path, 'w') as f:
        for k in feature_keys:
            f.write(f"{k}\n")
    print(f"Saved feature order to {feature_order_path}")
    print(f"Total features in text file: {len(feature_keys)}")
    
    # First determine the maximum number of keypoints across all tracks
    max_keypoints_count = 0
    for track_id, history in filtered_track_history.items():
        for _, kpts in history:
            max_keypoints_count = max(max_keypoints_count, len(kpts))
    
    # Calculate the expected number of coordinates (x,y per keypoint)
    max_coords = max_keypoints_count * 2
    print(f"Maximum number of keypoints: {max_keypoints_count} (coordinates: {max_coords})")
    
    # Prepare data for numpy array - only using valid track IDs
    all_rows = []
    
    # Store track-specific history for NaN replacement
    track_frame_history = {}
    track_feature_history = {}
    track_keypoint_history = {}
    track_norm_keypoint_history = {}
    
    for track_id, history in filtered_track_history.items():
        # Validate track_id again to be absolutely certain
        if track_id <= 0:
            print(f"Warning: Skipping invalid track_id {track_id} during export")
            continue
        
        # Sort history by frame index to ensure chronological order
        sorted_history = sorted(history, key=lambda x: x[0])
        
        # Initialize history trackers for this track
        track_frame_history[track_id] = []
        track_feature_history[track_id] = []
        track_keypoint_history[track_id] = []
        track_norm_keypoint_history[track_id] = []
        
        for frame_idx, kpts in sorted_history:
            # Get both original and view-invariant features
            features = gait_analyzer.get_features(track_id)
            invariant_features = gait_analyzer.calculate_view_invariant_features(track_id)
            
            # Create combined feature dictionary
            combined_features = {}
            if features is not None:
                combined_features.update({k: v for k, v in features.items() 
                                      if isinstance(v, (int, float, np.floating, np.integer))})
            if invariant_features is not None:
                combined_features.update({k: v for k, v in invariant_features.items() 
                                      if isinstance(v, (int, float, np.floating, np.integer))})
            
            # Create feature vector using the same order as in feature_order_path
            feature_vec = np.array([
                combined_features.get(k, np.nan) for k in feature_keys
            ], dtype=np.float32)
            
            # Initialize arrays for keypoints
            original_kpt_array = np.full(max_coords, np.nan, dtype=np.float32)
            normalized_kpt_array = np.full(max_coords, np.nan, dtype=np.float32)
            
            # Fill in available keypoints at their correct positions
            for i, pt in enumerate(kpts):
                if isinstance(pt, np.ndarray) and pt.size == 2 and i*2+1 < max_coords:
                    # Place coordinates at the correct indices
                    original_kpt_array[i*2] = pt[0]
                    original_kpt_array[i*2+1] = pt[1]
            
            # Get normalized keypoints
            norm_kpts = gait_analyzer.normalize_keypoints(kpts)
            if norm_kpts is not None:
                for i, pt in enumerate(norm_kpts):
                    if isinstance(pt, np.ndarray) and pt.size == 2 and i*2+1 < max_coords:
                        normalized_kpt_array[i*2] = pt[0]
                        normalized_kpt_array[i*2+1] = pt[1]
            
            # Fix NaN values using historical data
            # 1. Check for NaN values in keypoints
            if np.isnan(original_kpt_array).any():
                # If we have enough history, use that to fill NaNs
                if len(track_keypoint_history[track_id]) > 0:
                    # For each NaN value, try to replace with historical average
                    for i in range(len(original_kpt_array)):
                        if np.isnan(original_kpt_array[i]):
                            # Collect historical values for this coordinate
                            coord_history = [hist[i] for hist in track_keypoint_history[track_id][-30:] 
                                           if i < len(hist) and not np.isnan(hist[i])]
                            if coord_history:  # If we have any valid history
                                original_kpt_array[i] = np.mean(coord_history)
            
            # 2. Check for NaN values in normalized keypoints
            if np.isnan(normalized_kpt_array).any():
                # If we have enough history, use that to fill NaNs
                if len(track_norm_keypoint_history[track_id]) > 0:
                    # For each NaN value, try to replace with historical average
                    for i in range(len(normalized_kpt_array)):
                        if np.isnan(normalized_kpt_array[i]):
                            # Collect historical values for this coordinate
                            coord_history = [hist[i] for hist in track_norm_keypoint_history[track_id][-30:] 
                                           if i < len(hist) and not np.isnan(hist[i])]
                            if coord_history:  # If we have any valid history
                                normalized_kpt_array[i] = np.mean(coord_history)
            
            # 3. Check for NaN values in feature vector
            if np.isnan(feature_vec).any():
                # If we have enough history, use that to fill NaNs
                if len(track_feature_history[track_id]) > 0:
                    # For each NaN value, try to replace with historical average
                    for i in range(len(feature_vec)):
                        if np.isnan(feature_vec[i]):
                            # Collect historical values for this feature
                            feat_history = [hist[i] for hist in track_feature_history[track_id][-30:] 
                                          if i < len(hist) and not np.isnan(hist[i])]
                            if feat_history:  # If we have any valid history
                                feature_vec[i] = np.mean(feat_history)
            
            # Append current data to history after fixing NaNs
            track_keypoint_history[track_id].append(original_kpt_array.copy())
            track_norm_keypoint_history[track_id].append(normalized_kpt_array.copy())
            track_feature_history[track_id].append(feature_vec.copy())
            track_frame_history[track_id].append(frame_idx)
            
            # Combine all parts to create the row
            row = [float(track_id), float(frame_idx)] + \
                  original_kpt_array.tolist() + \
                  normalized_kpt_array.tolist() + \
                  feature_vec.tolist()
            
            all_rows.append(row)
    
    # Convert to numpy array
    all_rows_np = np.array(all_rows, dtype=np.float32)
    
    # Final pass to replace remaining NaNs with zeros
    nan_count_before = np.isnan(all_rows_np).sum()
    if nan_count_before > 0:
        print(f"Replacing {nan_count_before} remaining NaN values with zeros")
        all_rows_np = np.nan_to_num(all_rows_np, nan=0.0)
    
    # Save the cleaned data
    np.save(flat_npy_path, all_rows_np)
    
    # Calculate total values per row
    total_values_per_row = 2 + max_coords*2 + len(feature_keys)
    
    print(f"Saved flat numpy array with id, frame, keypoints, normalized keypoints, features to {flat_npy_path}")
    print(f"Each row contains: 1 track_id + 1 frame_idx + {max_coords} keypoint values + {max_coords} normalized keypoint values + {len(feature_keys)} features = {total_values_per_row} total values")
    print(f"Total rows: {len(all_rows_np)}")
    
    # Restore original track history
    gait_analyzer.track_history = original_track_history

def fix_invariant_features(gait_analyzer, flat_npy_path, feature_order_path, output_csv_path):
    """Fix invariant features by copying from track 1 to other tracks with proper scaling"""
    print("Fixing missing invariant features for all tracks...")
    
    # Load original CSV to get data with invariant features for track 1
    # First check if the CSV exists - if not, we need to generate it
    original_csv = output_csv_path.replace('_with_invariants.csv', '.csv')
    if not os.path.exists(original_csv):
        print(f"Original CSV {original_csv} not found, generating from analyzer...")
        gait_analyzer.export_features_csv(original_csv)
    
    # Read the CSV with pandas for easier manipulation
    try:
        df = pd.read_csv(original_csv)
        print(f"Loaded CSV with {len(df)} tracks")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Identify invariant feature columns (those starting with 'inv_')
    inv_columns = [col for col in df.columns if col.startswith('inv_')]
    print(f"Found {len(inv_columns)} invariant feature columns")
    
    if len(inv_columns) == 0:
        print("No invariant features found in the CSV!")
        return False
    
    # Find template track with the most non-zero invariant features
    template_track = None
    max_nonzero = -1
    
    for _, row in df.iterrows():
        track_id = row['track_id']
        nonzero_count = sum(1 for col in inv_columns if row[col] != 0)
        
        if nonzero_count > max_nonzero:
            max_nonzero = nonzero_count
            template_track = row
    
    if template_track is None or max_nonzero == 0:
        print("No tracks found with non-zero invariant features!")
        return False
    
    print(f"Using track {int(template_track['track_id'])} as template with {max_nonzero} non-zero invariant features")
    
    # Create a copy of the dataframe for modification
    df_fixed = df.copy()
    
    # For each track that's missing invariant features, copy and scale them from the template
    for idx, row in df_fixed.iterrows():
        track_id = int(row['track_id'])
        if track_id == template_track['track_id']:
            continue  # Skip template track
            
        missing_features = sum(1 for col in inv_columns if row[col] == 0)
        if missing_features > 0:
            print(f"Fixing {missing_features} invariant features for track {track_id}")
            
            # Calculate scaling factors based on height/torso length ratio
            # We'll use avg_neck_to_left_shoulder_length as an approximation
            template_scale = template_track.get('avg_neck_to_left_shoulder_length', 1.0)
            target_scale = row.get('avg_neck_to_left_shoulder_length', 1.0)
            
            if template_scale > 0 and target_scale > 0:
                scale_ratio = target_scale / template_scale
            else:
                scale_ratio = 1.0
                
            print(f"  Scale ratio: {scale_ratio:.2f}")
            
            # Copy each invariant feature with scaling
            for col in inv_columns:
                if row[col] == 0:  # Only replace if the value is missing
                    # Apply different scaling strategies based on feature type
                    template_value = template_track[col]
                    
                    if 'angle' in col:  # Angles should remain the same regardless of scale
                        df_fixed.at[idx, col] = template_value
                    elif 'ratio' in col:  # Ratios should remain roughly the same
                        df_fixed.at[idx, col] = template_value
                    else:  # Other features like lengths should be scaled
                        df_fixed.at[idx, col] = template_value * scale_ratio
    
    # Save the fixed dataframe as a new CSV
    df_fixed.to_csv(output_csv_path, index=False)
    print(f"Saved fixed CSV with invariant features to {output_csv_path}")
    return True

# New function to handle ID merging
def merge_ids(args, features_npy_path):
    """
    Interactive session to merge incorrectly split tracking IDs
    """
    print("\n=== Starting ID Merger ===")
    print("This utility will help you merge tracking IDs that belong to the same person.")
    
    # Initialize ID merger
    bbox_json_path = os.path.join(args.results_dir, "bbox_info.json")
    
    if not os.path.exists(bbox_json_path):
        print(f"Error: Bounding box info file not found at {bbox_json_path}")
        return None, None
    
    merger = IDMerger(args.video, bbox_json_path, features_npy_path)
    
    try:
        # Run interactive merging session
        merged_ids, id_to_name = merger.merge_ids_interactive()
        
        # Save results
        merger.save_updated_data(args.results_dir)
        
        return merged_ids, id_to_name
    
    finally:
        merger.close()

def generate_processed_csv(preprocessed_npy, output_csv_path, feature_order_path):
    """Generate a clean CSV without NaN values from preprocessed numpy data"""
    print(f"Generating processed CSV file at {output_csv_path}")
    
    # Load preprocessed data
    data = np.load(preprocessed_npy)
    
    # Read feature names from feature order file
    with open(feature_order_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Calculate column offsets
    track_id_col = 0
    frame_idx_col = 1
    
    # Calculate the number of keypoint columns
    remaining_cols = data.shape[1] - 2 - len(feature_names)
    n_keypoint_coords = remaining_cols // 2
    
    # Features start after track_id, frame_idx, original keypoints, and normalized keypoints
    feature_start_col = 2 + n_keypoint_coords * 2
    
    # Extract unique track IDs
    unique_track_ids = np.unique(data[:, track_id_col]).astype(int)
    
    # Dictionary to hold averaged features for each track
    track_features = {}
    
    # For each track, calculate average feature values
    for track_id in unique_track_ids:
        # Get all rows for this track
        track_mask = data[:, track_id_col] == track_id
        track_data = data[track_mask]
        
        # Calculate mean and std for each feature
        feature_means = np.mean(track_data[:, feature_start_col:], axis=0)
        feature_stds = np.std(track_data[:, feature_start_col:], axis=0)
        
        # Store in dictionary
        track_features[track_id] = {'means': feature_means, 'stds': feature_stds}
    
    # Write CSV header (track_id and feature names)
    with open(output_csv_path, 'w') as f:
        header = ['track_id'] + feature_names
        f.write(','.join(header) + '\n')
        
        # Write a row for each unique track
        for track_id in unique_track_ids:
            # Start with track ID
            row = [str(int(track_id))]
            
            # Add each feature value
            for i, feature in enumerate(feature_names):
                # Some features might be averages, others might be raw values
                # Use the means we calculated
                if i < len(track_features[track_id]['means']):
                    row.append(str(track_features[track_id]['means'][i]))
                else:
                    row.append('0.0')  # Fallback value if missing
            
            # Write the row
            f.write(','.join(row) + '\n')
    
    print(f"CSV created with data for {len(unique_track_ids)} tracks")

def main():
    args = parse_args()
    # Configure device settings
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Always use CPU for pose on Mac due to MPS bug
    pose_device = 'cpu' if torch.backends.mps.is_available() else device
    print(f"Using device: {device}, Pose device: {pose_device}")
    
    # Create results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Update output paths
    base_features_name = os.path.basename(args.output_features)
    features_path = os.path.join(results_dir, base_features_name)
    flat_npy_path = features_path.replace('.csv', '_flat.npy')
    bbox_json_path = os.path.join(results_dir, "bbox_info.json")
    feature_order_path = features_path.replace('.csv', '_feature_order.txt')
    
    # Configure output video path
    if args.output_video:
        output_video_path = os.path.join(results_dir, os.path.basename(args.output_video))
    else:
        output_video_path = None
    
    # Initialize models
    model_det = YOLO('model/yolo11x.pt').to(device)
    model_pose = YOLO('model/yolo11x-pose.pt').to(pose_device)
    
    # Initialize trackers with clear naming
    yolo_tracker = YOLOTracker(model_det, device=device)
    
    # Initialize TransReID tracker if enabled (default is now True)
    transreid_tracker = None
    if args.use_transreid:
        print("Using TransReID for robust re-identification and tracking")
        transreid_tracker = TransReIDTracker(
            device=device, 
            reid_model_path=args.transreid_model,
            iou_threshold=args.tracking_iou,
            max_age=args.tracking_age
        )
    else:
        print("Using YOLO's built-in tracking (not recommended)")
    
    # Initialize pose detector
    pose_detector = PoseDetector(model_pose, device=pose_device)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Initialize gait analyzer
    gait_analyzer = GaitFeatureExtractor()
    
    # Initialize person identifier if requested
    person_identifier = None
    if args.identify and os.path.exists(args.model):
        person_identifier = GaitValidator()
        if person_identifier.load_model(args.model):
            print("Loaded gait classifier for person identification")
        else:
            person_identifier = None
    
    # Initialize video processor
    video_processor = VideoProcessor(
        args.video, 
        args.start_frame,
        args.end_frame,
        output_video_path,
        args.headless
    )
    
    # Track state
    keypoints_history = defaultdict(lambda: [])
    HISTORY_LENGTH = 5
    bbox_info = defaultdict(list)
    
    def process_frame(frame, frame_count, fps):
        """Process a single video frame"""
        # Draw FPS counter
        visualizer.draw_fps(frame, fps)
        
        # Get detections using the correct tracker
        if args.use_transreid:
            # First get detections from YOLO
            yolo_detections = yolo_tracker.update(frame)
            # Then feed them to TransReID tracker
            detections = transreid_tracker.update(frame, yolo_detections)
            
            # Extra validation to ensure no invalid IDs pass through
            validated_detections = []
            for det in detections:
                # Ensure track_id is a positive integer (not 0, not negative)
                if not hasattr(det, 'track_id') or det.track_id <= 0:
                    # Generate new valid ID for this detection
                    new_id = transreid_tracker.next_id
                    det.track_id = new_id
                    transreid_tracker.next_id = new_id + 1
                validated_detections.append(det)
            detections = validated_detections
        else:
            # Use YOLO's built-in tracking
            detections = yolo_tracker.update(frame)
            
            # Even for YOLO tracking, validate IDs
            validated_detections = []
            for det in detections:
                if not hasattr(det, 'track_id') or det.track_id <= 0:
                    det.track_id = max(1, getattr(det, 'track_id', 0) + 100)  # Add offset to avoid ID conflicts
                validated_detections.append(det)
            detections = validated_detections
        
        # Process each detection
        for detection in detections:
            track_id = detection.track_id
            
            # Additional validation - ensure track_id is positive
            if track_id <= 0:
                print(f"Warning: Invalid track ID {track_id} found after validation. Setting to fallback ID.")
                track_id = max(1, frame_count % 1000 + 100)  # Use frame count as fallback ID source
                detection.track_id = track_id
            
            # Draw bounding box and get buffered box coordinates
            buffered_box, color = visualizer.draw_detection(
                frame, detection, args.buffer_size)
            x1_buf, y1_buf, x2_buf, y2_buf = buffered_box
            
            # Save bounding box info if requested
            if args.save_bbox_info:
                bbox_info[int(track_id)].append({
                    'track_id': int(track_id),
                    'frame_idx': frame_count,
                    'x1': x1_buf,
                    'y1': y1_buf, 
                    'x2': x2_buf,
                    'y2': y2_buf,
                    'original_box': [int(x) for x in detection.bbox]
                })
            
            # Crop person from frame
            person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
            if person_crop.size == 0:
                continue
            
            # Detect poses in the crop
            keypoints_list = pose_detector.detect(person_crop)
            
            for keypoints in keypoints_list:
                # Update keypoints history for smoothing
                if len(keypoints_history[track_id]) >= HISTORY_LENGTH:
                    keypoints_history[track_id].pop(0)
                keypoints_history[track_id].append(keypoints.copy())
                
                # Apply temporal smoothing to keypoints
                smoothed_keypoints = pose_detector.smooth_keypoints(
                    keypoints_history[track_id][:-1], keypoints)
                
                # Update gait analyzer with new keypoints
                gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
                
                # Perform identification if requested
                if person_identifier and frame_count % 15 == 0:
                    feature_vector = gait_analyzer.get_feature_vector(track_id)
                    if feature_vector is not None:
                        identity, confidence = person_identifier.identify_person(feature_vector)
                        visualizer.draw_identity(frame, detection, identity, confidence, color)
                
                # Draw keypoints and skeleton
                visualizer.draw_keypoints(frame, smoothed_keypoints, x1_buf, y1_buf, color)
        
        return frame

    # Process the video
    video_processor.process_video(process_frame)
    
    # Export collected data
    export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, 
                   feature_order_path, flat_npy_path)
    
    # Run ID merger if requested
    if args.merge_ids:
        merged_ids, id_to_name = merge_ids(args, flat_npy_path)
        print("ID merging complete!")
        if merged_ids:
            print(f"Merged {len(merged_ids)} IDs")
        if id_to_name:
            print(f"Assigned names to {len(id_to_name)} unique persons")

    # Generate clean CSV from processed numpy data
    processed_csv_path = features_path.replace('.csv', '_processed.csv')
    generate_processed_csv(flat_npy_path, processed_csv_path, feature_order_path)
    
    # Fix invariant features in a new CSV
    inv_features_csv = os.path.join(args.results_dir, 'industrial_gait_features_with_invariants.csv')
    fix_invariant_features(gait_analyzer, flat_npy_path, feature_order_path, inv_features_csv)
    print(f"Created enriched CSV with invariant features at {inv_features_csv}")
    
if __name__ == "__main__":
    main()