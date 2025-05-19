from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import logging
from collections import defaultdict
import argparse
import os
import json

# Import custom modules
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator
from utils.trackers import YOLOTracker, DeepSORTTracker, Detection
from utils.pose_detector import PoseDetector
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor
from utils.id_merger import IDMerger  # New import for ID merger

# Suppress Ultralytics YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Gait-based Person Identification")
    parser.add_argument("--video", type=str, default="../Person_New/input/3c.mp4",
                       help="Path to input video")
    parser.add_argument("--start_frame", type=int, default=150,
                       help="Starting frame number")
    parser.add_argument("--end_frame", type=int, default=200,
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
    parser.add_argument("--use_deepsort", action="store_true", default=False,
                       help="Use DeepSORT instead of YOLO's built-in tracker")
    # New argument for ID merging
    parser.add_argument("--merge_ids", action="store_true", default=False,
                       help="Run ID merger to merge incorrectly split tracking IDs")
    return parser.parse_args()

def export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, feature_order_path, flat_npy_path):
    """Export all collected data to files"""
    # Export features to CSV
    gait_analyzer.export_features_csv(features_path)
    print(f"Exported gait features to {features_path}")
    
    # Save bounding box info if requested
    if args.save_bbox_info:
        with open(bbox_json_path, 'w') as f:
            json.dump(bbox_info, f)
        print(f"Saved bounding box information to {bbox_json_path}")
    
    # Gather all unique feature keys across all tracks
    all_feature_keys = set()
    for track_id in gait_analyzer.track_history:
        features = gait_analyzer.get_features(track_id)
        if features is not None:
            all_feature_keys.update([k for k, v in features.items() 
                                  if isinstance(v, (int, float, np.floating, np.integer))])
    
    # Sort features for consistency
    feature_keys = sorted(all_feature_keys)
    
    # Save the feature order to text file
    with open(feature_order_path, 'w') as f:
        for k in feature_keys:
            f.write(f"{k}\n")
    print(f"Saved feature order to {feature_order_path}")
    print(f"Total features in text file: {len(feature_keys)}")
    
    # Prepare data for numpy array
    all_rows = []
    for track_id, history in gait_analyzer.track_history.items():
        features = gait_analyzer.get_features(track_id)
        # Create feature vector using the same order as in feature_order_path
        if features is not None:
            feature_vec = np.array([
                features[k] if (k in features and 
                               isinstance(features[k], (int, float, np.floating, np.integer))) 
                else np.nan for k in feature_keys
            ], dtype=np.float32)
        else:
            feature_vec = np.full(len(feature_keys), np.nan, dtype=np.float32)
            
        for frame_idx, kpts in history:
            flat_kpts = []
            for pt in kpts:
                if isinstance(pt, np.ndarray) and pt.size == 2:
                    flat_kpts.extend(pt.tolist())
                else:
                    flat_kpts.extend([0, 0])
            
            # Store track_id, frame_idx, keypoints, and features
            row = [int(track_id), int(frame_idx)] + flat_kpts + feature_vec.tolist()
            all_rows.append(row)
    
    all_rows_np = np.array(all_rows, dtype=np.float32)
    np.save(flat_npy_path, all_rows_np)
    
    # Calculate how many values are in each row
    keypoints_count = len(flat_kpts)
    total_values_per_row = 2 + keypoints_count + len(feature_keys)  # track_id + frame_idx + keypoints + features
    
    print(f"Saved flat numpy array with id, frame, keypoints, features to {flat_npy_path}")
    print(f"Each row contains: 1 track_id + 1 frame_idx + {keypoints_count} keypoint values + {len(feature_keys)} features = {total_values_per_row} total values")

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
    
    # Only initialize DeepSORT if we're using it (to save memory)
    deepsort_tracker = None
    if args.use_deepsort:
        print("Using DeepSORT for tracking")
        deepsort_tracker = DeepSORTTracker(device=device)
    else:
        print("Using YOLO's built-in tracking")
    
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
        args.headless or args.output_video
    )
    
    # Track state
    keypoints_history = defaultdict(lambda: [])
    HISTORY_LENGTH = 5
    bbox_info = defaultdict(list)
    
    def process_frame(frame, frame_count, fps):
        """Process a single video frame"""
        # Draw FPS counter
        visualizer.draw_fps(frame, fps)
        
        # IMPORTANT: Get detections using the correct tracker
        if args.use_deepsort:
            # First get detections from YOLO
            yolo_detections = yolo_tracker.update(frame)
            # Then feed them to DeepSORT
            detections = deepsort_tracker.update(frame, yolo_detections)
        else:
            # Use YOLO's built-in tracking
            detections = yolo_tracker.update(frame)
        
        # Process each detection
        for detection in detections:
            track_id = detection.track_id
            
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

if __name__ == "__main__":
    main()