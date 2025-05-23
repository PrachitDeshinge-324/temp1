#!/usr/bin/env python3
# filepath: /Users/prachit/self/Working/Person_Temp/gait_analysis.py
"""
Gait Analysis Main Orchestrator

This module serves as the main entry point for gait analysis,
coordinating different components like detection, tracking,
feature extraction, and identification.
"""
import os
import torch
import numpy as np
from collections import defaultdict
import sys

# Import models
from ultralytics import YOLO

# Import custom modules
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator
from utils.trackers import YOLOTracker, Detection
from utils.pose_detector import PoseDetector
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor

# Import trackers
from trackers.transreid_tracker import TransReIDTracker

# Import data processing components
from processors.data_exporter import GaitDataExporter
from processors.id_merger import IdMerger

# Import configuration
from config.cli_args import parse_args

class GaitAnalysisOrchestrator:
    """Main orchestrator for gait analysis"""
    
    def __init__(self, args):
        """Initialize the orchestrator with command line args"""
        self.args = args
        self.device = self._configure_device()
        self.bbox_info = defaultdict(list)
        self.keypoints_history = defaultdict(lambda: [])
        self.HISTORY_LENGTH = 5
        
        # Create results directory
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Update file paths
        self._configure_paths()
        
        # Initialize components
        self._initialize_models()
        self._initialize_trackers()
        self._initialize_analyzers()
        
    def _configure_device(self):
        """Configure device settings for optimal performance"""
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # Always use CPU for pose on Mac due to MPS bug
        self.pose_device = 'cpu' if torch.backends.mps.is_available() else device
        print(f"Using device: {device}, Pose device: {self.pose_device}")
        return device
        
    def _configure_paths(self):
        """Configure output file paths"""
        base_features_name = os.path.basename(self.args.output_features)
        self.features_path = os.path.join(self.args.results_dir, base_features_name)
        self.flat_npy_path = self.features_path.replace('.csv', '_flat.npy')
        self.bbox_json_path = os.path.join(self.args.results_dir, "bbox_info.json")
        self.feature_order_path = self.features_path.replace('.csv', '_feature_order.txt')
        
        # Configure output video path
        if self.args.output_video:
            self.output_video_path = os.path.join(self.args.results_dir, os.path.basename(self.args.output_video))
        else:
            self.output_video_path = None
            
    def _initialize_models(self):
        """Initialize YOLO and pose models"""
        self.model_det = YOLO('model/yolo11x.pt').to(self.device)
        self.model_pose = YOLO('model/yolo11x-pose.pt').to(self.pose_device)
        
    def _initialize_trackers(self):
        """Initialize tracking components"""
        self.yolo_tracker = YOLOTracker(self.model_det, device=self.device)
        
        # Initialize TransReID tracker if enabled
        self.transreid_tracker = None
        if self.args.use_transreid:
            print("Using TransReID for robust re-identification and tracking")
            self.transreid_tracker = TransReIDTracker(
                device=self.device, 
                reid_model_path=self.args.transreid_model,
                iou_threshold=self.args.tracking_iou,
                max_age=self.args.tracking_age
            )
        else:
            print("Using YOLO's built-in tracking (not recommended)")
            
    def _initialize_analyzers(self):
        """Initialize analysis components"""
        # Initialize pose detector
        self.pose_detector = PoseDetector(self.model_pose, device=self.pose_device)
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Initialize gait analyzer
        self.gait_analyzer = GaitFeatureExtractor()
        
        # Initialize person identifier if requested
        self.person_identifier = None
        if self.args.identify and os.path.exists(self.args.model):
            self.person_identifier = GaitValidator()
            if self.person_identifier.load_model(self.args.model):
                print("Loaded gait classifier for person identification")
            else:
                self.person_identifier = None
                
        # Initialize video processor
        self.video_processor = VideoProcessor(
            self.args.video, 
            self.args.start_frame,
            self.args.end_frame,
            self.output_video_path,
            self.args.headless
        )
    
    def process_frame(self, frame, frame_count, fps):
        """Process a single video frame"""
        # Draw FPS counter
        self.visualizer.draw_fps(frame, fps)
        
        # Get detections using the correct tracker
        if self.args.use_transreid:
            # First get detections from YOLO
            yolo_detections = self.yolo_tracker.update(frame)
            # Then feed them to TransReID tracker
            detections = self.transreid_tracker.update(frame, yolo_detections)
            
            # Extra validation to ensure no invalid IDs pass through
            validated_detections = []
            for det in detections:
                # Ensure track_id is a positive integer (not 0, not negative)
                if not hasattr(det, 'track_id') or det.track_id <= 0:
                    # Generate new valid ID for this detection
                    new_id = self.transreid_tracker.next_id
                    det.track_id = new_id
                    self.transreid_tracker.next_id = new_id + 1
                validated_detections.append(det)
            detections = validated_detections
        else:
            # Use YOLO's built-in tracking
            detections = self.yolo_tracker.update(frame)
            
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
            buffered_box, color = self.visualizer.draw_detection(
                frame, detection, self.args.buffer_size)
            x1_buf, y1_buf, x2_buf, y2_buf = buffered_box
            
            # Save bounding box info if requested
            if self.args.save_bbox_info:
                self.bbox_info[int(track_id)].append({
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
            keypoints_list = self.pose_detector.detect(person_crop)
            
            for keypoints in keypoints_list:
                # Update keypoints history for smoothing
                if len(self.keypoints_history[track_id]) >= self.HISTORY_LENGTH:
                    self.keypoints_history[track_id].pop(0)
                self.keypoints_history[track_id].append(keypoints.copy())
                
                # Apply temporal smoothing to keypoints
                smoothed_keypoints = self.pose_detector.smooth_keypoints(
                    self.keypoints_history[track_id][:-1], keypoints)
                
                # Update gait analyzer with new keypoints
                self.gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
                
                # Perform identification if requested
                if self.person_identifier and frame_count % 15 == 0:
                    feature_vector = self.gait_analyzer.get_feature_vector(track_id)
                    if feature_vector is not None:
                        identity, confidence = self.person_identifier.identify_person(feature_vector)
                        self.visualizer.draw_identity(frame, detection, identity, confidence, color)
                
                # Draw keypoints and skeleton
                self.visualizer.draw_keypoints(frame, smoothed_keypoints, x1_buf, y1_buf, color)
        
        return frame
        
    def run(self):
        """Run the full gait analysis pipeline"""
        print("Starting gait analysis...")
        
        # Process the video
        self.video_processor.process_video(self.process_frame)
        
        # Export collected data
        print("Exporting data...")
        GaitDataExporter.export_all_data(
            self.gait_analyzer, 
            self.args, 
            self.bbox_info, 
            self.features_path, 
            self.bbox_json_path, 
            self.feature_order_path, 
            self.flat_npy_path
        )
        
        # Run ID merger if requested
        if self.args.merge_ids:
            print("Running ID merger...")
            merged_ids, id_to_name = IdMerger.merge_ids_interactive(
                self.args.video, 
                self.bbox_json_path, 
                self.flat_npy_path,
                self.args.results_dir
            )
            print("ID merging complete!")
            if merged_ids:
                print(f"Merged {len(merged_ids)} IDs")
            if id_to_name:
                print(f"Assigned names to {len(id_to_name)} unique persons")
    
        # Generate clean CSV from processed numpy data
        print("Generating processed CSVs...")
        processed_csv_path = self.features_path.replace('.csv', '_processed.csv')
        GaitDataExporter.generate_processed_csv(
            self.flat_npy_path, 
            processed_csv_path, 
            self.feature_order_path
        )
        
        # Fix invariant features in a new CSV
        inv_features_csv = os.path.join(self.args.results_dir, 'industrial_gait_features_with_invariants.csv')
        GaitDataExporter.fix_invariant_features(
            self.gait_analyzer, 
            self.flat_npy_path, 
            self.feature_order_path, 
            inv_features_csv
        )
        print(f"Created enriched CSV with invariant features at {inv_features_csv}")
        print("Gait analysis complete!")

def main():
    """Main entry point for gait analysis"""
    args = parse_args()
    orchestrator = GaitAnalysisOrchestrator(args)
    orchestrator.run()

if __name__ == "__main__":
    main()
