"""
Simple Gait Analysis Main Script

Minimal orchestrator that coordinates detection, pose analysis, and data processing.
Clean separation of concerns with just the essential workflow.
"""

import argparse
import os
import torch

# Import our modular components
from detection_tracking import DetectionTracker
from pose_analysis import PoseAnalyzer
from data_processing import DataProcessor

# Import utilities
from utils.visualization import Visualizer
from utils.video_processor import VideoProcessor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gait Analysis with Modular Architecture")
    
    # Input/Output
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file (optional)")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory to save all output files")
    
    # Model settings
    parser.add_argument("--use_transreid", action="store_true", default=True,
                       help="Use TransReID for person tracking")
    parser.add_argument("--transreid_model", type=str, default="model/transreid_vitbase.pth",
                       help="Path to TransReID model weights")
    
    # Tracking parameters
    parser.add_argument("--tracking_iou", type=float, default=0.5,
                       help="IoU threshold for tracking association")
    parser.add_argument("--tracking_age", type=int, default=30,
                       help="Maximum age for tracks before deletion")
    
    # Processing options
    parser.add_argument("--buffer_size", type=float, default=0.05,
                       help="Buffer size ratio around detected person")
    parser.add_argument("--save_bbox_info", action="store_true", default=False,
                       help="Save bounding box information to JSON")
    
    # Post-processing
    parser.add_argument("--merge_ids", action="store_true", default=False,
                       help="Run interactive ID merging after processing")
    
    # Display options
    parser.add_argument("--display", action="store_true", default=False,
                       help="Display video during processing")
    parser.add_argument("--save_video", action="store_true", default=False,
                       help="Save processed video with visualizations")
    
    return parser.parse_args()

def get_device_config():
    """Configure devices for optimal performance"""
    if torch.cuda.is_available():
        device = 'cuda'
        pose_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        pose_device = 'cpu'  # MPS might not support pose detection well
    else:
        device = 'cpu'
        pose_device = 'cpu'
    
    print(f"Using device: {device}, Pose device: {pose_device}")
    return device, pose_device

def main():
    """Main processing pipeline"""
    args = parse_args()
    
    # Setup devices
    device, pose_device = get_device_config()
    
    # Initialize modular components
    print("Initializing components...")
    
    # 1. Detection and Tracking
    detector_tracker = DetectionTracker(
        device=device,
        pose_device=pose_device,
        use_transreid=args.use_transreid,
        transreid_model=args.transreid_model,
        tracking_iou=args.tracking_iou,
        tracking_age=args.tracking_age
    )
    
    # 2. Pose Analysis
    pose_analyzer = PoseAnalyzer(
        model_pose=detector_tracker.model_pose,
        pose_device=pose_device,
        history_length=5
    )
    
    # 3. Data Processing
    data_processor = DataProcessor()
    
    # 4. Visualization and Video Processing
    visualizer = Visualizer()
    video_processor = VideoProcessor(
        args.video,
        output_path=args.output if args.save_video else None,
        headless=not args.display  # headless is opposite of display
    )
    
    print("Starting video processing...")
    
    # Main processing loop
    def process_frame(frame, frame_count, fps):
        """Process each frame of the video"""
        
        # 1. Detect and track people
        tracked_detections = detector_tracker.detect_and_track(frame)
        
        # 2. Process poses for each detection
        for detection in tracked_detections:
            if not hasattr(detection, 'track_id') or detection.track_id <= 0:
                continue
            
            # Collect bounding box info
            data_processor.collect_bbox_info(detection.track_id, detection.bbox, frame_count)
            
            # Process pose
            pose_results = pose_analyzer.process_pose(frame, detection, frame_count)
            
            if pose_results:
                # Visualize results
                color = visualizer.get_color(detection.track_id)
                
                for pose_result in pose_results:
                    keypoints = pose_result['keypoints']
                    crop_offset = pose_result['crop_offset']
                    identity = pose_result['identity']
                    confidence = pose_result['confidence']
                    
                    # Draw bounding box and ID
                    visualizer.draw_bbox(frame, detection.bbox, detection.track_id, color)
                    
                    # Draw identity if available
                    if identity:
                        visualizer.draw_identity(frame, detection, identity, confidence, color)
                    
                    # Draw keypoints and skeleton
                    visualizer.draw_keypoints(frame, keypoints, crop_offset[0], crop_offset[1], color)
        
        return frame
    
    # Process the video
    video_processor.process_video(process_frame)
    
    print("Video processing complete. Exporting data...")
    
    # 3. Export all collected data
    paths = data_processor.export_complete_dataset(pose_analyzer.gait_analyzer, args)
    
    # 4. Run ID merger if requested
    if args.merge_ids:
        print("\nStarting ID merging...")
        merged_ids, id_to_name = data_processor.merge_ids(args, paths['flat_npy'])
        
        if merged_ids:
            print(f"Successfully merged {len(merged_ids)} IDs")
        if id_to_name:
            print(f"Assigned names to {len(id_to_name)} unique persons")
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to: {args.results_dir}")
    print(f"Main features: {paths['features_csv']}")
    print(f"Enhanced features: {paths['invariants_csv']}")
    
    # Summary statistics
    track_history = pose_analyzer.get_track_history()
    valid_tracks = [tid for tid in track_history.keys() if tid > 0]
    
    print(f"\nSummary:")
    print(f"- Total valid tracks: {len(valid_tracks)}")
    print(f"- Track IDs: {sorted(valid_tracks)}")
    
    for track_id in sorted(valid_tracks):
        frames_count = len(track_history[track_id])
        gait_features = pose_analyzer.get_gait_features(track_id)
        feature_count = len(gait_features) if gait_features else 0
        print(f"  Track {track_id}: {frames_count} frames, {feature_count} features")

if __name__ == "__main__":
    main()