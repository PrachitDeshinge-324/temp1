#!/usr/bin/env python3
# filepath: /Users/prachit/self/Working/Person_Temp/config/cli_args.py
"""
Command line argument parsing for gait analysis
"""
import argparse

def parse_args():
    """Parse command line arguments for gait analysis"""
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
