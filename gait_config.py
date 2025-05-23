"""
Configuration settings for gait recognition system
"""

from dataclasses import dataclass
import argparse
import torch
import os

@dataclass
class GaitConfig:
    """Configuration class for gait recognition system"""
    # Input/Output paths
    video: str = "../Person_New/input/3c.mp4"
    start_frame: int = 150
    end_frame: int = 2000
    output_features: str = "industrial_gait_features.csv"
    model: str = "gait_validation_results/gait_classifier_model.pkl"
    output_video: str = ""
    results_dir: str = "results"
    
    # Model settings
    use_transreid: bool = True
    transreid_model: str = "model/transreid_vitbase.pth"
    
    # Tracking settings
    tracking_iou: float = 0.5
    tracking_age: int = 30
    
    # Processing settings
    identify: bool = False
    headless: bool = False
    buffer_size: float = 0.05
    save_bbox_info: bool = False
    merge_ids: bool = False
    
    # System settings
    device: str = None
    pose_device: str = None
    
    def __post_init__(self):
        """Initialize device settings after initialization"""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Always use CPU for pose on Mac due to MPS bug
        if self.pose_device is None:
            self.pose_device = 'cpu' if torch.backends.mps.is_available() else self.device
    
    def update_paths(self):
        """Update paths based on results directory"""
        base_features_name = os.path.basename(self.output_features)
        self.features_path = os.path.join(self.results_dir, base_features_name)
        self.flat_npy_path = self.features_path.replace('.csv', '_flat.npy')
        self.bbox_json_path = os.path.join(self.results_dir, "bbox_info.json")
        self.feature_order_path = self.features_path.replace('.csv', '_feature_order.txt')
        self.processed_csv_path = self.features_path.replace('.csv', '_processed.csv')
        self.inv_features_csv = os.path.join(self.results_dir, 'industrial_gait_features_with_invariants.csv')
        
        # Configure output video path
        if self.output_video:
            self.output_video_path = os.path.join(self.results_dir, os.path.basename(self.output_video))
        else:
            self.output_video_path = None

def parse_args():
    """Parse command line arguments and return a GaitConfig object"""
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
    parser.add_argument("--merge_ids", action="store_true", default=False,
                    help="Run ID merger to merge incorrectly split tracking IDs")
    
    args = parser.parse_args()
    
    # Convert to GaitConfig object
    config = GaitConfig(
        video=args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        output_features=args.output_features,
        model=args.model,
        identify=args.identify,
        output_video=args.output_video,
        headless=args.headless,
        buffer_size=args.buffer_size,
        save_bbox_info=args.save_bbox_info,
        results_dir=args.results_dir,
        use_transreid=args.use_transreid,
        transreid_model=args.transreid_model,
        tracking_iou=args.tracking_iou,
        tracking_age=args.tracking_age,
        merge_ids=args.merge_ids
    )
    
    # Update paths based on results directory
    config.update_paths()
    
    return config
