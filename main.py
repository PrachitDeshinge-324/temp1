import argparse
import os
import cv2
import csv
import numpy as np
import torch
import logging
import traceback
from tqdm import tqdm
from ultralytics import YOLO
from utils.helper import get_best_device
from utils.transreid import TransReIDModel
from utils.visualize import draw_person_detection, generate_colors, create_video_writer
from utils.reid_tracker import ReIDEnhancedTracker
from utils.opengait_model import OpenGaitModel
from utils.silhouette_processor import SilhouetteProcessor
from utils.gait_gallery import GaitGallery

# Configure logging to show only necessary information
def setup_logging():
    """Configure logging for the application with minimal terminal output."""
    # Set up main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('torchvision').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Suppress OpenGait internal logging
    logging.getLogger('opengait').setLevel(logging.WARNING)
    
    # Create application logger
    logger = logging.getLogger('gait_analysis')
    logger.setLevel(logging.INFO)
    return logger

def assess_silhouette_quality(silhouettes):
    """
    Assess the quality and completeness of a batch of silhouettes.
    
    Args:
        silhouettes: List of silhouette images (numpy arrays)
    
    Returns:
        dict: Quality metrics including completeness score, consistency, and coverage
    """
    if not silhouettes or len(silhouettes) == 0:
        return {"completeness": 0.0, "consistency": 0.0, "coverage": 0.0, "is_complete": False}
    
    # Calculate pixel coverage metrics
    coverage_scores = []
    consistency_scores = []
    
    # Standard expected silhouette area (relative to image size)
    expected_min_coverage = 0.15  # Minimum 15% of image should be person pixels
    expected_max_coverage = 0.45  # Maximum 45% to avoid noise
    
    for silhouette in silhouettes:
        if silhouette is None or silhouette.size == 0:
            coverage_scores.append(0.0)
            continue
            
        # Calculate pixel coverage
        total_pixels = silhouette.shape[0] * silhouette.shape[1]
        person_pixels = np.sum(silhouette > 128)  # Binary threshold
        coverage_ratio = person_pixels / total_pixels
        
        # Score based on whether coverage is in expected range
        if expected_min_coverage <= coverage_ratio <= expected_max_coverage:
            coverage_score = 1.0
        elif coverage_ratio < expected_min_coverage:
            coverage_score = coverage_ratio / expected_min_coverage
        else:
            coverage_score = max(0.0, 1.0 - (coverage_ratio - expected_max_coverage) / 0.2)
        
        coverage_scores.append(coverage_score)
    
    # Calculate consistency between consecutive silhouettes
    if len(silhouettes) > 1:
        # Find common dimensions for all silhouettes
        target_height = 128  # Standard height
        target_width = 64   # Standard width for consistency
        
        for i in range(1, len(silhouettes)):
            if silhouettes[i-1] is not None and silhouettes[i] is not None:
                try:
                    # Resize both silhouettes to same dimensions for comparison
                    prev_sil = cv2.resize(silhouettes[i-1], (target_width, target_height))
                    curr_sil = cv2.resize(silhouettes[i], (target_width, target_height))
                    
                    # Convert to float and normalize
                    prev_sil = prev_sil.astype(np.float32) / 255.0
                    curr_sil = curr_sil.astype(np.float32) / 255.0
                    
                    # Calculate overlap and size consistency
                    intersection = np.sum((prev_sil > 0.5) & (curr_sil > 0.5))
                    union = np.sum((prev_sil > 0.5) | (curr_sil > 0.5))
                    
                    if union > 0:
                        iou = intersection / union
                        consistency_scores.append(iou)
                    else:
                        consistency_scores.append(0.0)
                except Exception as e:
                    # Skip problematic silhouettes
                    consistency_scores.append(0.0)
    
    # Calculate overall metrics
    avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    
    # Completeness combines coverage quality and temporal consistency
    completeness = (avg_coverage * 0.7 + avg_consistency * 0.3)
    
    # Determine if this is a "complete" dataset
    is_complete = (completeness >= 0.85 and 
                   len(silhouettes) >= 30 and 
                   avg_coverage >= 0.8)
    
    return {
        "completeness": completeness,
        "consistency": avg_consistency,
        "coverage": avg_coverage,
        "is_complete": is_complete,
        "frame_count": len(silhouettes),
        "valid_frames": len([s for s in coverage_scores if s > 0.1])
    }

def create_gait_energy_image(silhouettes, output_path):
    """
    Create a Gait Energy Image (GEI) from a sequence of silhouettes.
    
    Args:
        silhouettes: List of silhouette images (numpy arrays)
        output_path: Path to save the GEI
    """
    if not silhouettes:
        return None
    
    # Find maximum dimensions
    max_height = max([s.shape[0] for s in silhouettes])
    max_width = max([s.shape[1] for s in silhouettes])
    
    # Resize all silhouettes to same dimensions
    resized_silhouettes = []
    for silhouette in silhouettes:
        resized = cv2.resize(silhouette, (max_width, max_height))
        resized_silhouettes.append(resized)
    
    # Create GEI by averaging silhouettes
    gei = np.zeros((max_height, max_width), dtype=np.float32)
    for silhouette in resized_silhouettes:
        gei += silhouette
    
    gei = gei / len(silhouettes)
    gei = (gei * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, gei)
    return gei

def get_arguments():
    parser = argparse.ArgumentParser(description="Process video for person detection, tracking and segmentation.")
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder path')
    parser.add_argument('--weights_dir', type=str, required=True, help='Weights folder path')
    parser.add_argument('--transreid_weights', type=str, default=None, help='TransReID weights path')
    parser.add_argument('--device', type=str, default=get_best_device(), help='Device to run the model on (cpu or cuda)')
    parser.add_argument('--output_video', action='store_true', help='Generate output video with annotations')
    parser.add_argument('--display', action='store_true', help='Display video in real-time using cv2.imshow')
    parser.add_argument('--track_conf', type=float, default=0.3, help='Tracking confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU threshold for tracking')
    parser.add_argument('--save_crops', action='store_true', help='Save segmented person crops')
    parser.add_argument('--gait_analysis', action='store_true', help='Generate gait analysis data')
    parser.add_argument('--reid_similarity', type=float, default=0.7, help='Similarity threshold for ReID (0.0-1.0)')
    parser.add_argument('--opengait_weights', type=str, default="", help='Path to OpenGait model weights')
    parser.add_argument('--opengait_config', type=str, default="OpenGait/configs/default.yaml", help='Path to OpenGait model configuration')
    parser.add_argument('--gait_gallery', type=str, default="gait_gallery.pkl", help='Path to gait embedding gallery')
    parser.add_argument('--gait_threshold', type=float, default=0.98, help='Threshold for gait matching (0.0-1.0)')
    parser.add_argument('--build_gallery', action='store_true', help='Build or update the gait embedding gallery')
    parser.add_argument('--save_silhouettes', action='store_true', help='Save silhouette images during gait analysis')
    parser.add_argument('--force_new_identities', action='store_true',
                        help='Force creation of new identities even with matches')
    parser.add_argument('--clear_gallery', action='store_true',
                        help='Clear existing gallery and start fresh')
    parser.add_argument('--gallery_build_frames', type=int, default=200,
                        help='Number of frames to use for initial gallery building')
    parser.add_argument('--prevent_identity_conflicts', action='store_true',
                        help='Prevent multiple tracks in same frame having same identity')

    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.weights_dir):
        raise FileNotFoundError(f"Weights folder {args.weights_dir} does not exist.")
    return args

def main():
    # Setup logging first
    logger = setup_logging()
    
    args = get_arguments()
    
    # Log input parameters
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Weights directory: {args.weights_dir}")
    if args.opengait_weights:
        logger.info(f"OpenGait weights: {args.opengait_weights}")
    
    video_path = args.input
    output_dir = args.output_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    logger.info(f"Processing video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video stats: {frame_count} frames, {frame_width}x{frame_height}, {fps:.1f} FPS")

    # Always initialize video writer to save output video
    output_video_path = os.path.join(output_dir, "output_with_detections.mp4")
    video_writer = create_video_writer(output_video_path, fps, frame_width, frame_height)
    logger.info(f"Output video: {output_video_path}")
    
    # Create directory for segmented crops and silhouettes
    seg_crops_dir = os.path.join(output_dir, "segmented_crops")
    silhouette_dir = os.path.join(output_dir, "silhouettes")
    gait_dir = os.path.join(output_dir, "gait_analysis")
    
    if args.save_crops and not os.path.exists(seg_crops_dir):
        os.makedirs(seg_crops_dir)
        logger.info(f"Segmented crops will be saved to: {seg_crops_dir}")
    
    if args.gait_analysis:
        if not os.path.exists(silhouette_dir):
            os.makedirs(silhouette_dir)
        if not os.path.exists(gait_dir):
            os.makedirs(gait_dir)
        logger.info(f"Gait analysis data will be saved to: {gait_dir}")
    
    if args.display:
        logger.info("Real-time display enabled - press 'q' to quit, 'p' to pause/resume")
        cv2.namedWindow('Person Detection & Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Person Detection & Tracking', 960, 540)  # Resize for better viewing

    # Generate colors for person IDs
    person_colors = generate_colors(50)  # Generate 50 distinct colors

    # Initialize YOLO models for detection and segmentation
    logger.info("Loading YOLO models...")
    person_detector = YOLO(os.path.join(args.weights_dir, "yolo11x.pt"))  # For person detection
    segmentation_model = YOLO(os.path.join(args.weights_dir, "yolo11x-seg.pt"))  # For person segmentation
    logger.info("YOLO models loaded successfully")
    
    # Initialize TransReID model for feature extraction (optional)
    transreid_model = None
    use_reid_features = False
    reid_tracker = None
    
    if args.transreid_weights:
        try:
            logger.info("Loading TransReID model...")
            transreid_model = TransReIDModel(args.transreid_weights, device=args.device)
            logger.info("TransReID model loaded successfully")
            use_reid_features = True
            
            # Initialize ReID tracker with TransReID model
            reid_tracker = ReIDEnhancedTracker(
                transreid_model=transreid_model,
                similarity_threshold=args.reid_similarity,
                feature_history_size=10,
                reid_memory_frames=30
            )
            logger.info(f"ReID tracker initialized (threshold: {args.reid_similarity})")
            
        except Exception as e:
            logger.warning(f"Could not load TransReID model: {e}")
            logger.info("Proceeding without ReID feature extraction")

    # Initialize OpenGait components if requested
    opengait_model = None
    silhouette_processor = None
    gait_gallery = None

    if args.opengait_weights:
        try:
            logger.info("Loading OpenGait model...")
            # Use the config path from command-line arguments
            opengait_model = OpenGaitModel(args.opengait_weights, args.opengait_config, device=args.device)
            silhouette_processor = SilhouetteProcessor()
            
            # Initialize gait gallery
            gait_gallery = GaitGallery(args.gait_gallery)
            
            # Clear gallery if requested
            if args.clear_gallery and gait_gallery:
                logger.info("Clearing existing gallery")
                gait_gallery.gallery = {}
                gait_gallery.next_id = 1
            
            # Print gallery statistics
            if gait_gallery and hasattr(gait_gallery, 'gallery_stats'):
                stats = gait_gallery.gallery_stats()
                logger.info(f"Gallery stats: {stats['total_identities']} identities, {stats['total_embeddings']} embeddings")
                if stats['embedding_dimensions']:
                    logger.info(f"Embedding dimensions: {stats['embedding_dimensions']}")
            else:
                logger.info("OpenGait model loaded successfully")
        except Exception as e:
            logger.error(f"Could not load OpenGait model: {e}")
            import traceback
            traceback.print_exc()  # Keep detailed error for debugging
            opengait_model = None
            silhouette_processor = None
            gait_gallery = None

    # Configure tracking parameters
    tracking_config = {
        'tracker_type': 'bytetrack',  # Use ByteTrack algorithm
        'track_high_thresh': args.track_conf,  # High confidence threshold 
        'track_low_thresh': args.track_conf * 0.5,  # Low confidence threshold (half of high)
        'new_track_thresh': args.track_conf,  # New tracks threshold
        'track_buffer': 30,  # How many frames to keep track of disappearing objects
        'match_thresh': args.iou_thresh  # IOU matching threshold
    }
    
    logger.info(f"ByteTrack config: conf={args.track_conf}, iou={args.iou_thresh}")
    
    # Store silhouettes for gait analysis
    person_silhouettes = {}
    
    # Optimized embedding management variables 
    track_frame_buffer = {}      # person_id -> list of silhouettes (buffering ~100 frames)
    track_embeddings = {}        # person_id -> final aggregated embedding
    track_last_processed = {}    # person_id -> frame when last processed
    track_processed_count = {}   # person_id -> number of times processed
    embedding_batch_size = 100   # Target frames to collect before processing
    min_frames_for_embedding = 30  # Minimum frames needed for embedding generation
    min_frames_between_updates = 50  # Minimum frames between re-processing same track
    
    # Enhanced dataset quality tracking for periodic updates
    track_silhouette_quality = {}    # person_id -> dict with quality metrics
    track_completeness_history = {}  # person_id -> list of completeness scores per batch
    track_last_complete_batch = {}   # person_id -> frame index of last complete batch
    dataset_quality_threshold = 0.85  # Minimum completeness score for "complete" dataset
    periodic_update_interval = 300    # Frames between periodic quality assessments
    min_batches_for_comparison = 2    # Minimum batches needed before quality comparison
    
    csv_path = os.path.join(output_dir, "detections.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Headers with segmentation info
        headers = ['frame', 'person_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'seg_file']
        writer.writerow(headers)

        frame_index = 0
        paused = False
        
        # Create track history for visualization
        track_history = {}
        
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO tracking (built-in ByteTrack)
                tracking_results = person_detector.track(
                    source=frame,
                    persist=True,      # Maintain tracks between frames
                    verbose=False,
                    classes=0,         # Only track persons (class 0)
                    tracker="bytetrack.yaml",  # Use ByteTrack algorithm
                    conf=args.track_conf,      # Detection confidence threshold
                    iou=args.iou_thresh       # IOU threshold for tracking
                )
                
                # Process tracking results
                if len(tracking_results) > 0:
                    detections = tracking_results[0]
                    
                    # Extract bounding boxes and person IDs
                    if len(detections.boxes) > 0:
                        boxes = detections.boxes.xyxy.cpu().numpy()
                        confidences = detections.boxes.conf.cpu().numpy()
                        person_ids = detections.boxes.id.int().cpu().numpy() if detections.boxes.id is not None else np.arange(len(boxes))
                        
                        # Apply ReID verification to refine person IDs if available
                        if reid_tracker is not None:
                            # Get corrected IDs from ReID tracker
                            person_ids = reid_tracker.update(frame, boxes, person_ids, confidences)
                        
                        # Create person crops for processing
                        person_crops = []
                        valid_detections = []
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)
                            # Ensure box is within image boundaries
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            
                            # Skip invalid boxes
                            if x1 >= x2 or y1 >= y2:
                                continue
                            
                            crop = frame[y1:y2, x1:x2]
                            person_crops.append(crop)
                            valid_detections.append(i)
                            
                            # Update tracks history for visualization
                            track_id = int(person_ids[i])
                            if track_id not in track_history:
                                track_history[track_id] = []
                            track_history[track_id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                            # Keep only last 30 frames
                            if len(track_history[track_id]) > 30:
                                track_history[track_id] = track_history[track_id][-30:]
                        
                        # Process crops with segmentation model
                        segmentation_results = []
                        segmentation_files = []
                        
                        for idx, crop in enumerate(person_crops):
                            # Apply segmentation model to crop
                            seg_results = segmentation_model(crop, verbose=False)
                            segmentation_results.append(seg_results)
                            
                            # Get person ID for this crop
                            person_id = int(person_ids[valid_detections[idx]])
                            
                            # Process segmentation results
                            if len(seg_results) > 0 and seg_results[0].masks is not None:
                                # Get the segmentation mask
                                mask = seg_results[0].masks.data[0].cpu().numpy()
                                
                                # Get the dimensions of the crop
                                crop_height, crop_width = crop.shape[:2]
                                
                                # Handle different mask dimensions
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                
                                # Resize mask to match crop size
                                mask = cv2.resize(mask, (crop_width, crop_height))
                                
                                # Ensure mask values are in proper range [0, 1]
                                if mask.max() > 1.0:
                                    mask = mask / mask.max()
                                
                                # Save standard segmented crop if requested
                                if args.save_crops:
                                    seg_file = f"frame_{frame_index:06d}_person_{person_id:03d}.png"
                                    seg_path = os.path.join(seg_crops_dir, seg_file)
                                    
                                    # Create masked crop
                                    mask_3ch = np.stack([mask] * 3, axis=2)
                                    masked_crop = (crop * mask_3ch).astype(np.uint8)
                                    cv2.imwrite(seg_path, masked_crop)
                                    segmentation_files.append(seg_file)
                                
                                # Process gait analysis data if enabled
                                if args.gait_analysis:
                                    # Convert to binary silhouette (0 or 255)
                                    silhouette = np.zeros((crop_height, crop_width), dtype=np.uint8)
                                    silhouette[mask > 0.5] = 255
                                    
                                    # Apply morphological operations to clean the silhouette
                                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
                                    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel)
                                    
                                    # Save binary silhouette
                                    # sil_file = f"silhouette_{frame_index:06d}_id_{person_id:03d}.png"
                                    # sil_path = os.path.join(silhouette_dir, sil_file)
                                    # cv2.imwrite(sil_path, silhouette)
                                    
                                    # Standard height for gait analysis is often 128 pixels
                                    standard_height = 128
                                    aspect_ratio = crop_width / crop_height
                                    standard_width = int(standard_height * aspect_ratio)
                                    normalized_silhouette = cv2.resize(silhouette, (standard_width, standard_height))
                                    
                                    # Always initialize the person_id key if it doesn't exist
                                    if person_id not in person_silhouettes:
                                        person_silhouettes[person_id] = []
                                        
                                    # Store normalized silhouette for this frame 
                                    person_silhouettes[person_id].append(normalized_silhouette)
                            else:
                                if args.save_crops:
                                    # No mask found, save original crop
                                    seg_file = f"frame_{frame_index:06d}_person_{person_id:03d}_no_mask.png"
                                    seg_path = os.path.join(seg_crops_dir, seg_file)
                                    cv2.imwrite(seg_path, crop)
                                    segmentation_files.append(seg_file)
                                else:
                                    segmentation_files.append("")
                        
                        # Create output frame for visualization
                        output_frame = frame.copy()
                        
                        # Process each detection
                        for idx, i in enumerate(valid_detections):
                            box = boxes[i]
                            x1, y1, x2, y2 = map(int, box)
                            person_id = int(person_ids[i])
                            confidence = confidences[i]
                            
                            # Get color for this person ID
                            color = person_colors[person_id % len(person_colors)]
                            
                            # Get identity ID if available from gait gallery
                            identity_id = None
                            if gait_gallery and hasattr(gait_gallery, 'track_to_identity') and person_id in gait_gallery.track_to_identity:
                                identity_id = gait_gallery.track_to_identity[person_id]
                            
                            # Draw bounding box and person ID with identity information
                            draw_person_detection(output_frame, (x1, y1, x2, y2), person_id, color, identity_id=identity_id)
                            
                            # Draw track history
                            if person_id in track_history:
                                for point in track_history[person_id]:
                                    cv2.circle(output_frame, point, 1, color, -1)
                                
                                # Connect points with lines
                                points = np.array(track_history[person_id], dtype=np.int32)
                                if len(points) > 1:
                                    cv2.polylines(output_frame, [points], False, color, 1)
                            
                            # Draw segmentation mask on main frame if available
                            if idx < len(segmentation_results) and len(segmentation_results[idx]) > 0:
                                seg_result = segmentation_results[idx][0]
                                if hasattr(seg_result, 'masks') and seg_result.masks is not None:
                                    # Get first mask (assuming one person per crop)
                                    mask = seg_result.masks.data[0].cpu().numpy()
                                    
                                    # Handle different mask dimensions
                                    if len(mask.shape) == 3:
                                        mask = mask[0]
                                    
                                    # Resize mask to match crop size
                                    mask = cv2.resize(mask, (x2-x1, y2-y1))
                                    
                                    # Ensure mask values are in proper range [0, 1]
                                    if mask.max() > 1.0:
                                        mask = mask / mask.max()
                                    
                                    # Create color overlay
                                    colored_mask = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                                    colored_mask[..., 0] = mask * color[0] * 0.5  # B channel
                                    colored_mask[..., 1] = mask * color[1] * 0.5  # G channel
                                    colored_mask[..., 2] = mask * color[2] * 0.5  # R channel
                                    
                                    # Apply mask overlay to output frame
                                    roi = output_frame[y1:y2, x1:x2]
                                    mask_binary = (mask > 0).astype(np.uint8)
                                    roi_with_mask = cv2.addWeighted(roi, 1, colored_mask, 0.5, 0)
                                    roi_masked = roi.copy()
                                    roi_masked[mask_binary > 0] = roi_with_mask[mask_binary > 0]
                                    output_frame[y1:y2, x1:x2] = roi_masked
                            
                            # Save to CSV with segmentation file info
                            seg_file = segmentation_files[idx] if idx < len(segmentation_files) else ""
                            writer.writerow([frame_index, person_id, x1, y1, x2, y2, confidence, seg_file])
                                    
                        # Write frame to output video
                        video_writer.write(output_frame)
                        
                        # Optimized gait analysis with batch processing (after all detections processed)
                        if args.gait_analysis and opengait_model and person_silhouettes:
                            try:
                                # Process silhouettes for embedding generation
                                for person_id in person_silhouettes:
                                    # Initialize frame buffer for this person if not exists
                                    if person_id not in track_frame_buffer:
                                        track_frame_buffer[person_id] = []
                                    
                                    # Add current silhouettes to buffer (only new ones from this frame)
                                    current_silhouettes = person_silhouettes[person_id]
                                    if current_silhouettes:
                                        track_frame_buffer[person_id].extend(current_silhouettes)
                                        
                                        # Limit buffer size to prevent memory issues
                                        if len(track_frame_buffer[person_id]) > embedding_batch_size * 2:
                                            track_frame_buffer[person_id] = track_frame_buffer[person_id][-embedding_batch_size:]
                                    
                                    # Check if we should process this track for embedding generation
                                    buffer_size = len(track_frame_buffer[person_id])
                                    
                                    should_process = False
                                    process_reason = ""
                                    
                                    # Condition 1: Enough frames for first-time processing
                                    if person_id not in track_embeddings and buffer_size >= min_frames_for_embedding:
                                        should_process = True
                                        process_reason = "initial_processing"
                                    
                                    # Condition 2: Batch size reached for embedding update  
                                    elif buffer_size >= embedding_batch_size:
                                        # Check if enough time has passed since last processing
                                        if (person_id not in track_last_processed or 
                                            (frame_index - track_last_processed[person_id]) >= min_frames_between_updates):
                                            should_process = True
                                            process_reason = "batch_update"
                                    
                                    # Condition 3: Clear gallery mode - process tracks with sufficient data
                                    elif args.clear_gallery and buffer_size >= min_frames_for_embedding:
                                        should_process = True
                                        process_reason = "clear_gallery_mode"
                                    
                                    if should_process:
                                        print(f"Processing embedding for Track {person_id}: {process_reason} ({buffer_size} frames)")
                                        
                                        # Assess quality of current batch before processing
                                        current_quality = assess_silhouette_quality(track_frame_buffer[person_id])
                                        
                                        # Initialize quality tracking for this person
                                        if person_id not in track_silhouette_quality:
                                            track_silhouette_quality[person_id] = {
                                                "best_quality": current_quality,
                                                "current_quality": current_quality,
                                                "last_assessment": frame_index
                                            }
                                        
                                        if person_id not in track_completeness_history:
                                            track_completeness_history[person_id] = []
                                        
                                        # Add current quality to history
                                        track_completeness_history[person_id].append({
                                            "frame": frame_index,
                                            "quality": current_quality,
                                            "batch_size": buffer_size
                                        })
                                        
                                        # Determine if we should update based on dataset completeness
                                        should_update_embedding = False
                                        update_reason = ""
                                        
                                        # Check if this is first processing or a superior complete dataset
                                        if person_id not in track_embeddings:
                                            should_update_embedding = True
                                            update_reason = "initial_embedding"
                                        else:
                                            # Compare with previous best quality
                                            prev_best = track_silhouette_quality[person_id]["best_quality"]
                                            
                                            # Update if current dataset is significantly more complete
                                            if current_quality["is_complete"] and not prev_best["is_complete"]:
                                                should_update_embedding = True
                                                update_reason = "first_complete_dataset"
                                            elif (current_quality["is_complete"] and prev_best["is_complete"] and 
                                                  current_quality["completeness"] > prev_best["completeness"] + 0.05):
                                                should_update_embedding = True
                                                update_reason = "superior_complete_dataset"
                                            elif (frame_index - track_silhouette_quality[person_id]["last_assessment"] >= periodic_update_interval and
                                                  current_quality["completeness"] > prev_best["completeness"] + 0.1):
                                                should_update_embedding = True
                                                update_reason = "periodic_quality_improvement"
                                        
                                        if should_update_embedding:
                                            logger.info(f"Track {person_id}: {update_reason} (completeness: {current_quality['completeness']:.3f}, frames: {buffer_size})")
                                            
                                            # Prepare sequence for OpenGait (use all buffered frames)
                                            sequence = silhouette_processor.prepare_sequence(track_frame_buffer[person_id])
                                            
                                            # Extract gait embedding (single processing per batch)
                                            gait_embedding = opengait_model.extract_embeddings(sequence)
                                            
                                            if gait_embedding is not None:
                                                # Store the final aggregated embedding (no per-frame writes)
                                                track_embeddings[person_id] = gait_embedding
                                                
                                                # Update quality tracking with successful processing
                                                track_silhouette_quality[person_id]["current_quality"] = current_quality
                                                if current_quality["completeness"] > track_silhouette_quality[person_id]["best_quality"]["completeness"]:
                                                    track_silhouette_quality[person_id]["best_quality"] = current_quality
                                                    track_last_complete_batch[person_id] = frame_index
                                                track_silhouette_quality[person_id]["last_assessment"] = frame_index
                                                
                                                # Update processing history
                                                track_last_processed[person_id] = frame_index
                                                track_processed_count[person_id] = track_processed_count.get(person_id, 0) + 1
                                                
                                                # Database operations - conditional updates only for complete datasets
                                                if gait_gallery:
                                                    # Ensure track_to_identity mapping exists
                                                    if not hasattr(gait_gallery, 'track_to_identity'):
                                                        gait_gallery.track_to_identity = {}
                                                    
                                                    # Check if this track already has an identity (conditional update)
                                                    if person_id in gait_gallery.track_to_identity:
                                                        identity_id = gait_gallery.track_to_identity[person_id] 
                                                        
                                                        # Only update if we have a complete dataset or significant improvement
                                                        if (current_quality["is_complete"] or 
                                                            update_reason in ["superior_complete_dataset", "periodic_quality_improvement"]):
                                                            # Use higher weight for complete datasets
                                                            update_weight = 0.5 if current_quality["is_complete"] else 0.3
                                                            update_success = gait_gallery.update_embedding(identity_id, gait_embedding, weight=update_weight)
                                                            if update_success:
                                                                logger.info(f"Updated Track {person_id} → Identity {identity_id} (weight: {update_weight})")
                                                    else:
                                                        # New track - match or assign identity (single database write)
                                                        identity_id, confidence, is_new = gait_gallery.get_or_assign_identity(
                                                            gait_embedding,
                                                            threshold=args.gait_threshold,
                                                            force_new=args.clear_gallery or (frame_index < 200)
                                                        )
                                                        
                                                        if is_new:
                                                            logger.info(f"New identity: Track {person_id} → Identity {identity_id}")
                                                        else:
                                                            logger.info(f"Match found: Track {person_id} → Identity {identity_id} (conf: {confidence:.2f})")
                                                        
                                                        # Map track ID to identity ID (single mapping write)
                                                        gait_gallery.track_to_identity[person_id] = identity_id
                                                
                                                # Clear processed frames from buffer to save memory (keep overlap for complete datasets)
                                                if process_reason == "batch_update":
                                                    # Keep larger overlap for high-quality datasets
                                                    overlap_size = min(30 if current_quality["is_complete"] else 20, 
                                                                     len(track_frame_buffer[person_id]) // 4)
                                                    track_frame_buffer[person_id] = track_frame_buffer[person_id][-overlap_size:] if overlap_size > 0 else []
                                            else:
                                                logger.warning(f"Failed to extract embedding for Track {person_id}")
                                        else:
                                            # Skip update but log reason for debugging
                                            pass  # Removed verbose logging here
                                
                                # Clear silhouettes for next frame to avoid duplication
                                person_silhouettes.clear()
                                
                                # Periodic gallery save (reduce I/O frequency)
                                if frame_index % 200 == 0 and gait_gallery:
                                    gait_gallery.save_gallery()
                                    logger.info(f"Gallery saved at frame {frame_index}")
                                
                                # Periodic quality monitoring (every 300 frames)
                                if frame_index % periodic_update_interval == 0:
                                    logger.info(f"Quality assessment at frame {frame_index}")
                                    for person_id in track_frame_buffer:
                                        if len(track_frame_buffer[person_id]) >= 20:  # Only assess if enough frames
                                            current_quality = assess_silhouette_quality(track_frame_buffer[person_id])
                                            logger.info(f"  Track {person_id}: {len(track_frame_buffer[person_id])} frames, completeness={current_quality['completeness']:.3f}")
                                    
                            except Exception as e:
                                logger.error(f"Error in gait analysis: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Display frame if requested
                        if args.display:
                            # Handle pause functionality
                            while paused:
                                key = cv2.waitKey(30) & 0xFF
                                if key == ord('p'):  # Resume
                                    paused = False
                                elif key == ord('q'):  # Quit
                                    break
                            
                            if not paused:
                                cv2.imshow('Person Detection & Tracking', output_frame)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord('q'):  # Quit
                                    break
                                elif key == ord('p'):  # Pause
                                    paused = True
                    else:
                        # No detections, write empty frame
                        video_writer.write(frame)
                else:
                    # No results, write empty frame
                    video_writer.write(frame)
                
                frame_index += 1
                pbar.update(1)
                
                # Save gallery periodically if we're building it (every 100 frames)
                if args.build_gallery and gait_gallery and frame_index % 100 == 0:
                    gait_gallery.save_gallery()
                
                # Print tracking statistics every 100 frames
                if frame_index % 100 == 0:
                    active_tracks = len([tid for tid, history in track_history.items() 
                                        if len(history) > 0 and frame_index - 30 <= max(p[0] for p in history)])
                    logger.info(f"Frame {frame_index}: {active_tracks} active tracks, {len(track_history)} total tracks")

    cap.release()
    video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()
    
    # Save gallery at the end if in building mode
    if args.build_gallery and gait_gallery:
        logger.info("Saving gait embedding gallery...")
        gait_gallery.save_gallery()

    # Create Gait Energy Images (GEI) for each person if gait analysis is enabled
    if args.gait_analysis and track_frame_buffer:
        logger.info("Generating Gait Energy Images (GEI) for each person...")
        for person_id, silhouettes in track_frame_buffer.items():
            if len(silhouettes) >= 10:  # Only create GEI if we have enough frames
                gei_path = os.path.join(gait_dir, f"gei_person_{person_id:03d}.png")
                create_gait_energy_image(silhouettes, gei_path)
                logger.info(f"  Created GEI for Person {person_id} using {len(silhouettes)} frames")
                
                # Generate gait cycle visualization
                cycle_frames = min(len(silhouettes), 16)  # Use up to 16 frames for visualization
                step = max(1, len(silhouettes) // cycle_frames)

                # Create grid of silhouettes showing the gait cycle
                grid_cols = min(8, cycle_frames)
                grid_rows = (cycle_frames + grid_cols - 1) // grid_cols

                # IMPORTANT: Standardize all silhouettes to exactly the same size first
                std_silhouettes = []
                for sil in silhouettes:
                    # Use the first silhouette's dimensions as reference
                    if not std_silhouettes:
                        std_height, std_width = sil.shape
                    # Resize all silhouettes to exact same dimensions
                    resized_sil = cv2.resize(sil, (std_width, std_height))
                    std_silhouettes.append(resized_sil)

                # Create the grid image
                grid_img = np.zeros((grid_rows * std_height, grid_cols * std_width), dtype=np.uint8)

                # Fill the grid with silhouettes
                for i in range(cycle_frames):
                    row = i // grid_cols
                    col = i % grid_cols
                    idx = i * step
                    if idx < len(std_silhouettes):
                        grid_img[row*std_height:(row+1)*std_height, col*std_width:(col+1)*std_width] = std_silhouettes[idx]
                # Save gait cycle visualization
                cycle_path = os.path.join(gait_dir, f"gait_cycle_person_{person_id:03d}.png")
                cv2.imwrite(cycle_path, grid_img)
        
        logger.info(f"Gait analysis data saved to: {gait_dir}")
    
    logger.info(f"Output video saved to: {output_video_path}")
    if args.save_crops:
        logger.info(f"Segmented person crops saved to: {seg_crops_dir}")
    
    logger.info("=== Final Tracking Statistics ===")
    logger.info(f"Total unique persons detected: {len(track_history)}")
    
    # Print optimized embedding statistics
    if args.gait_analysis and opengait_model:
        logger.info("=== Optimized Embedding Management Statistics ===")
        logger.info(f"Tracks with embeddings generated: {len(track_embeddings)}")
        total_processing_runs = sum(track_processed_count.values()) if track_processed_count else 0
        logger.info(f"Total embedding processing runs: {total_processing_runs}")
        if track_processed_count:
            avg_processing = total_processing_runs / len(track_processed_count)
            logger.info(f"Average processing runs per track: {avg_processing:.1f}")
        
        # Show buffer statistics
        total_buffered_frames = sum(len(buffer) for buffer in track_frame_buffer.values())
        logger.info(f"Total frames buffered: {total_buffered_frames}")
        if track_frame_buffer:
            avg_buffer_size = total_buffered_frames / len(track_frame_buffer)
            logger.info(f"Average buffer size per track: {avg_buffer_size:.1f} frames")
        
        # Enhanced quality and completeness statistics
        if track_silhouette_quality:
            logger.info("=== Dataset Quality Assessment Results ===")
            complete_datasets = 0
            total_completeness = 0
            total_coverage = 0
            
            for person_id, quality_info in track_silhouette_quality.items():
                best_quality = quality_info["best_quality"]
                current_quality = quality_info["current_quality"]
                
                if best_quality["is_complete"]:
                    complete_datasets += 1
                
                total_completeness += best_quality["completeness"]
                total_coverage += best_quality["coverage"]
                
                # Report quality for each track
                logger.info(f"  Track {person_id}: Completeness={best_quality['completeness']:.3f}, "
                      f"Coverage={best_quality['coverage']:.3f}, Complete={best_quality['is_complete']}")
            
            logger.info(f"Tracks with complete datasets: {complete_datasets}/{len(track_silhouette_quality)}")
            if track_silhouette_quality:
                avg_completeness = total_completeness / len(track_silhouette_quality)
                avg_coverage = total_coverage / len(track_silhouette_quality)
                logger.info(f"Average dataset completeness: {avg_completeness:.3f}")
                logger.info(f"Average silhouette coverage: {avg_coverage:.3f}")
        
        # Show database efficiency
        if gait_gallery and hasattr(gait_gallery, 'track_to_identity'):
            logger.info("=== Gallery Update Statistics ===")
            logger.info(f"Gallery identities created: {len(set(gait_gallery.track_to_identity.values()))}")
            logger.info(f"Track-to-identity mappings: {len(gait_gallery.track_to_identity)}")
            
            # Show periodic update statistics
            if track_last_complete_batch:
                logger.info(f"Tracks with complete batch updates: {len(track_last_complete_batch)}")
                recent_complete_updates = sum(1 for frame_idx in track_last_complete_batch.values() 
                                            if frame_index - frame_idx <= periodic_update_interval)
                logger.info(f"Recent complete updates (last {periodic_update_interval} frames): {recent_complete_updates}")
    
    logger.info(f"Video processing completed. Results saved to {csv_path}")
    return 0

if __name__ == "__main__":
    main()