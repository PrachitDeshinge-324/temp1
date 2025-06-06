import argparse
import os
import cv2
import csv
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from utils.helper import get_best_device
from utils.transreid import TransReIDModel
from utils.visualize import draw_person_detection, generate_colors, create_video_writer
from utils.reid_tracker import ReIDEnhancedTracker
from utils.opengait_model import OpenGaitModel
from utils.silhouette_processor import SilhouetteProcessor
from utils.gait_gallery import GaitGallery

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
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output_dir}")
    print(f"Weights folder: {args.weights_dir}")
    return args

def main():
    args = get_arguments()
    video_path = args.input
    output_dir = args.output_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    print(f"Processing video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total frames in video: {frame_count}")
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Video FPS: {fps}")

    # Always initialize video writer to save output video
    output_video_path = os.path.join(output_dir, "output_with_detections.mp4")
    video_writer = create_video_writer(output_video_path, fps, frame_width, frame_height)
    print(f"Output video will be saved to: {output_video_path}")
    print("Annotations will be displayed on the output video")
    
    # Create directory for segmented crops and silhouettes
    seg_crops_dir = os.path.join(output_dir, "segmented_crops")
    silhouette_dir = os.path.join(output_dir, "silhouettes")
    gait_dir = os.path.join(output_dir, "gait_analysis")
    
    if args.save_crops and not os.path.exists(seg_crops_dir):
        os.makedirs(seg_crops_dir)
        print(f"Segmented crops will be saved to: {seg_crops_dir}")
    
    if args.gait_analysis:
        if not os.path.exists(silhouette_dir):
            os.makedirs(silhouette_dir)
        if not os.path.exists(gait_dir):
            os.makedirs(gait_dir)
        print(f"Gait analysis data will be saved to: {gait_dir}")
    
    if args.display:
        print("Real-time display enabled - press 'q' to quit, 'p' to pause/resume")
        cv2.namedWindow('Person Detection & Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Person Detection & Tracking', 960, 540)  # Resize for better viewing

    # Generate colors for person IDs
    person_colors = generate_colors(50)  # Generate 50 distinct colors

    # Initialize YOLO models for detection and segmentation
    print(f"Loading YOLOv8 models...")
    person_detector = YOLO(os.path.join(args.weights_dir, "yolo11x.pt"))  # For person detection
    segmentation_model = YOLO(os.path.join(args.weights_dir, "yolo11x-seg.pt"))  # For person segmentation
    print("YOLOv8 models loaded successfully.")
    
    # Initialize TransReID model for feature extraction (optional)
    transreid_model = None
    use_reid_features = False
    reid_tracker = None
    
    if args.transreid_weights:
        try:
            print(f"Loading TransReID model...")
            transreid_model = TransReIDModel(args.transreid_weights, device=args.device)
            print("TransReID model loaded successfully for feature extraction.")
            use_reid_features = True
            
            # Initialize ReID tracker with TransReID model
            print("Initializing ReID-enhanced tracker...")
            reid_tracker = ReIDEnhancedTracker(
                transreid_model=transreid_model,
                similarity_threshold=args.reid_similarity,
                feature_history_size=10,
                reid_memory_frames=30
            )
            print(f"ReID-enhanced tracker initialized with similarity threshold: {args.reid_similarity}")
            
        except Exception as e:
            print(f"Warning: Could not load TransReID model: {e}")
            print("Proceeding without ReID feature extraction.")

    # Initialize OpenGait components if requested
    opengait_model = None
    silhouette_processor = None
    gait_gallery = None

    if args.opengait_weights:
        try:
            print(f"Loading OpenGait model...")
            # Use the config path from command-line arguments
            opengait_model = OpenGaitModel(args.opengait_weights, args.opengait_config, device=args.device)
            silhouette_processor = SilhouetteProcessor()
            
            # Initialize gait gallery
            gait_gallery = GaitGallery(args.gait_gallery)
            
            # Clear gallery if requested
            if args.clear_gallery and gait_gallery:
                print("Clearing existing gallery as requested...")
                gait_gallery.gallery = {}
                gait_gallery.next_id = 1
            # Print gallery statistics
            if gait_gallery and hasattr(gait_gallery, 'gallery_stats'):
                stats = gait_gallery.gallery_stats()
                print(f"OpenGait model loaded successfully.")
                print(f"Gallery statistics:")
                print(f"  Total identities: {stats['total_identities']}")
                print(f"  Total embeddings: {stats['total_embeddings']}")
                if stats['embedding_dimensions']:
                    print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
            else:
                print(f"OpenGait model loaded successfully. No gallery statistics available.")
        except Exception as e:
            print(f"Warning: Could not load OpenGait model: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
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
    
    print(f"Using ByteTrack with configuration:")
    for key, value in tracking_config.items():
        print(f"  {key}: {value}")
    
    # Store silhouettes for gait analysis
    person_silhouettes = {}
    
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
                                        
                                    # Store normalized silhouette for GEI generation
                                    # We always collect silhouettes to ensure we have enough for analysis
                                    person_silhouettes[person_id].append(normalized_silhouette)
                                    
                                    # Limit the number of stored silhouettes to prevent memory issues
                                    # Keep only the most recent 60 frames (arbitrary limit that can be adjusted)
                                    if len(person_silhouettes[person_id]) > 60:
                                        person_silhouettes[person_id] = person_silhouettes[person_id][-60:]
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
                        
                            # Add these variables at the beginning of main()
                            track_last_processed = {}  # Frame when track was last processed
                            track_processed_count = {}  # How many times each track has been processed
                            min_frames_between_updates = 30  # Minimum frames between updates for same track

                            # Then modify the gait analysis section:
                            if args.gait_analysis and opengait_model:
                                try:
                                    for person_id in person_silhouettes:
                                        # Check if we have enough silhouettes
                                        if len(person_silhouettes[person_id]) >= 10:
                                            # Only process this track if:
                                            # 1. We've never processed it before, OR
                                            # 2. It's been enough frames since last update, OR
                                            # 3. We've only processed it a few times (better embeddings with more data)
                                            should_process = (
                                                person_id not in track_last_processed or
                                                (frame_index - track_last_processed[person_id]) > min_frames_between_updates or
                                                (person_id in track_processed_count and track_processed_count[person_id] < 3)
                                            )
                                            
                                            if should_process:
                                                # Process sequence for OpenGait
                                                sequence = silhouette_processor.prepare_sequence(person_silhouettes[person_id])
                                                
                                                # Extract gait embedding
                                                gait_embedding = opengait_model.extract_embeddings(sequence)
                                                
                                                # Update processing history
                                                track_last_processed[person_id] = frame_index
                                                if person_id not in track_processed_count:
                                                    track_processed_count[person_id] = 1
                                                else:
                                                    track_processed_count[person_id] += 1
                                                
                                                # Only proceed if we got valid embeddings
                                                if gait_embedding is not None and gait_gallery:
                                                    # Ensure track_to_identity exists
                                                    if not hasattr(gait_gallery, 'track_to_identity'):
                                                        gait_gallery.track_to_identity = {}
                                                    
                                                    # Check if this track already has an identity
                                                    if person_id in gait_gallery.track_to_identity:
                                                        # UPDATE existing identity with improved embedding
                                                        identity_id = gait_gallery.track_to_identity[person_id]
                                                        gait_gallery.update_embedding(identity_id, gait_embedding)
                                                        print(f"Updated embedding for Track {person_id} → Identity {identity_id}")
                                                    else:
                                                        # This is a new track - match or assign identity
                                                        identity_id, confidence, is_new = gait_gallery.get_or_assign_identity(
                                                            gait_embedding, 
                                                            threshold=args.gait_threshold,
                                                            force_new=(frame_index < 200)
                                                        )
                                                        
                                                        if is_new:
                                                            print(f"New identity created: Track {person_id} → Identity {identity_id}")
                                                        else:
                                                            print(f"Gait match found: Track {person_id} → Identity {identity_id} (confidence: {confidence:.2f})")
                                                        
                                                        # Map track ID to identity ID
                                                        gait_gallery.track_to_identity[person_id] = identity_id
                                                        
                                                    # Save gallery after processing
                                                    if frame_index % 100 == 0:
                                                        gait_gallery.save_gallery()
                                except Exception as e:
                                    print(f"Error in gait analysis: {e}")
                                    
                        # Write frame to output video
                        video_writer.write(output_frame)
                        
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
                    print(f"\nSaving gallery at frame {frame_index}...")
                    gait_gallery.save_gallery()
                
                # Print tracking statistics every 100 frames
                if frame_index % 100 == 0:
                    active_tracks = len([tid for tid, history in track_history.items() 
                                        if len(history) > 0 and frame_index - 30 <= max(p[0] for p in history)])
                    print(f"\nFrame {frame_index}: {active_tracks} active tracks, {len(track_history)} total tracks")

    cap.release()
    video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()
    
    # Save gallery at the end if in building mode
    if args.build_gallery and gait_gallery:
        print("Saving gait embedding gallery...")
        gait_gallery.save_gallery()

    # Create Gait Energy Images (GEI) for each person if gait analysis is enabled
    if args.gait_analysis and person_silhouettes:
        print("\nGenerating Gait Energy Images (GEI) for each person...")
        for person_id, silhouettes in person_silhouettes.items():
            if len(silhouettes) >= 10:  # Only create GEI if we have enough frames
                gei_path = os.path.join(gait_dir, f"gei_person_{person_id:03d}.png")
                create_gait_energy_image(silhouettes, gei_path)
                print(f"  Created GEI for Person {person_id} using {len(silhouettes)} frames")
                
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
        
        print(f"Gait analysis data saved to: {gait_dir}")
    
    print(f"Output video saved to: {output_video_path}")
    if args.save_crops:
        print(f"Segmented person crops saved to: {seg_crops_dir}")
    print(f"\n=== Final Tracking Statistics ===")
    print(f"Total unique persons detected: {len(track_history)}")
    print(f"Video processing completed. Results saved to {csv_path}")
    return 0

if __name__ == "__main__":
    main()