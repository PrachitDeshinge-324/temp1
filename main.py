import argparse
import os
import cv2
import csv
from tqdm import tqdm
from utils.yolo import PersonDetector, PoseDetector
from utils.helper import get_best_device
from utils.transreid import TransReIDModel, PersonTracker, PersonTrackerKalman
from utils.visualize import draw_person_detection, draw_keypoints, generate_colors, create_video_writer

def get_arguments():
    parser = argparse.ArgumentParser(description="Process video for person detection, pose estimation, and tracking.")
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder path')
    parser.add_argument('--weights_dir', type=str, required=True, help='Weights folder path')
    parser.add_argument('--transreid_weights', type=str, default=None, help='TransReID weights path')
    parser.add_argument('--device', type=str, default=get_best_device(), help='Device to run the model on (cpu or cuda)')
    parser.add_argument('--output_video', action='store_true', help='Generate output video with annotations')
    parser.add_argument('--display', action='store_true', help='Display video in real-time using cv2.imshow')
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
    
    if args.display:
        print("Real-time display enabled - press 'q' to quit, 'p' to pause/resume")
        cv2.namedWindow('Person Detection & Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Person Detection & Tracking', 960, 540)  # Resize for better viewing

    # Generate colors for person IDs
    person_colors = generate_colors(50)  # Generate 50 distinct colors

    person_detector = PersonDetector(
        weights_dir=args.weights_dir,
        model_filename="yolo11x.pt",
        device=args.device
    )

    pose_detector = PoseDetector(
        weights_dir=args.weights_dir,
        model_filename="yolo11x-pose.pt",
        device=args.device
    )

    # Initialize TransReID model and tracker if weights provided
    person_tracker = None
    use_tracking = False
    if args.transreid_weights:
        try:
            print(f"Loading TransReID model...")
            transreid_model = TransReIDModel(args.transreid_weights, device=args.device)
            person_tracker = PersonTrackerKalman(transreid_model, similarity_threshold=0.7, max_disappeared=30)
            person_tracker.enable_debug(False)  # Enable debug mode for detailed tracking info
            use_tracking = True
            print("TransReID model loaded successfully. Enhanced person tracking enabled.")
            print("Features: Advanced motion prediction, lost track recovery, feature-based matching")
        except Exception as e:
            print(f"Warning: Could not load TransReID model: {e}")
            print("Proceeding without person tracking.")
            use_tracking = False

    csv_path = os.path.join(output_dir, "detections.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Add keypoint columns (for 17 keypoints: x1,y1,x2,y2,...,x17,y17) and person_id
        keypoint_headers = [f'kp{i}_{axis}' for i in range(1, 18) for axis in ('x', 'y')]
        headers = ['frame', 'person_id', 'x1', 'y1', 'x2', 'y2'] + keypoint_headers
        writer.writerow(headers)

        frame_index = 0
        paused = False
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                bboxes = person_detector.detect(frame)
                
                # Extract person crops for tracking
                person_crops = []
                for bbox in bboxes:
                    coords = bbox.flatten().astype(int).tolist()
                    x1, y1, x2, y2 = coords
                    crop = frame[y1:y2, x1:x2]
                    person_crops.append(crop)
                
                # Assign person IDs if tracking is enabled
                person_ids = []
                if use_tracking and person_crops:
                    # Convert bboxes to the required format for the tracker
                    bbox_coords = []
                    for bbox in bboxes:
                        coords = bbox.flatten().astype(int).tolist()
                        bbox_coords.append(coords)
                    person_ids = person_tracker.assign_ids(person_crops, bbox_coords, frame_index)
                else:
                    # If no tracking, use frame-specific IDs (not persistent across frames)
                    person_ids = list(range(len(person_crops)))
                
                # Create a copy of the frame for video output (always save video)
                output_frame = frame.copy()
                
                # Process each detection
                for i, bbox in enumerate(bboxes):
                    coords = bbox.flatten().astype(int).tolist()
                    x1, y1, x2, y2 = coords
                    
                    # Get person ID
                    person_id = person_ids[i] if i < len(person_ids) else -1
                    
                    # Detect keypoints on the crop
                    crop = person_crops[i]
                    keypoints_list = pose_detector.detect(crop)
                    
                    # If keypoints are detected, extract and save; else, fill with Nones
                    if keypoints_list and len(keypoints_list) > 0:
                        # keypoints_list[0] is a Keypoints object
                        keypoints = keypoints_list[0].xy.flatten().tolist()
                        # Convert relative keypoints to absolute coordinates
                        absolute_keypoints = []
                        for j in range(0, len(keypoints), 2):
                            abs_x = keypoints[j] + x1 if keypoints[j] is not None else None
                            abs_y = keypoints[j + 1] + y1 if keypoints[j + 1] is not None else None
                            absolute_keypoints.extend([abs_x, abs_y])
                    else:
                        keypoints = [None] * 34  # 17 keypoints * 2 (x, y)
                        absolute_keypoints = keypoints
                    
                    # Save detection data to CSV
                    writer.writerow([frame_index, person_id] + coords + keypoints)
                    
                    # Always draw annotations on frame
                    # Get color for this person ID
                    color = person_colors[person_id % len(person_colors)] if person_id >= 0 else (128, 128, 128)
                    
                    # Draw bounding box and person ID
                    draw_person_detection(output_frame, (x1, y1, x2, y2), person_id, color)
                    
                    # Draw keypoints if detected
                    if absolute_keypoints and any(kp is not None for kp in absolute_keypoints):
                        draw_keypoints(output_frame, absolute_keypoints, person_id, color)
                
                # Write frame to output video (always save video with annotations)
                video_writer.write(output_frame)
                
                # Display frame in real-time if requested
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
                
                frame_index += 1
                pbar.update(1)
                
                # Print tracking statistics every 100 frames
                if use_tracking and frame_index % 100 == 0:
                    stats = person_tracker.get_stats()
                    print(f"\nFrame {frame_index}: {stats['active_persons']} active, "
                          f"{stats['confirmed_persons']} confirmed, "
                          f"{stats['recovery_count']} recoveries")
                    
                    # Show comprehensive summary every 300 frames
                    if frame_index % 300 == 0:
                        print(person_tracker.get_tracking_summary())

    cap.release()
    # Always close video writer since we always create it
    video_writer.release()
    
    # Close display window if it was opened
    if args.display:
        cv2.destroyAllWindows()
    
    print(f"Output video saved to: {os.path.join(output_dir, 'output_with_detections.mp4')}")
    
    # Print final tracking statistics
    if use_tracking:
        print(person_tracker.get_tracking_summary())
        final_stats = person_tracker.get_stats()
        print(f"\n=== Final Tracking Statistics ===")
        print(f"Total unique persons detected: {final_stats['next_id'] - 1}")
        print(f"Active persons at end: {final_stats['active_persons']}")
        print(f"Confirmed persons: {final_stats['confirmed_persons']}")
        print(f"Successful recoveries: {final_stats['recovery_count']}")
        
        # Calculate and display ID consistency metrics
        recovery_rate = (final_stats['recovery_count'] / max(1, final_stats['total_tracks_created'])) * 100
        print(f"ID Recovery Rate: {recovery_rate:.1f}%")
        print(f"Tracking Quality: {'Excellent' if recovery_rate > 20 else 'Good' if recovery_rate > 10 else 'Improving'}")
    
    print(f"Video processing completed. Results saved to {csv_path}")
    return 0

if __name__ == "__main__":
    main()