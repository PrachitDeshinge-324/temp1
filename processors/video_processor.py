"""
Video processing functionality
"""
import os
import cv2
import numpy as np
from collections import defaultdict
import tqdm

from utils.trackers import Detection

class VideoProcessor:
    """Process video for gait analysis"""
    
    def __init__(self, input_path, start_frame=0, end_frame=None, output_path=None, headless=False):
        self.input_path = input_path
        self.start_frame = start_frame
        self.output_path = output_path
        self.headless = headless
        
        self.cap = cv2.VideoCapture(input_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(end_frame, self.total_frames)
            
        self.writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            print(f"Writing output to {output_path}")

    def process_video(self, process_frame_func):
        """Process video with function that processes each frame"""
        frame_count = self.start_frame
        prev_time = cv2.getTickCount()
        fps = 0
        
        with tqdm.tqdm(total=self.end_frame-self.start_frame, 
                        desc="Processing frames") as pbar:
            while self.cap.isOpened() and frame_count < self.end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Calculate FPS
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - prev_time) / cv2.getTickFrequency()
                if elapsed_time > 0:
                    fps = 1.0 / elapsed_time
                prev_time = current_time
                
                # Process the frame using the provided function
                processed_frame = process_frame_func(frame, frame_count, fps)
                
                # Write processed frame to output video if writer exists
                if self.writer:
                    self.writer.write(processed_frame)
                
                # Display frame if not in headless mode and no output_path
                if not self.headless and self.output_path is None:
                    cv2.imshow('Gait Recognition', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                # Update progress bar
                pbar.update(1)
                frame_count += 1
        
        self.release()
    
    def release(self):
        """Release video resources"""
        self.cap.release()
        if self.writer:
            self.writer.release()
        if not self.headless and self.output_path is None:
            cv2.destroyAllWindows()

class GaitFrameProcessor:
    """Process individual frames for gait analysis"""
    
    def __init__(self, detector, tracker, pose_detector, gait_analyzer, visualizer, buffer_size=0.1):
        self.detector = detector
        self.tracker = tracker
        self.pose_detector = pose_detector
        self.gait_analyzer = gait_analyzer
        self.visualizer = visualizer
        self.buffer_size = buffer_size
        
        # Tracking state
        self.keypoints_history = defaultdict(lambda: [])
        self.history_length = 5
        self.bbox_info = defaultdict(list)

    def process_frame(self, frame, frame_count, fps, person_identifier=None, save_bbox_info=False):
        """Process a single video frame"""
        # Draw FPS counter
        self.visualizer.draw_fps(frame, fps)
        
        # Get detections using the tracker
        if hasattr(self.detector, 'update'):
            # First get detections from detector
            detections = self.detector.update(frame)
            # Then update tracker if it's separate from detector
            if self.tracker is not None and self.tracker != self.detector:
                detections = self.tracker.update(frame, detections)
        else:
            # Use tracker directly
            detections = self.tracker.update(frame)
            
        # Extra validation to ensure no invalid IDs pass through
        validated_detections = []
        for det in detections:
            # Ensure track_id is a positive integer (not 0, not negative)
            if not hasattr(det, 'track_id') or det.track_id <= 0:
                # Generate new valid ID for this detection
                if hasattr(self.tracker, 'next_id'):
                    new_id = self.tracker.next_id
                    det.track_id = new_id
                    self.tracker.next_id = new_id + 1
                else:
                    det.track_id = max(1, frame_count % 1000 + 100)  # Use frame count as fallback
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
                frame, detection, self.buffer_size)
            x1_buf, y1_buf, x2_buf, y2_buf = buffered_box
            
            # Save bounding box info if requested
            if save_bbox_info:
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
                if len(self.keypoints_history[track_id]) >= self.history_length:
                    self.keypoints_history[track_id].pop(0)
                self.keypoints_history[track_id].append(keypoints.copy())
                
                # Apply temporal smoothing to keypoints
                smoothed_keypoints = self.pose_detector.smooth_keypoints(
                    self.keypoints_history[track_id][:-1], keypoints)
                
                # Update gait analyzer with new keypoints
                self.gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
                
                # Perform identification if requested
                if person_identifier and frame_count % 15 == 0:
                    feature_vector = self.gait_analyzer.get_feature_vector(track_id)
                    if feature_vector is not None:
                        identity, confidence = person_identifier.identify_person(feature_vector)
                        self.visualizer.draw_identity(frame, detection, identity, confidence, color)
                
                # Draw keypoints and skeleton
                self.visualizer.draw_keypoints(frame, smoothed_keypoints, x1_buf, y1_buf, color)
        
        return frame
    
    def get_bbox_info(self):
        """Return the collected bounding box information"""
        return self.bbox_info
