import cv2 as cv
import os
import tqdm
from collections import defaultdict

class VideoProcessor:
    def __init__(self, input_path, start_frame=0, end_frame=100, 
                 output_path=None, headless=False):
        self.input_path = input_path
        self.start_frame = start_frame
        self.output_path = output_path
        self.headless = headless
        
        self.cap = cv.VideoCapture(input_path)
        self.cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(end_frame, self.total_frames)
            
        self.writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.writer = cv.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            print(f"Writing output to {output_path}")
    
    def process_video(self, process_frame_func):
        """Process video with function that processes each frame"""
        frame_count = self.start_frame
        prev_time = cv.getTickCount()
        fps = 0
        
        with tqdm.tqdm(total=self.end_frame-self.start_frame, 
                       desc="Processing frames") as pbar:
            while self.cap.isOpened() and frame_count < self.end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                current_time = cv.getTickCount()
                time_diff = (current_time - prev_time) / cv.getTickFrequency()
                if time_diff > 0:
                    fps = 1.0 / time_diff
                prev_time = current_time
                
                # Process frame
                processed_frame = process_frame_func(frame, frame_count, fps)
                
                # Write or display frame
                if self.writer:
                    self.writer.write(processed_frame)

                if not self.headless:
                    cv.imshow('Video Processing', processed_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                pbar.update(1)
        
        self.release()
    
    def release(self):
        """Release video resources"""
        self.cap.release()
        if self.writer:
            self.writer.release()
        if not self.headless and self.output_path is None:
            cv.destroyAllWindows()