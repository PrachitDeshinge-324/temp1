import cv2
import numpy as np
import random

def generate_colors(num_colors):
    """Generate a list of distinct colors for person IDs"""
    colors = []
    for i in range(num_colors):
        # Generate distinct colors using HSV color space
        hue = (i * 137.508) % 360  # Golden angle approximation for good distribution
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors

def draw_person_detection(frame, bbox, person_id, color=(0, 255, 0), thickness=2, identity_id=None):
    """
    Draw bounding box and person ID on the frame
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        person_id: Person ID to display (tracking ID)
        color: Color for the bounding box and text
        thickness: Line thickness for bounding box
        identity_id: Optional persistent identity ID from gait analysis
    
    Returns:
        Frame with annotations
    """
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare text - show identity if available, otherwise just show tracking ID
    if identity_id is not None:
        text = f"ID:{identity_id} (T{person_id})"
    else:
        text = f"Track {person_id}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    
    # Draw text background
    text_x = x1
    text_y = y1 - 10
    if text_y < text_height:
        text_y = y1 + text_height + 10
    
    cv2.rectangle(frame, 
                  (text_x, text_y - text_height - 5), 
                  (text_x + text_width + 10, text_y + 5), 
                  color, -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x + 5, text_y - 5), 
                font, font_scale, (255, 255, 255), text_thickness)
    
    return frame

def draw_keypoints(frame, keypoints, person_id, color=(0, 255, 0)):
    """
    Draw pose keypoints on the frame
    
    Args:
        frame: Input frame
        keypoints: List of keypoint coordinates [x1, y1, x2, y2, ...]
        person_id: Person ID
        color: Color for keypoints
    
    Returns:
        Frame with keypoint annotations
    """
    if keypoints is None or len(keypoints) != 34:
        return frame
    
    # COCO 17 keypoints connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Convert keypoints to (x, y) pairs
    points = []
    for i in range(0, len(keypoints), 2):
        x, y = keypoints[i], keypoints[i + 1]
        if x is not None and y is not None and x > 0 and y > 0:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx < len(points) and end_idx < len(points) and 
            points[start_idx] is not None and points[end_idx] is not None):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
    
    # Draw keypoints
    for i, point in enumerate(points):
        if point is not None:
            cv2.circle(frame, point, 3, color, -1)
    
    return frame

def create_video_writer(output_path, fps, frame_width, frame_height):
    """
    Create a video writer object
    
    Args:
        output_path: Output video file path
        fps: Frames per second
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out