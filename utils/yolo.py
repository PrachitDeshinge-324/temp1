from ultralytics import YOLO
import os
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_yolo_model(weights_dir: str, model_filename: str = "yolov8n.pt", device: str = "cpu"):
    model_path = os.path.join(weights_dir, model_filename)
    print(f"Trying to load model from: {model_path}")
    if not os.path.isfile(model_path):
        print(f"Model file does not exist: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"Model loaded successfully from {model_path} on device {device}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

class PersonDetector:
    def __init__(self, weights_dir: str, model_filename: str = "yolov8n.pt", device: str = "cpu"):
        self.model = load_yolo_model(weights_dir, model_filename, device)

    def detect(self, frame):
        if self.model is None:
            print("Model not loaded. Cannot perform detection.")
            return []
        results = self.model(frame)
        bboxes = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Assuming class 0 is 'person'
                    bboxes.append(box.xyxy.cpu().numpy())
        return bboxes
    
class PoseDetector:
    def __init__(self, weights_dir: str, model_filename: str = "yolov8n-pose.pt", device: str = "cpu"):
        self.model = load_yolo_model(weights_dir, model_filename, device)

    def detect(self, frame):
        if self.model is None:
            print("Model not loaded. Cannot perform detection.")
            return []
        results = self.model(frame)
        poses = []
        for result in results:
            for pose in result.keypoints:
                poses.append(pose.cpu().numpy())
        return poses
    
class ShiiloteDetector:
    def __init__(self, weights_dir: str, model_filename: str = "yolov8n-shiilote.pt", device: str = "cpu"):
        self.model = load_yolo_model(weights_dir, model_filename, device)

    def detect(self, frame):
        if self.model is None:
            print("Model not loaded. Cannot perform detection.")
            return []
        results = self.model(frame)
        shiilotes = []
        for result in results:
            for shiilote in result.boxes:
                if shiilote.cls == 1:  # Assuming class 1 is 'shiilote'
                    shiilotes.append(shiilote.xyxy.cpu().numpy())
        return shiilotes