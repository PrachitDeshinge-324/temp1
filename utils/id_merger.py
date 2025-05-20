import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class IDMerger:
    def __init__(self, video_path, bbox_json_path, features_npy_path=None):
        """Initialize the ID merger with paths to video and bbox data"""
        self.video_path = video_path
        self.bbox_json_path = bbox_json_path
        self.features_npy_path = features_npy_path
        
        # Load bbox data
        with open(bbox_json_path, 'r') as f:
            self.bbox_data = json.load(f)
            
        # Load features data if provided
        self.features_data = None
        if features_npy_path and os.path.exists(features_npy_path):
            self.features_data = np.load(features_npy_path)
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Store ID mappings
        self.id_to_name = {}
        self.merged_ids = {}  # {original_id: merged_id}
        
    def extract_id_samples(self, track_id, num_samples=3):
        """Extract sample frames for a specific tracking ID"""
        if str(track_id) not in self.bbox_data:
            return []
            
        detections = self.bbox_data[str(track_id)]
        
        # Select evenly spaced frame indices
        if len(detections) <= num_samples:
            indices = range(len(detections))
        else:
            step = len(detections) // num_samples
            indices = range(0, len(detections), step)[:num_samples]
        
        samples = []
        for idx in indices:
            detection = detections[idx]
            frame_idx = detection['frame_idx']
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            
            # Set video to the correct frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                # Crop person
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    samples.append((crop, frame_idx))
        
        return samples
    
    def display_id_samples(self, track_id, samples):
        """Display sample images for a specific tracking ID"""
        if not samples:
            print(f"No samples available for Track ID: {track_id}")
            return
            
        # Display samples for this ID
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Track ID: {track_id}", fontsize=16)
        
        for i, (img, frame_idx) in enumerate(samples):
            plt.subplot(1, len(samples), i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
            
        plt.tight_layout()
        
        # Check for Colab environment by looking for common environment variables
        is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
        
        if is_colab:
            # For Colab, we need to make the plot display non-blocking
            try:
                plt.show(block=False)
                # Small pause to ensure the plot is displayed
                import time
                time.sleep(0.5)
            except Exception:
                plt.show()
        else:
            # Normal display for non-Colab environments
            plt.show()
    
    def merge_ids_interactive(self):
        """Interactive session to merge tracking IDs"""
        # Step 1: Display each ID one by one and get names
        print("\n=== Step 1: Name each person ID ===")
        
        # Sort IDs numerically for consistent display
        track_ids = sorted([int(track_id) for track_id in self.bbox_data.keys()])
        
        # First pass: Display each ID and get names
        for track_id in track_ids:
            # Skip IDs that were already merged in a previous session
            if track_id in self.merged_ids:
                continue
                
            samples = self.extract_id_samples(track_id)
            
            if samples:
                print(f"\nShowing samples for person with ID {track_id}")
                self.display_id_samples(track_id, samples)
                
                # Ask for person's name
                name = input(f"Enter name for person with ID {track_id} (press Enter to skip): ")
                
                if name.strip():
                    self.id_to_name[track_id] = name.strip()
                    print(f"Assigned name '{name}' to ID {track_id}")
            else:
                print(f"No samples available for ID {track_id}")
        
        # Step 2: Merge IDs
        print("\n\n=== Step 2: Merge IDs for the same person ===")
        print("Now you can merge IDs that belong to the same person.")
        print("Available IDs and their assigned names:")
        
        # Show the list of IDs and names
        for track_id in track_ids:
            if track_id in self.merged_ids:
                continue
            name = self.id_to_name.get(track_id, "unnamed")
            print(f"  ID {track_id}: {name}")
        
        # Get user input for merges
        while True:
            merge_input = input("\nEnter IDs to merge (comma-separated, e.g., '1,4') or 'done' to finish: ")
            if merge_input.lower() == 'done':
                break
                
            try:
                ids_to_merge = [int(id_str.strip()) for id_str in merge_input.split(',')]
                if len(ids_to_merge) < 2:
                    print("Need at least 2 IDs to merge. Try again.")
                    continue
                
                # Check if all IDs exist
                for id_val in ids_to_merge:
                    if str(id_val) not in self.bbox_data:
                        print(f"ID {id_val} does not exist. Try again.")
                        continue
                    
                # Assign all these IDs to the lowest ID value
                target_id = min(ids_to_merge)
                
                # If any of the IDs have a name, use that for the merged ID
                target_name = None
                for id_val in ids_to_merge:
                    if id_val in self.id_to_name:
                        target_name = self.id_to_name[id_val]
                        break
                
                # If no name was found, ask for one
                if target_name is None:
                    target_name = input(f"Enter name for merged person with ID {target_id}: ")
                    
                # Update mappings
                self.id_to_name[target_id] = target_name
                for id_val in ids_to_merge:
                    if id_val != target_id:
                        self.merged_ids[id_val] = target_id
                        # Remove individual name if it exists
                        if id_val in self.id_to_name:
                            del self.id_to_name[id_val]
                
                print(f"Merged IDs {ids_to_merge} to ID {target_id} with name '{target_name}'")
                
            except ValueError:
                print("Invalid input. Please enter comma-separated integers.")
                
        return self.merged_ids, self.id_to_name
    
    def update_feature_data(self):
        """Update the feature numpy array with the merged IDs"""
        if self.features_data is None or not self.merged_ids:
            return None
            
        updated_data = self.features_data.copy()
        
        # Update track IDs in the first column
        for i in range(len(updated_data)):
            original_id = int(updated_data[i, 0])
            if original_id in self.merged_ids:
                updated_data[i, 0] = self.merged_ids[original_id]
                
        return updated_data
    
    def save_updated_data(self, output_dir="results"):
        """Save the updated data (ID mappings and feature data)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ID to name mapping
        id_mapping_path = os.path.join(output_dir, "id_to_name.json")
        with open(id_mapping_path, 'w') as f:
            json.dump(self.id_to_name, f, indent=2)
        print(f"Saved ID to name mapping to {id_mapping_path}")
        
        # Save merged IDs mapping
        merged_ids_path = os.path.join(output_dir, "merged_ids.json")
        with open(merged_ids_path, 'w') as f:
            json.dump(self.merged_ids, f, indent=2)
        print(f"Saved merged IDs mapping to {merged_ids_path}")
        
        # Save updated feature data
        if self.features_data is not None and self.features_npy_path:
            updated_data = self.update_feature_data()
            if updated_data is not None:
                output_path = os.path.join(output_dir, os.path.basename(self.features_npy_path))
                np.save(output_path, updated_data)
                print(f"Saved updated feature data to {output_path}")
    
    def close(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        plt.close('all')