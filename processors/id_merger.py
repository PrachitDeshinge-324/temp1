"""
ID merger for correcting tracking IDs
"""
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class IdMerger:
    """Interactive session to merge incorrectly split tracking IDs"""
    
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
            print(f"Track ID {track_id} not found in bbox data")
            return []
            
        detections = self.bbox_data[str(track_id)]
        
        # Select evenly spaced frame indices
        if len(detections) <= num_samples:
            indices = list(range(len(detections)))
        else:
            indices = np.linspace(0, len(detections) - 1, num_samples, dtype=int)
        
        samples = []
        for idx in indices:
            detect = detections[idx]
            frame_idx = detect['frame_idx']
            x1, y1, x2, y2 = detect['x1'], detect['y1'], detect['x2'], detect['y2']
            
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                # Crop person
                crop = frame[y1:y2, x1:x2].copy()
                # Add to samples
                samples.append((crop, frame_idx))
        
        return samples
    
    def display_id_samples(self, track_id, samples):
        """Display sample images for a specific tracking ID"""
        if not samples:
            print(f"No samples available for track ID {track_id}")
            return
            
        # Display samples for this ID
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Track ID: {track_id}", fontsize=16)
        
        for i, (img, frame_idx) in enumerate(samples):
            plt.subplot(1, len(samples), i+1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
            
        plt.tight_layout()
        
        # Check for Colab environment by looking for common environment variables
        is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
        
        if is_colab:
            from google.colab.patches import cv2_imshow
            plt.close()
            for img, _ in samples:
                cv2_imshow(img)
        else:
            plt.show()
    
    def merge_ids_interactive(self):
        """Interactive session to merge incorrectly split tracking IDs"""
        # Get unique track IDs
        track_ids = sorted([int(id_) for id_ in self.bbox_data.keys()])
        print(f"Found {len(track_ids)} unique track IDs")
        
        # Initialize merged ID mapping
        self.merged_ids = {}  # {original_id: target_id}
        self.id_to_name = {}  # {id: name}
        
        # Process each track ID
        i = 0
        while i < len(track_ids):
            track_id = track_ids[i]
            
            # Skip if this ID has already been merged
            if track_id in self.merged_ids:
                i += 1
                continue
                
            # Extract and display samples for this ID
            samples = self.extract_id_samples(track_id)
            
            if not samples:
                print(f"Skipping track ID {track_id} - no samples available")
                i += 1
                continue
                
            print(f"\n=== Track ID: {track_id} ===")
            self.display_id_samples(track_id, samples)
            
            # Ask for action
            action = input("Merge with another ID? (y/n/q to quit): ").strip().lower()
            if action == 'q':
                break
            elif action == 'y':
                target_id = input("Enter target ID to merge with: ").strip()
                try:
                    target_id = int(target_id)
                    if target_id in track_ids:
                        # Show target ID samples for confirmation
                        target_samples = self.extract_id_samples(target_id)
                        if target_samples:
                            print(f"=== Target ID: {target_id} ===")
                            self.display_id_samples(target_id, target_samples)
                            
                            # Confirm merge
                            confirm = input(f"Confirm merging {track_id} into {target_id}? (y/n): ").strip().lower()
                            if confirm == 'y':
                                self.merged_ids[track_id] = target_id
                                print(f"Merged track ID {track_id} into {target_id}")
                    else:
                        print(f"Target ID {target_id} not found")
                except ValueError:
                    print("Invalid target ID")
            
            # Ask for name assignment
            name_action = input("Assign a name to this ID? (y/n): ").strip().lower()
            if name_action == 'y':
                person_name = input("Enter person name: ").strip()
                if person_name:
                    # Determine which ID to associate with this name
                    id_to_name = track_id
                    if track_id in self.merged_ids:
                        id_to_name = self.merged_ids[track_id]
                    self.id_to_name[id_to_name] = person_name
                    print(f"Assigned name '{person_name}' to ID {id_to_name}")
            
            i += 1
        
        print("\nID merging complete!")
        print(f"Merged {len(self.merged_ids)} IDs")
        print(f"Named {len(self.id_to_name)} IDs")
        
        return self.merged_ids, self.id_to_name
    
    def update_feature_data(self):
        """Update feature data with merged IDs"""
        if self.features_data is None or not self.merged_ids:
            return False
            
        print("Updating feature data with merged IDs...")
        
        # Make a copy of the feature data to avoid modifying the original
        updated_data = self.features_data.copy()
        
        # Replace track IDs in the first column
        for i in range(len(updated_data)):
            track_id = int(updated_data[i, 0])
            if track_id in self.merged_ids:
                updated_data[i, 0] = float(self.merged_ids[track_id])
                
        return updated_data
    
    def save_updated_data(self, output_dir="results"):
        """Save updated data (merged IDs, feature data) to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save merged IDs
        if self.merged_ids:
            merged_ids_path = os.path.join(output_dir, "merged_ids.json")
            with open(merged_ids_path, 'w') as f:
                # Convert ints to strings for JSON compatibility
                merged_ids_str = {str(k): int(v) for k, v in self.merged_ids.items()}
                json.dump(merged_ids_str, f)
            print(f"Saved merged IDs to {merged_ids_path}")
            
        # Save ID to name mapping
        if self.id_to_name:
            id_to_name_path = os.path.join(output_dir, "id_to_name.json")
            with open(id_to_name_path, 'w') as f:
                # Convert ints to strings for JSON compatibility
                id_to_name_str = {str(k): v for k, v in self.id_to_name.items()}
                json.dump(id_to_name_str, f)
            print(f"Saved ID to name mapping to {id_to_name_path}")
            
        # Save updated feature data
        if self.features_data is not None and self.merged_ids:
            updated_data = self.update_feature_data()
            if updated_data is not None:
                # Save to the same path with _preprocessed suffix
                output_path = self.features_npy_path.replace('.npy', '_preprocessed.npy')
                np.save(output_path, updated_data)
                print(f"Saved updated feature data to {output_path}")
                
        return True
    
    def close(self):
        """Clean up resources"""
        if self.cap.isOpened():
            self.cap.release()

def merge_ids(args, features_npy_path):
    """
    Interactive session to merge incorrectly split tracking IDs
    """
    print("\n=== Starting ID Merger ===")
    print("This utility will help you merge tracking IDs that belong to the same person.")
    
    # Initialize ID merger
    bbox_json_path = os.path.join(args.results_dir, "bbox_info.json")
    
    if not os.path.exists(bbox_json_path):
        print(f"Error: Bounding box info file not found at {bbox_json_path}")
        return None, None
    
    merger = IdMerger(args.video, bbox_json_path, features_npy_path)
    
    try:
        # Run interactive merging session
        merged_ids, id_to_name = merger.merge_ids_interactive()
        
        # Save results
        merger.save_updated_data(args.results_dir)
        
        return merged_ids, id_to_name
    
    finally:
        merger.close()
