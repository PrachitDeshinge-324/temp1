# utils/silhouette_processor.py
import cv2
import numpy as np
import torch

class SilhouetteProcessor:
    def __init__(self, sequence_length=30, height=64, width=44):
        """Initialize silhouette processor"""
        self.sequence_length = sequence_length
        self.height = height
        self.width = width
        
    def normalize_silhouette(self, silhouette):
        """Normalize a single silhouette"""
        # Resize to standard dimensions
        normalized = cv2.resize(silhouette, (self.width, self.height))
        # Ensure binary (0 or 255)
        _, normalized = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        return normalized
    
    def prepare_sequence(self, silhouettes):
        """Prepare a sequence of silhouettes for OpenGait"""
        try:
            # Check if the input is valid
            if silhouettes is None or len(silhouettes) == 0:
                print("Warning: Empty silhouette sequence provided")
                return None
                
            # Handle varying sequence lengths
            if len(silhouettes) < self.sequence_length:
                # Pad with blank silhouettes
                padding = [np.zeros((self.height, self.width), dtype=np.uint8) 
                          for _ in range(self.sequence_length - len(silhouettes))]
                padded_sequence = silhouettes + padding
            else:
                # Sample or take consecutive frames
                step = max(1, len(silhouettes) // self.sequence_length)
                padded_sequence = [silhouettes[min(i*step, len(silhouettes)-1)] for i in range(self.sequence_length)]
                
            # Normalize all silhouettes
            normalized = [self.normalize_silhouette(sil) for sil in padded_sequence]
            
            # Stack into tensor format expected by OpenGait
            tensor_sequence = np.stack(normalized, axis=0)
            return torch.from_numpy(tensor_sequence).float() / 255.0
        except Exception as e:
            print(f"Error preparing silhouette sequence: {e}")
            import traceback
            traceback.print_exc()
            return None