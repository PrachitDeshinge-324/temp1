# utils/gait_gallery.py
import os
import pickle
import numpy as np
import torch
from sklearn.preprocessing import normalize

class GaitGallery:
    def __init__(self, gallery_path=None):
        """Initialize gait embedding gallery"""
        self.gallery = {}  # person_id -> list of embeddings
        self.gallery_path = gallery_path
        
        if gallery_path and os.path.exists(gallery_path):
            self.load_gallery(gallery_path)
            
    def gallery_stats(self):
        """Get statistics about the gallery"""
        stats = {
            "total_identities": len(self.gallery),
            "total_embeddings": sum(len(embs) for embs in self.gallery.values()),
            "embeddings_per_identity": {pid: len(embs) for pid, embs in self.gallery.items()},
            "embedding_dimensions": None
        }
        
        # Get dimensions of first embedding if available
        for pid in self.gallery:
            if self.gallery[pid] and len(self.gallery[pid]) > 0:
                if isinstance(self.gallery[pid][0], torch.Tensor):
                    stats["embedding_dimensions"] = list(self.gallery[pid][0].shape)
                break
                
        return stats
    
    def add_embedding(self, person_id, embedding):
        """Add a gait embedding to the gallery"""
        try:
            # Validate inputs
            if embedding is None or (isinstance(embedding, torch.Tensor) and embedding.numel() == 0):
                print(f"Warning: Attempt to add empty embedding for person {person_id}")
                return False
                
            if person_id not in self.gallery:
                self.gallery[person_id] = []
                
            # Add the embedding to the gallery
            self.gallery[person_id].append(embedding)
            return True
        except Exception as e:
            print(f"Error adding embedding for person {person_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_aggregated_embedding(self, person_id):
        """Get view-invariant aggregated embedding for a person"""
        try:
            # Check if person_id exists in gallery and has embeddings
            if person_id not in self.gallery or not self.gallery[person_id]:
                print(f"Warning: No embeddings found for person {person_id}")
                return None
            
            # Check if we have valid embeddings
            valid_embeddings = [emb for emb in self.gallery[person_id] 
                              if emb is not None and isinstance(emb, torch.Tensor) and emb.numel() > 0]
            
            if not valid_embeddings:
                print(f"Warning: No valid embeddings found for person {person_id}")
                return None
                
            # Average pooling of all embeddings
            embeddings = torch.stack(valid_embeddings)
            aggregated = torch.mean(embeddings, dim=0)
            
            # Normalize
            norm = aggregated.norm()
            if norm > 0:
                aggregated = aggregated / norm
            else:
                print(f"Warning: Zero norm embedding for person {person_id}")
                return None
                
            return aggregated
        except Exception as e:
            print(f"Error getting aggregated embedding for person {person_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_match(self, query_embedding, threshold=0.6):
        """Find matching person ID for a query embedding"""
        try:
            # Validate query embedding
            if query_embedding is None or not isinstance(query_embedding, torch.Tensor) or query_embedding.numel() == 0:
                print("Warning: Invalid query embedding provided to find_match")
                return None, 0.0
                
            # Check if gallery is empty
            if not self.gallery or len(self.gallery) == 0:
                print("Warning: Gallery is empty, cannot find matches")
                return None, 0.0
                
            best_score = -1
            best_match = None
            
            for person_id in self.gallery:
                # Get reference embedding for this person
                ref_embedding = self.get_aggregated_embedding(person_id)
                if ref_embedding is None:
                    print(f"Warning: Could not get reference embedding for person {person_id}")
                    continue
                    
                # Ensure both embeddings are on the same device (CPU)
                query_embedding_cpu = query_embedding.cpu()
                ref_embedding_cpu = ref_embedding.cpu()
                
                # Compute similarity
                try:
                    score = torch.cosine_similarity(
                        query_embedding_cpu.unsqueeze(0),
                        ref_embedding_cpu.unsqueeze(0)
                    ).item()
                    
                    print(f"Match score for person {person_id}: {score:.4f}")
                    
                    if score > best_score and score > threshold:
                        best_score = score
                        best_match = person_id
                except Exception as e:
                    print(f"Error computing similarity for person {person_id}: {e}")
                    continue
                    
            return best_match, best_score
        except Exception as e:
            print(f"Error in find_match: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def save_gallery(self, path=None):
        """Save gallery to disk"""
        try:
            save_path = path or self.gallery_path
            if not save_path:
                print("Warning: No gallery path specified for saving")
                return False
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save the gallery
            with open(save_path, 'wb') as f:
                pickle.dump(self.gallery, f)
                
            print(f"Gallery saved successfully to {save_path}")
            print(f"Gallery contains {len(self.gallery)} identities")
            return True
        except Exception as e:
            print(f"Error saving gallery: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_gallery(self, path=None):
        """Load gallery from disk"""
        try:
            load_path = path or self.gallery_path
            if not load_path:
                print("Warning: No gallery path specified for loading")
                return False
                
            if not os.path.exists(load_path):
                print(f"Warning: Gallery file not found at {load_path}")
                return False
                
            # Load the gallery
            with open(load_path, 'rb') as f:
                loaded_gallery = pickle.load(f)
                
            # Validate loaded data
            if not isinstance(loaded_gallery, dict):
                print(f"Error: Invalid gallery format in {load_path}")
                return False
                
            # Update the gallery
            self.gallery = loaded_gallery
            
            print(f"Gallery loaded successfully from {load_path}")
            print(f"Gallery contains {len(self.gallery)} identities")
            return True
        except Exception as e:
            print(f"Error loading gallery: {e}")
            import traceback
            traceback.print_exc()
            return False