# utils/gait_gallery.py
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

class GaitGallery:
    def __init__(self, gallery_path=None):
        """Initialize gait embedding gallery"""
        self.gallery = {}  # person_id -> list of embeddings
        self.gallery_path = gallery_path
        self.next_id = 1  # For assigning new unique IDs to unmatched people
        self.track_to_identity = {}  # Initialize the mapping attribute
        
        # Frame-level ID management to prevent duplicates within single frames
        self.current_frame_index = -1
        self.frame_assigned_ids = set()  # Track which IDs have been assigned in current frame
        
        if gallery_path and os.path.exists(gallery_path):
            self.load_gallery(gallery_path)
            # Set next_id to be one more than the highest ID in the gallery
            if self.gallery:
                try:
                    numeric_ids = [int(id) for id in self.gallery.keys() if str(id).isdigit()]
                    if numeric_ids:
                        self.next_id = max(numeric_ids) + 1
                except:
                    pass  # Keep default if something goes wrong
    
    def reset_frame_tracking(self, frame_index):
        """Reset frame-level tracking for new frame to prevent duplicate IDs"""
        if frame_index != self.current_frame_index:
            if self.frame_assigned_ids:
                print(f"Frame {self.current_frame_index}: Assigned IDs {sorted(list(self.frame_assigned_ids))}")
            self.current_frame_index = frame_index
            self.frame_assigned_ids.clear()
    
    def get_or_assign_identity(self, query_embedding, threshold=0.98, force_new=False, frame_index=0):
        """
        Match against gallery or assign new identity if no match found
        
        Args:
            query_embedding: The embedding to match
            threshold: Similarity threshold for matching (higher = stricter)
            force_new: Force creation of a new identity regardless of matches
            frame_index: Current frame index for adaptive thresholding
                
        Returns:
            tuple: (person_id, confidence, is_new_identity)
        """
        # Reset frame tracking if this is a new frame
        self.reset_frame_tracking(frame_index)
        
        # Always assign new identities during gallery building phase or if forced
        if force_new:
            new_id = str(self.next_id)
            self.next_id += 1
            self.gallery[new_id] = [query_embedding]  # Store as a list with one embedding
            self.frame_assigned_ids.add(new_id)
            return new_id, 0.0, True
            
        # Find match in existing gallery, but exclude IDs already assigned this frame
        match_id, confidence = self.find_match(query_embedding, threshold, exclude_ids=self.frame_assigned_ids)
        
        if match_id is not None:
            # Update the embedding for this identity with the new one (replace, don't accumulate)
            self.gallery[match_id] = [query_embedding]  # Replace with latest
            self.frame_assigned_ids.add(match_id)
            return match_id, confidence, False
        else:
            # No match found, assign new identity
            new_id = str(self.next_id)
            self.next_id += 1
            self.gallery[new_id] = [query_embedding]  # Store as a list with one embedding
            self.frame_assigned_ids.add(new_id)
            return new_id, 0.0, True

    def update_embedding(self, identity_id, new_embedding, weight=0.3):
        """
        Update an identity's embedding with a new one using weighted average
        
        Args:
            identity_id: Identity ID to update
            new_embedding: New embedding tensor
            weight: Weight for new embedding (0-1)
        """
        if identity_id in self.gallery and self.gallery[identity_id]:
            # Get current aggregated embedding
            current = self.get_aggregated_embedding(identity_id)
            
            if current is not None:
                # Perform weighted average update
                updated = (1-weight) * current + weight * new_embedding
                
                # Replace with normalized version
                import torch.nn.functional as F
                updated = F.normalize(updated, p=2, dim=0)
                
                # Store as the new embedding
                self.gallery[identity_id] = [updated]
                return True
        
        return False

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
                
            # Stack embeddings and compute quality scores (L2 norm as confidence)
            embeddings = torch.stack(valid_embeddings)
            quality_scores = torch.norm(embeddings, dim=1)
            weights = F.softmax(quality_scores, dim=0)
            
            # Weighted aggregation
            aggregated = torch.sum(embeddings * weights.unsqueeze(1), dim=0)
            
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
            return None
    
    def find_match(self, query_embedding, threshold=0.6, exclude_ids=None):
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
                
            if exclude_ids is None:
                exclude_ids = set()
                
            best_score = -1
            best_match = None
            
            for person_id in self.gallery:
                # Skip IDs that are already assigned in this frame
                if person_id in exclude_ids:
                    continue
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

    def save_gallery(self):
        """Save gallery to file"""
        if self.gallery_path:
            try:
                # Save the entire object, not just the gallery dict
                with open(self.gallery_path, 'wb') as f:
                    pickle.dump(self, f)
                print(f"Gallery saved successfully to {self.gallery_path}")
                print(f"Gallery contains {len(self.gallery)} identities")
                return True
            except Exception as e:
                print(f"Error saving gallery: {e}")
        return False

    def load_gallery(self, gallery_path):
        """Load gallery from file"""
        if os.path.exists(gallery_path):
            try:
                with open(gallery_path, 'rb') as f:
                    loaded_obj = pickle.load(f)
                    
                # Copy all attributes from loaded object
                self.gallery = loaded_obj.gallery
                self.next_id = getattr(loaded_obj, 'next_id', 1)
                self.track_to_identity = getattr(loaded_obj, 'track_to_identity', {})
                
                print(f"Loaded gallery from {gallery_path} with {len(self.gallery)} identities")
                return True
            except Exception as e:
                print(f"Error loading gallery: {e}")
        return False