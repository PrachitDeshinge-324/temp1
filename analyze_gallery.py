#!/usr/bin/env python3
# analyze_gallery.py - A more detailed analysis of the gait embedding gallery

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.gait_gallery import GaitGallery

def analyze_gallery(gallery_path="gait_gallery.pkl"):
    """Analyze the gait embedding gallery and display useful information."""
    
    # Check if the gallery file exists
    if not os.path.exists(gallery_path):
        print(f"Error: Gallery file '{gallery_path}' not found!")
        return
    
    # Load the gallery
    print(f"Loading gallery from {gallery_path}...")
    gallery = GaitGallery(gallery_path)
    
    # Basic gallery info
    print("\n=== Gait Embedding Gallery Summary ===")
    print(f"Number of people in gallery: {len(gallery.gallery)}")
    total_embeddings = sum(len(embs) for embs in gallery.gallery.values())
    print(f"Total number of embeddings: {total_embeddings}")
    
    # Show info for each person
    for person_id, embeddings in gallery.gallery.items():
        print(f"\nPerson ID: {person_id}")
        print(f"  Number of embeddings: {len(embeddings)}")
        
        if embeddings:
            first_embedding = embeddings[0]
            embedding_dim = first_embedding.shape[1] if len(first_embedding.shape) > 1 else first_embedding.shape[0]
            print(f"  Embedding dimension: {embedding_dim}")
            
            # Get normalized embeddings for this person
            norm_embeddings = []
            for e in embeddings:
                # Ensure the embedding is a 1D tensor for cosine similarity
                if len(e.shape) > 1:
                    e = e.squeeze()  # Remove extra dimensions
                norm_embeddings.append(e / e.norm())
            
            # Calculate intra-similarity matrix if there are multiple embeddings
            if len(embeddings) > 1:
                intra_similarities = []
                for i in range(len(norm_embeddings)):
                    for j in range(i+1, len(norm_embeddings)):
                        sim = torch.cosine_similarity(
                            norm_embeddings[i].unsqueeze(0),
                            norm_embeddings[j].unsqueeze(0)
                        ).item()
                        intra_similarities.append(sim)
                
                print("  Intra-person similarity statistics:")
                print(f"    Min: {min(intra_similarities):.4f}")
                print(f"    Max: {max(intra_similarities):.4f}")
                print(f"    Avg: {sum(intra_similarities) / len(intra_similarities):.4f}")
                print(f"    Std: {np.std(intra_similarities):.4f}")
    
    # Inter-person similarity analysis
    print("\n=== Inter-Person Similarity Analysis ===")
    person_ids = list(gallery.gallery.keys())
    
    if len(person_ids) > 1:
        # Get aggregated embeddings for each person
        agg_embeddings = {}
        for person_id in person_ids:
            agg_emb = gallery.get_aggregated_embedding(person_id)
            if agg_emb is not None:
                # Ensure the embedding is a 1D tensor for cosine similarity
                if len(agg_emb.shape) > 1:
                    agg_emb = agg_emb.squeeze()  # Remove extra dimensions
                agg_embeddings[person_id] = agg_emb
        
        # Calculate similarity between different people
        if len(agg_embeddings) > 1:
            inter_similarities = []
            for i, id1 in enumerate(agg_embeddings.keys()):
                emb1 = agg_embeddings[id1]
                remaining_ids = list(agg_embeddings.keys())[i+1:]
                
                for id2 in remaining_ids:
                    emb2 = agg_embeddings[id2]
                    sim = torch.cosine_similarity(
                        emb1.unsqueeze(0),
                        emb2.unsqueeze(0)
                    ).item()
                    inter_similarities.append((id1, id2, sim))
                    print(f"  Similarity between Person {id1} and Person {id2}: {sim:.4f}")
            
            if inter_similarities:
                all_sims = [s[2] for s in inter_similarities]
                print("\nInter-person similarity statistics:")
                print(f"  Min: {min(all_sims):.4f}")
                print(f"  Max: {max(all_sims):.4f}")
                print(f"  Avg: {sum(all_sims) / len(all_sims):.4f}")
                print(f"  Std: {np.std(all_sims):.4f}")
        else:
            print("  Not enough aggregated embeddings for inter-person analysis.")
    else:
        print("  Not enough people in the gallery for inter-person analysis.")
    
    # Perform PCA visualization of embeddings if matplotlib is available
    try:
        from sklearn.decomposition import PCA
        
        print("\n=== Embedding Visualization with PCA ===")
        
        # Collect all embeddings and their labels
        all_embeddings = []
        all_labels = []
        
        for person_id, embeddings in gallery.gallery.items():
            for emb in embeddings:
                all_embeddings.append(emb.numpy())
                all_labels.append(person_id)
        
        if all_embeddings:
            # Convert to numpy array
            all_embeddings = np.array(all_embeddings)
            
            # Reshape if needed (ensure 2D)
            if len(all_embeddings.shape) > 2:
                all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)
            
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(all_embeddings)
            
            # Plot PCA results
            plt.figure(figsize=(10, 8))
            
            # Get unique person IDs for coloring
            unique_ids = set(all_labels)
            
            for person_id in unique_ids:
                indices = [i for i, label in enumerate(all_labels) if label == person_id]
                points = reduced_embeddings[indices]
                plt.scatter(points[:, 0], points[:, 1], label=f'Person {person_id}', alpha=0.7)
            
            plt.title('PCA Visualization of Gait Embeddings')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            output_path = 'gait_embeddings_pca.png'
            plt.savefig(output_path)
            print(f"PCA visualization saved to: {output_path}")
            
            # Try to display if in interactive environment
            try:
                plt.show()
            except:
                pass
        else:
            print("  No embeddings available for PCA visualization.")
    except ImportError:
        print("  Matplotlib or scikit-learn not available for visualization.")
    except Exception as e:
        print(f"  Error in PCA visualization: {e}")

if __name__ == "__main__":
    analyze_gallery()
