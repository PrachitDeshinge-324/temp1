#!/usr/bin/env python3
# read_gallery.py - Display information about the saved gait embedding gallery

import os
import pickle
import torch
import numpy as np
from utils.gait_gallery import GaitGallery

def main():
    # Path to the gallery file
    gallery_path = "gait_gallery.pkl"
    
    # Check if the gallery file exists
    if not os.path.exists(gallery_path):
        print(f"Error: Gallery file '{gallery_path}' not found!")
        return
    
    # Load the gallery
    print(f"Loading gallery from {gallery_path}...")
    gallery = GaitGallery(gallery_path)
    
    # Display gallery information
    print("\n=== Gait Embedding Gallery Information ===")
    print(f"Number of people in gallery: {len(gallery.gallery)}")
    
    # Display information about each person
    for person_id, embeddings in gallery.gallery.items():
        print(f"\nPerson ID: {person_id}")
        print(f"  Number of embeddings: {len(embeddings)}")
        
        if embeddings:
            # Get the first embedding shape
            first_embedding = embeddings[0]
            print(f"  Embedding shape: {first_embedding.shape}")
            
            # Get aggregated embedding
            agg_embedding = gallery.get_aggregated_embedding(person_id)
            print(f"  Aggregated embedding shape: {agg_embedding.shape}")
            
            # Display some statistics
            norms = [emb.norm().item() for emb in embeddings]
            print(f"  Embedding norms: min={min(norms):.4f}, max={max(norms):.4f}, avg={sum(norms)/len(norms):.4f}")
            
            # If there are multiple embeddings, show similarity between them
            if len(embeddings) > 1:
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = torch.cosine_similarity(
                            embeddings[i].unsqueeze(0),
                            embeddings[j].unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                
                if similarities:
                    print(f"  Intra-person similarities: min={min(similarities):.4f}, max={max(similarities):.4f}, avg={sum(similarities)/len(similarities):.4f}")
    
    print("\n=== Cross-Person Similarity Analysis ===")
    # Calculate similarity between different people
    person_ids = list(gallery.gallery.keys())
    if len(person_ids) > 1:
        cross_similarities = []
        for i, id1 in enumerate(person_ids):
            emb1 = gallery.get_aggregated_embedding(id1)
            if emb1 is None:
                continue
                
            for j, id2 in enumerate(person_ids[i+1:], i+1):
                emb2 = gallery.get_aggregated_embedding(id2)
                if emb2 is None:
                    continue
                    
                sim = torch.cosine_similarity(
                    emb1.unsqueeze(0),
                    emb2.unsqueeze(0)
                ).item()
                cross_similarities.append((id1, id2, sim))
        
        if cross_similarities:
            # Sort by similarity (highest first)
            cross_similarities.sort(key=lambda x: x[2], reverse=True)
            
            print("Top 5 most similar different people:")
            for i, (id1, id2, sim) in enumerate(cross_similarities[:5]):
                print(f"  {i+1}. Person {id1} - Person {id2}: {sim:.4f}")
            
            print("\nOverall cross-person similarity statistics:")
            all_sims = [s[2] for s in cross_similarities]
            print(f"  Min: {min(all_sims):.4f}, Max: {max(all_sims):.4f}, Avg: {sum(all_sims)/len(all_sims):.4f}")
    else:
        print("Not enough people in the gallery for cross-person analysis.")

if __name__ == "__main__":
    main()
