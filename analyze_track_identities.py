#!/usr/bin/env python3
# filepath: /Users/prachit/self/Working/Person_Temp/analyze_track_identities.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_track_identity_mapping(gallery_path="gait_gallery.pkl"):
    """Analyze how tracks map to identities."""
    
    # Check if the gallery file exists
    if not os.path.exists(gallery_path):
        print(f"Error: Gallery file '{gallery_path}' not found!")
        return
    
    # Load the gallery
    print(f"Loading gallery from {gallery_path}...")
    with open(gallery_path, 'rb') as f:
        gallery_data = pickle.load(f)
    
    # Check if track_to_identity exists
    if not hasattr(gallery_data, 'track_to_identity') or not gallery_data.track_to_identity:
        print("No track-to-identity mapping found in the gallery!")
        return
    
    track_to_identity = gallery_data.track_to_identity
    
    # Analysis
    print("\n=== Track to Identity Mapping Analysis ===")
    print(f"Total tracks: {len(track_to_identity)}")
    
    # Count identities
    identities = list(track_to_identity.values())
    identity_count = Counter(identities)
    print(f"Total unique identities: {len(identity_count)}")
    
    # Show distribution
    print("\nIdentity Distribution:")
    for identity, count in identity_count.items():
        print(f"  Identity {identity}: {count} tracks")
    
    # Check for potential identity confusion
    if len(identity_count) < len(track_to_identity) / 2:
        print("\nWARNING: There may be identity confusion! Too few identities for the number of tracks.")
    
    # Visualize the mapping
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot as a scatter plot where x = track_id, y = identity_id
        tracks = list(track_to_identity.keys())
        identities = [track_to_identity[t] for t in tracks]
        
        # Convert to numeric for plotting
        tracks_num = np.array([int(t) for t in tracks])
        identities_num = np.array([int(i) for i in identities])
        
        plt.scatter(tracks_num, identities_num, alpha=0.8, s=100)
        
        # Add labels and grid
        plt.xlabel('Track ID')
        plt.ylabel('Identity ID')
        plt.title('Track to Identity Mapping')
        plt.grid(True, alpha=0.3)
        
        # Add text labels
        for i, (t, id) in enumerate(zip(tracks_num, identities_num)):
            plt.annotate(f"T{t}", (t, id), fontsize=9)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('track_identity_mapping.png')
        print("Mapping visualization saved to: track_identity_mapping.png")
        
        try:
            plt.show()
        except:
            pass
            
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    analyze_track_identity_mapping()