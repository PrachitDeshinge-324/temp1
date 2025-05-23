"""
Data export functionality for gait analysis
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

class GaitDataExporter:
    """Handles exporting gait feature data to various formats"""
    
    @staticmethod
    def export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, feature_order_path, flat_npy_path):
        """Export all collected data to files with improved NaN handling"""
        # Filter out tracks with invalid IDs
        valid_track_ids = [track_id for track_id in gait_analyzer.track_history.keys() if track_id > 0]
        
        # Print statistics about valid and invalid tracks
        print(f"Total tracks: {len(gait_analyzer.track_history)}")
        print(f"Valid tracks (ID > 0): {len(valid_track_ids)}")
        print(f"Invalid tracks: {len(gait_analyzer.track_history) - len(valid_track_ids)}")
        
        # Create a filtered track history with only valid IDs
        filtered_track_history = {
            track_id: gait_analyzer.track_history[track_id] 
            for track_id in valid_track_ids
        }
        
        # Store the original track history
        original_track_history = gait_analyzer.track_history
        
        # Temporarily replace with filtered version for export
        gait_analyzer.track_history = filtered_track_history
        
        # Export features to CSV (now using only valid track IDs)
        gait_analyzer.export_features_csv(features_path)
        print(f"Exported gait features to {features_path}")
        
        # Save bounding box info if requested - also filter out invalid IDs
        if args.save_bbox_info:
            filtered_bbox_info = {
                track_id: bbox_info[track_id]
                for track_id in bbox_info
                if track_id > 0
            }
            with open(bbox_json_path, 'w') as f:
                json.dump(filtered_bbox_info, f)
            print(f"Saved bounding box information to {bbox_json_path}")
        
        # Gather all unique feature keys across all tracks
        all_feature_keys = set()
        for track_id in filtered_track_history:
            # Get both original features and view-invariant features
            features = gait_analyzer.get_features(track_id)
            invariant_features = gait_analyzer.calculate_view_invariant_features(track_id)
            
            if features is not None:
                all_feature_keys.update([k for k, v in features.items() 
                                if isinstance(v, (int, float, np.floating, np.integer))])
            if invariant_features is not None:
                all_feature_keys.update([k for k, v in invariant_features.items() 
                                if isinstance(v, (int, float, np.floating, np.integer))])
        
        # Sort features for consistency
        feature_keys = sorted(all_feature_keys)
        
        # Save the feature order to text file
        with open(feature_order_path, 'w') as f:
            for k in feature_keys:
                f.write(f"{k}\n")
        print(f"Saved feature order to {feature_order_path}")
        print(f"Total features in text file: {len(feature_keys)}")
        
        # First determine the maximum number of keypoints across all tracks
        max_keypoints_count = 0
        for track_id, history in filtered_track_history.items():
            for _, kpts in history:
                max_keypoints_count = max(max_keypoints_count, len(kpts))
        
        # Calculate the expected number of coordinates (x,y per keypoint)
        max_coords = max_keypoints_count * 2
        print(f"Maximum number of keypoints: {max_keypoints_count} (coordinates: {max_coords})")
        
        # Prepare data for numpy array - only using valid track IDs
        all_rows = []
        
        # Store track-specific history for NaN replacement
        track_frame_history = {}
        track_feature_history = {}
        track_keypoint_history = {}
        track_norm_keypoint_history = {}
        
        for track_id, history in filtered_track_history.items():
            # Validate track_id again to be absolutely certain
            if track_id <= 0:
                print(f"Warning: Skipping invalid track_id {track_id} during export")
                continue
            
            # Sort history by frame index to ensure chronological order
            sorted_history = sorted(history, key=lambda x: x[0])
            
            # Initialize history trackers for this track
            track_frame_history[track_id] = []
            track_feature_history[track_id] = []
            track_keypoint_history[track_id] = []
            track_norm_keypoint_history[track_id] = []
            
            for frame_idx, kpts in sorted_history:
                # Get both original and view-invariant features
                features = gait_analyzer.get_features(track_id)
                invariant_features = gait_analyzer.calculate_view_invariant_features(track_id)
                
                # Create combined feature dictionary
                combined_features = {}
                if features is not None:
                    combined_features.update({k: v for k, v in features.items() 
                                          if isinstance(v, (int, float, np.floating, np.integer))})
                if invariant_features is not None:
                    combined_features.update({k: v for k, v in invariant_features.items() 
                                          if isinstance(v, (int, float, np.floating, np.integer))})
                
                # Create feature vector using the same order as in feature_order_path
                feature_vec = np.array([
                    combined_features.get(k, np.nan) for k in feature_keys
                ], dtype=np.float32)
                
                # Initialize arrays for keypoints
                original_kpt_array = np.full(max_coords, np.nan, dtype=np.float32)
                normalized_kpt_array = np.full(max_coords, np.nan, dtype=np.float32)
                
                # Fill in available keypoints at their correct positions
                for i, pt in enumerate(kpts):
                    if isinstance(pt, np.ndarray) and pt.size == 2 and i*2+1 < max_coords:
                        # Place coordinates at the correct indices
                        original_kpt_array[i*2] = pt[0]
                        original_kpt_array[i*2+1] = pt[1]
                
                # Get normalized keypoints
                norm_kpts = gait_analyzer.normalize_keypoints(kpts)
                if norm_kpts is not None:
                    for i, pt in enumerate(norm_kpts):
                        if isinstance(pt, np.ndarray) and pt.size == 2 and i*2+1 < max_coords:
                            normalized_kpt_array[i*2] = pt[0]
                            normalized_kpt_array[i*2+1] = pt[1]
                
                # Fix NaN values using historical data
                # 1. Check for NaN values in keypoints
                if np.isnan(original_kpt_array).any():
                    # If we have enough history, use that to fill NaNs
                    if len(track_keypoint_history[track_id]) > 0:
                        # For each NaN value, try to replace with historical average
                        for i in range(len(original_kpt_array)):
                            if np.isnan(original_kpt_array[i]):
                                # Collect historical values for this coordinate
                                coord_history = [hist[i] for hist in track_keypoint_history[track_id][-30:] 
                                               if i < len(hist) and not np.isnan(hist[i])]
                                if coord_history:  # If we have any valid history
                                    original_kpt_array[i] = np.mean(coord_history)
                
                # 2. Check for NaN values in normalized keypoints
                if np.isnan(normalized_kpt_array).any():
                    # If we have enough history, use that to fill NaNs
                    if len(track_norm_keypoint_history[track_id]) > 0:
                        # For each NaN value, try to replace with historical average
                        for i in range(len(normalized_kpt_array)):
                            if np.isnan(normalized_kpt_array[i]):
                                # Collect historical values for this coordinate
                                coord_history = [hist[i] for hist in track_norm_keypoint_history[track_id][-30:] 
                                               if i < len(hist) and not np.isnan(hist[i])]
                                if coord_history:  # If we have any valid history
                                    normalized_kpt_array[i] = np.mean(coord_history)
                
                # 3. Check for NaN values in feature vector
                if np.isnan(feature_vec).any():
                    # If we have enough history, use that to fill NaNs
                    if len(track_feature_history[track_id]) > 0:
                        # For each NaN value, try to replace with historical average
                        for i in range(len(feature_vec)):
                            if np.isnan(feature_vec[i]):
                                # Collect historical values for this feature
                                feat_history = [hist[i] for hist in track_feature_history[track_id][-30:] 
                                              if i < len(hist) and not np.isnan(hist[i])]
                                if feat_history:  # If we have any valid history
                                    feature_vec[i] = np.mean(feat_history)
                
                # Append current data to history after fixing NaNs
                track_keypoint_history[track_id].append(original_kpt_array.copy())
                track_norm_keypoint_history[track_id].append(normalized_kpt_array.copy())
                track_feature_history[track_id].append(feature_vec.copy())
                track_frame_history[track_id].append(frame_idx)
                
                # Combine all parts to create the row
                row = [float(track_id), float(frame_idx)] + \
                      original_kpt_array.tolist() + \
                      normalized_kpt_array.tolist() + \
                      feature_vec.tolist()
                
                all_rows.append(row)
        
        # Convert to numpy array
        all_rows_np = np.array(all_rows, dtype=np.float32)
        
        # Final pass to replace remaining NaNs with zeros
        nan_count_before = np.isnan(all_rows_np).sum()
        if nan_count_before > 0:
            print(f"Replacing {nan_count_before} remaining NaN values with zeros")
            all_rows_np = np.nan_to_num(all_rows_np, nan=0.0)
        
        # Save the cleaned data
        np.save(flat_npy_path, all_rows_np)
        
        # Calculate total values per row
        total_values_per_row = 2 + max_coords*2 + len(feature_keys)
        
        print(f"Saved flat numpy array with id, frame, keypoints, normalized keypoints, features to {flat_npy_path}")
        print(f"Each row contains: 1 track_id + 1 frame_idx + {max_coords} keypoint values + {max_coords} normalized keypoint values + {len(feature_keys)} features = {total_values_per_row} total values")
        print(f"Total rows: {len(all_rows_np)}")
        
        # Restore original track history
        gait_analyzer.track_history = original_track_history
        return all_rows_np

    @staticmethod
    def fix_invariant_features(gait_analyzer, flat_npy_path, feature_order_path, output_csv_path):
        """Fix invariant features by copying from track 1 to other tracks with proper scaling"""
        print("Fixing missing invariant features for all tracks...")
        
        # Load original CSV to get data with invariant features for track 1
        # First check if the CSV exists - if not, we need to generate it
        original_csv = output_csv_path.replace('_with_invariants.csv', '.csv')
        if not os.path.exists(original_csv):
            print(f"Original CSV {original_csv} not found, generating from analyzer...")
            gait_analyzer.export_features_csv(original_csv)
        
        # Read the CSV with pandas for easier manipulation
        try:
            df = pd.read_csv(original_csv)
            print(f"Loaded CSV with {len(df)} tracks")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
        
        # Identify invariant feature columns (those starting with 'inv_')
        inv_columns = [col for col in df.columns if col.startswith('inv_')]
        print(f"Found {len(inv_columns)} invariant feature columns")
        
        if len(inv_columns) == 0:
            print("No invariant features found in the CSV!")
            return False
        
        # Find template track with the most non-zero invariant features
        template_track = None
        max_nonzero = -1
        
        for _, row in df.iterrows():
            track_id = row['track_id']
            nonzero_count = sum(1 for col in inv_columns if row[col] != 0)
            
            if nonzero_count > max_nonzero:
                max_nonzero = nonzero_count
                template_track = row
        
        if template_track is None or max_nonzero == 0:
            print("No tracks found with non-zero invariant features!")
            return False
        
        print(f"Using track {int(template_track['track_id'])} as template with {max_nonzero} non-zero invariant features")
        
        # Create a copy of the dataframe for modification
        df_fixed = df.copy()
        
        # For each track that's missing invariant features, copy and scale them from the template
        for idx, row in df_fixed.iterrows():
            track_id = int(row['track_id'])
            if track_id == template_track['track_id']:
                continue  # Skip template track
                
            missing_features = sum(1 for col in inv_columns if row[col] == 0)
            if missing_features > 0:
                print(f"Fixing {missing_features} invariant features for track {track_id}")
                
                # Calculate scaling factors based on height/torso length ratio
                # We'll use avg_neck_to_left_shoulder_length as an approximation
                template_scale = template_track.get('avg_neck_to_left_shoulder_length', 1.0)
                target_scale = row.get('avg_neck_to_left_shoulder_length', 1.0)
                
                if template_scale > 0 and target_scale > 0:
                    scale_ratio = target_scale / template_scale
                else:
                    scale_ratio = 1.0
                    
                print(f"  Scale ratio: {scale_ratio:.2f}")
                
                # Copy each invariant feature with scaling
                for col in inv_columns:
                    if row[col] == 0:  # Only replace if the value is missing
                        # Apply different scaling strategies based on feature type
                        template_value = template_track[col]
                        
                        if 'angle' in col:  # Angles should remain the same regardless of scale
                            df_fixed.at[idx, col] = template_value
                        elif 'ratio' in col:  # Ratios should remain roughly the same
                            df_fixed.at[idx, col] = template_value
                        else:  # Other features like lengths should be scaled
                            df_fixed.at[idx, col] = template_value * scale_ratio
        
        # Save the fixed dataframe as a new CSV
        df_fixed.to_csv(output_csv_path, index=False)
        print(f"Saved fixed CSV with invariant features to {output_csv_path}")
        return True

    @staticmethod
    def generate_processed_csv(preprocessed_npy, output_csv_path, feature_order_path):
        """Generate a clean CSV without NaN values from preprocessed numpy data"""
        print(f"Generating processed CSV file at {output_csv_path}")
        
        # Load preprocessed data
        data = np.load(preprocessed_npy)
        
        # Read feature names from feature order file
        with open(feature_order_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Calculate column offsets
        track_id_col = 0
        frame_idx_col = 1
        
        # Calculate the number of keypoint columns
        remaining_cols = data.shape[1] - 2 - len(feature_names)
        n_keypoint_coords = remaining_cols // 2
        
        # Features start after track_id, frame_idx, original keypoints, and normalized keypoints
        feature_start_col = 2 + n_keypoint_coords * 2
        
        # Extract unique track IDs
        unique_track_ids = np.unique(data[:, track_id_col]).astype(int)
        
        # Dictionary to hold averaged features for each track
        track_features = {}
        
        # For each track, calculate average feature values
        for track_id in unique_track_ids:
            # Get all rows for this track
            track_mask = data[:, track_id_col] == track_id
            track_data = data[track_mask]
            
            # Calculate mean and std for each feature
            feature_means = np.mean(track_data[:, feature_start_col:], axis=0)
            feature_stds = np.std(track_data[:, feature_start_col:], axis=0)
            
            # Store in dictionary
            track_features[track_id] = {'means': feature_means, 'stds': feature_stds}
        
        # Write CSV header (track_id and feature names)
        with open(output_csv_path, 'w') as f:
            header = ['track_id'] + feature_names
            f.write(','.join(header) + '\n')
            
            # Write a row for each unique track
            for track_id in unique_track_ids:
                # Start with track ID
                row = [str(int(track_id))]
                
                # Add each feature value
                for i, feature in enumerate(feature_names):
                    # Some features might be averages, others might be raw values
                    # Use the means we calculated
                    if i < len(track_features[track_id]['means']):
                        row.append(str(track_features[track_id]['means'][i]))
                    else:
                        row.append('0.0')  # Fallback value if missing
                
                # Write the row
                f.write(','.join(row) + '\n')
        
        print(f"CSV created with data for {len(unique_track_ids)} tracks")
        return True
