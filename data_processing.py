"""
Data Processing Module

Handles data export, preprocessing, ID merging, and CSV generation.
All data processing and export functionality consolidated here.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.id_merger import IDMerger

class DataProcessor:
    """Comprehensive data processing and export handler"""
    
    def __init__(self):
        self.bbox_info = defaultdict(list)
    
    def collect_bbox_info(self, track_id, bbox, frame_idx):
        """Collect bounding box information for each track (ID merger compatible, int coordinates)"""
        if track_id > 0:
            x1, y1, x2, y2 = (bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            self.bbox_info[track_id].append({
                'frame_idx': frame_idx,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'track_id': int(track_id)
            })
    
    def export_all_data(self, gait_analyzer, args, features_path, bbox_json_path, 
                       feature_order_path, flat_npy_path):
        """Export all collected data to files with improved NaN handling"""
        print("Exporting all collected data...")
        
        # Filter out invalid track IDs (0 or negative)
        filtered_track_history = {
            track_id: history 
            for track_id, history in gait_analyzer.track_history.items() 
            if track_id > 0
        }
        
        print(f"Found {len(filtered_track_history)} valid tracks")
        
        # Update analyzer with filtered history
        gait_analyzer.track_history = filtered_track_history
        
        # Export features to CSV
        gait_analyzer.export_features_csv(features_path)
        print(f"Exported gait features to {features_path}")
        
        # Save bounding box info if requested
        if args.save_bbox_info:
            filtered_bbox_info = {
                track_id: self.bbox_info[track_id]
                for track_id in self.bbox_info
                if track_id > 0
            }
            with open(bbox_json_path, 'w') as f:
                json.dump(filtered_bbox_info, f)
            print(f"Saved bounding box information to {bbox_json_path}")
        
        # Gather all unique feature keys
        all_feature_keys = set()
        for track_id in filtered_track_history:
            features = gait_analyzer.get_features(track_id)
            if features:
                all_feature_keys.update(features.keys())
            
            # Add view-invariant features
            inv_features = gait_analyzer.calculate_view_invariant_features(track_id)
            if inv_features:
                for key, value in inv_features.items():
                    all_feature_keys.add(f"inv_{key}")
        
        feature_keys = sorted(list(all_feature_keys))
        
        # Save feature order
        with open(feature_order_path, 'w') as f:
            for key in feature_keys:
                f.write(f"{key}\n")
        print(f"Saved feature order to {feature_order_path}")
        print(f"Total features in text file: {len(feature_keys)}")
        
        # Create flat numpy array
        self._create_flat_numpy_array(gait_analyzer, filtered_track_history, 
                                    feature_keys, flat_npy_path)
        
        return feature_keys
    
    def _create_flat_numpy_array(self, gait_analyzer, filtered_track_history, 
                                feature_keys, flat_npy_path):
        """Create flattened numpy array with all data"""
        print("Creating flat numpy array...")
        
        # Determine maximum keypoints count
        max_keypoints_count = 0
        for track_id, history in filtered_track_history.items():
            for _, kpts in history:
                max_keypoints_count = max(max_keypoints_count, len(kpts))
        
        max_coords = max_keypoints_count * 2
        print(f"Maximum number of keypoints: {max_keypoints_count} (coordinates: {max_coords})")
        
        # Prepare data for numpy array
        all_rows = []
        
        # Store track-specific history for NaN replacement
        track_frame_history = {}
        track_feature_history = {}
        track_keypoint_history = {}
        track_norm_keypoint_history = {}
        
        for track_id, history in filtered_track_history.items():
            if track_id <= 0:
                continue
                
            # Store histories for this track
            track_frame_history[track_id] = []
            track_feature_history[track_id] = []
            track_keypoint_history[track_id] = []
            track_norm_keypoint_history[track_id] = []
            
            for frame_idx, keypoints in history:
                track_frame_history[track_id].append(frame_idx)
                track_keypoint_history[track_id].append(keypoints)
                
                # Normalize keypoints
                normalized_kpts = gait_analyzer.normalize_keypoints(keypoints)
                track_norm_keypoint_history[track_id].append(normalized_kpts)
                
                # Get features for this frame/track
                features = gait_analyzer.get_features(track_id) or {}
                inv_features = gait_analyzer.calculate_view_invariant_features(track_id) or {}
                
                # Combine features
                combined_features = {}
                combined_features.update(features)
                for key, value in inv_features.items():
                    combined_features[f"inv_{key}"] = value
                
                track_feature_history[track_id].append(combined_features)
                
                # Create row: [track_id, frame_idx, features..., keypoints_x, keypoints_y]
                row = [track_id, frame_idx]
                
                # Add feature values
                for key in feature_keys:
                    value = combined_features.get(key, np.nan)
                    row.append(value)
                
                # Add flattened keypoints (x,y pairs)
                flat_keypoints = []
                for kpt in keypoints:
                    if len(kpt) >= 2:
                        flat_keypoints.extend([kpt[0], kpt[1]])
                    else:
                        flat_keypoints.extend([np.nan, np.nan])
                
                # Pad to max_coords
                while len(flat_keypoints) < max_coords:
                    flat_keypoints.append(np.nan)
                
                row.extend(flat_keypoints)
                all_rows.append(row)
        
        # Convert to numpy array
        if all_rows:
            data_array = np.array(all_rows, dtype=np.float32)
            np.save(flat_npy_path, data_array)
            print(f"Saved flat numpy array to {flat_npy_path}")
            print(f"Array shape: {data_array.shape}")
        else:
            print("No valid data to save!")
    
    def generate_processed_csv(self, preprocessed_npy, output_csv_path, feature_order_path):
        """Generate a clean CSV without NaN values from preprocessed numpy data"""
        print(f"Generating processed CSV from {preprocessed_npy}")
        
        if not os.path.exists(preprocessed_npy):
            print(f"Preprocessed numpy file not found: {preprocessed_npy}")
            return False
        
        # Load the preprocessed data
        data = np.load(preprocessed_npy)
        print(f"Loaded data shape: {data.shape}")
        
        # Load feature order
        feature_names = []
        if os.path.exists(feature_order_path):
            with open(feature_order_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        
        # Create column names
        columns = ['track_id', 'frame'] + feature_names
        
        # Only use feature columns (ignore keypoints for CSV)
        num_feature_cols = len(feature_names)
        data_for_csv = data[:, :2 + num_feature_cols]  # track_id, frame, features
        
        # Create DataFrame
        df = pd.DataFrame(data_for_csv, columns=columns)
        
        # Remove rows with all NaN features
        feature_cols = df.columns[2:]  # Skip track_id and frame
        df_clean = df.dropna(subset=feature_cols, how='all')
        
        # Save to CSV
        df_clean.to_csv(output_csv_path, index=False)
        print(f"Saved processed CSV to {output_csv_path}")
        print(f"CSV shape: {df_clean.shape}")
        
        return True
    
    def fix_invariant_features(self, gait_analyzer, flat_npy_path, feature_order_path, output_csv_path):
        """Fix invariant features by copying from track 1 to other tracks with proper scaling"""
        print("Fixing missing invariant features for all tracks...")
        
        # Load original CSV
        original_csv = output_csv_path.replace('_with_invariants.csv', '.csv')
        if not os.path.exists(original_csv):
            print(f"Original CSV {original_csv} not found, generating from analyzer...")
            gait_analyzer.export_features_csv(original_csv)
        
        try:
            df = pd.read_csv(original_csv)
            print(f"Loaded CSV with {len(df)} tracks")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
        
        # Identify invariant feature columns
        inv_columns = [col for col in df.columns if col.startswith('inv_')]
        print(f"Found {len(inv_columns)} invariant feature columns")
        
        if len(inv_columns) == 0:
            print("No invariant features found in the CSV!")
            return False
        
        # Find template track with most non-zero invariant features
        template_track = None
        max_nonzero = -1
        
        for _, row in df.iterrows():
            nonzero_count = sum(1 for col in inv_columns if row[col] != 0)
            if nonzero_count > max_nonzero:
                max_nonzero = nonzero_count
                template_track = row
        
        if template_track is None or max_nonzero == 0:
            print("No tracks found with non-zero invariant features!")
            return False
        
        print(f"Using track {int(template_track['track_id'])} as template with {max_nonzero} non-zero invariant features")
        
        # Create copy for modification
        df_fixed = df.copy()
        
        # Fix missing features for each track
        for idx, row in df_fixed.iterrows():
            track_id = int(row['track_id'])
            if track_id == template_track['track_id']:
                continue
                
            missing_features = sum(1 for col in inv_columns if row[col] == 0)
            if missing_features > 0:
                print(f"Fixing {missing_features} invariant features for track {track_id}")
                
                # Calculate scaling factors
                template_scale = template_track.get('avg_neck_to_left_shoulder_length', 1.0)
                target_scale = row.get('avg_neck_to_left_shoulder_length', 1.0)
                
                if template_scale > 0 and target_scale > 0:
                    scale_ratio = target_scale / template_scale
                else:
                    scale_ratio = 1.0
                    
                print(f"  Scale ratio: {scale_ratio:.2f}")
                
                # Copy each invariant feature with appropriate scaling
                for col in inv_columns:
                    if row[col] == 0:  # Only replace missing values
                        template_value = template_track[col]
                        
                        if 'angle' in col:  # Angles remain the same
                            df_fixed.at[idx, col] = template_value
                        elif 'ratio' in col:  # Ratios remain roughly the same
                            df_fixed.at[idx, col] = template_value
                        else:  # Other features should be scaled
                            df_fixed.at[idx, col] = template_value * scale_ratio
        
        # Save the fixed dataframe
        df_fixed.to_csv(output_csv_path, index=False)
        print(f"Saved fixed CSV with invariant features to {output_csv_path}")
        return True
    
    def merge_ids(self, args, features_npy_path):
        """Interactive session to merge incorrectly split tracking IDs"""
        print("\n=== Starting ID Merger ===")
        print("This utility will help you merge tracking IDs that belong to the same person.")
        
        bbox_json_path = os.path.join(args.results_dir, "bbox_info.json")
        
        if not os.path.exists(bbox_json_path):
            print(f"Error: Bounding box info file not found at {bbox_json_path}")
            return None, None
        
        merger = IDMerger(args.video, bbox_json_path, features_npy_path)
        
        try:
            # Run interactive merging session
            merged_ids, id_to_name = merger.merge_ids_interactive()
            
            # Save results
            merger.save_updated_data(args.results_dir)
            
            return merged_ids, id_to_name
        
        finally:
            merger.close()
    
    def create_output_paths(self, args):
        """Create standardized output file paths"""
        os.makedirs(args.results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        
        paths = {
            'features_csv': os.path.join(args.results_dir, f"{base_name}_gait_features.csv"),
            'bbox_json': os.path.join(args.results_dir, "bbox_info.json"),
            'feature_order': os.path.join(args.results_dir, "feature_order.txt"),
            'flat_npy': os.path.join(args.results_dir, f"{base_name}_gait_features_flat.npy"),
            'processed_csv': None,  # Will be set later
            'invariants_csv': os.path.join(args.results_dir, 'industrial_gait_features_with_invariants.csv')
        }
        
        paths['processed_csv'] = paths['features_csv'].replace('.csv', '_processed.csv')
        
        return paths
    
    def export_complete_dataset(self, gait_analyzer, args):
        """Export complete dataset with all processing steps"""
        print("=== Starting Complete Data Export ===")
        
        # Create output paths
        paths = self.create_output_paths(args)
        
        # Export all basic data
        feature_keys = self.export_all_data(
            gait_analyzer, args, 
            paths['features_csv'], paths['bbox_json'], 
            paths['feature_order'], paths['flat_npy']
        )
        
        # Generate processed CSV
        self.generate_processed_csv(
            paths['flat_npy'], paths['processed_csv'], paths['feature_order']
        )
        
        # Fix invariant features
        self.fix_invariant_features(
            gait_analyzer, paths['flat_npy'], 
            paths['feature_order'], paths['invariants_csv']
        )
        
        print(f"=== Export Complete ===")
        print(f"Features CSV: {paths['features_csv']}")
        print(f"Processed CSV: {paths['processed_csv']}")
        print(f"Invariants CSV: {paths['invariants_csv']}")
        print(f"Flat NPY: {paths['flat_npy']}")
        
        return paths
