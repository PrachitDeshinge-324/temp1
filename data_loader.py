import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split

class GaitDataset(Dataset):
    """Dataset for gait recognition"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]

def prepare_data_lstm(npy_file, id_to_name_file, seq_length=30, stride=15):
    """
    Prepare data for LSTM model with sliding window approach
    
    Args:
        npy_file: Path to numpy file with gait data
        id_to_name_file: Path to JSON mapping IDs to names
        seq_length: Length of sequences to create
        stride: Stride for sliding window
    
    Returns:
        sequences: numpy array of shape (num_sequences, seq_length, feature_dim)
        labels: numpy array of shape (num_sequences,)
        id_to_name: dict mapping IDs to names
        id_to_label: dict mapping IDs to label indices
    """
    # Load data
    data = np.load(npy_file)
    
    with open(id_to_name_file, 'r') as f:
        id_to_name = json.load(f)
    
    # Convert string keys to integers
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    
    # Create mapping from ID to label index
    unique_ids = sorted(list(id_to_name.keys()))
    id_to_label = {id_val: i for i, id_val in enumerate(unique_ids)}
    
    # Group data by ID and frame
    sequences = []
    labels = []
    
    # Group by track_id
    track_groups = {}
    for row in data:
        track_id = int(row[0])
        if track_id not in track_groups:
            track_groups[track_id] = []
        track_groups[track_id].append(row)
    
    # Create sequences of length seq_length for each track
    for track_id, rows in track_groups.items():
        # Only use tracks that have a name
        if track_id not in id_to_name:
            continue
        
        # Sort by frame index
        rows.sort(key=lambda x: x[1])
        
        # Create sliding window sequences with overlap
        for i in range(0, len(rows) - seq_length + 1, stride):
            seq = np.array(rows[i:i+seq_length])
            # Remove track_id and frame_idx columns
            features = seq[:, 2:].astype(np.float32)  # Keep only features
            
            # Apply data normalization (Z-score normalization per sequence)
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
            features = (features - mean) / std
            
            sequences.append(features)
            labels.append(id_to_label[track_id])
    
    return np.array(sequences), np.array(labels), id_to_name, id_to_label

def create_data_loaders(sequences, labels, batch_size=16, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create train/val/test data loaders with stratification
    
    Args:
        sequences: numpy array of sequences
        labels: numpy array of labels
        batch_size: batch size for dataloaders
        val_ratio: ratio of data for validation
        test_ratio: ratio of data for testing
        random_state: random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        class_weights: tensor of class weights for weighted loss function
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=test_ratio, random_state=random_state, stratify=labels)
    
    # Then split the remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio/(1-test_ratio), 
        random_state=random_state, stratify=y_temp)
    
    # Create datasets
    train_dataset = GaitDataset(X_train, y_train)
    val_dataset = GaitDataset(X_val, y_val)
    test_dataset = GaitDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights for weighted loss function
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return train_loader, val_loader, test_loader, class_weights