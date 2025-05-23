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
    track_ids = data[:, 0].astype(int)
    valid_track_ids = [tid for tid in np.unique(track_ids) if tid > 0]
    sequences = []
    labels = []
    for tid in valid_track_ids:
        tid_rows = data[track_ids == tid]
        # Optionally, sort by frame index if needed
        tid_rows = tid_rows[np.argsort(tid_rows[:, 1])]
        # Sliding window
        for start in range(0, len(tid_rows) - seq_length + 1, stride):
            seq = tid_rows[start:start+seq_length]
            if seq.shape[0] == seq_length:
                sequences.append(seq)
                labels.append(tid)
    # Load id_to_name and build id_to_label
    with open(id_to_name_file) as f:
        id_to_name = json.load(f)
    id_to_label = {int(k): i for i, k in enumerate(sorted(id_to_name.keys(), key=int))}
    return sequences, labels, id_to_name, id_to_label

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