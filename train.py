import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

# Import models (uncomment the one you want to use)
from models.stgcn_model import create_stgcn_model
# from models.tcn_model import GaitTCN
# from models.cnn_bilstm_model import CNNBiLSTMGaitModel
# from models.transformer_model import GaitTransformer

class GaitDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Make sure features are in the format expected by the model
        return torch.FloatTensor(self.features[idx]), self.labels[idx]

def prepare_data(npy_file, id_to_name_file, seq_length=30):
    """Prepare data for training from the merged data"""
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
        
        # Create sliding window sequences
        for i in range(0, len(rows) - seq_length + 1, seq_length // 2):  # 50% overlap
            seq = np.array(rows[i:i+seq_length])
            # Remove track_id and frame_idx columns
            features = seq[:, 2:]
            
            # Reshape features to match the expected input format for ST-GCN
            # ST-GCN expects input in format: [N, C, T, V]
            # N = batch size, C = channels, T = sequence length, V = nodes/joints
            # Assuming features shape is [seq_length, feature_dim]
            feature_dim = features.shape[1]
            
            # Reshape to create 2 channels by splitting the features
            # If feature_dim is odd, pad with zeros
            if feature_dim % 2 != 0:
                padding = np.zeros((seq_length, 1))
                features = np.concatenate((features, padding), axis=1)
                feature_dim += 1
                
            half_dim = feature_dim // 2
            features_reshaped = np.stack([features[:, :half_dim], features[:, half_dim:]], axis=0)  # [2, seq_length, half_dim]
            features_reshaped = np.transpose(features_reshaped, (0, 1, 2))  # [2, seq_length, half_dim]
            
            sequences.append(features_reshaped)
            labels.append(id_to_label[track_id])
    
    return np.array(sequences), np.array(labels), id_to_name, id_to_label

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Train the model"""
    # Move model to device
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    npy_file = 'results/industrial_gait_features_flat.npy'
    id_to_name_file = 'results/id_to_name.json'
    
    # Prepare data
    sequences, labels, id_to_name, id_to_label = prepare_data(npy_file, id_to_name_file)
    
    # Print dataset info
    print(f"Total sequences: {len(sequences)}")
    print(f"Number of classes: {len(id_to_name)}")
    print(f"Classes: {id_to_name}")
    
    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        person_id = list(id_to_label.keys())[list(id_to_label.values()).index(label)]
        name = id_to_name.get(person_id, "Unknown")
        print(f"Class {label} ({name}): {count} samples")
    
    # For small datasets, ensure a minimum number of test samples
    min_samples_per_class = min(counts)
    print(f"Minimum samples for any class: {min_samples_per_class}")
    
    # Determine test_size based on dataset size
    num_classes = len(id_to_name)
    if len(sequences) < num_classes * 5:  # Very small dataset
        print("Very small dataset detected. Using leave-one-out strategy.")
        # Use at least one sample per class for testing
        test_size = max(0.3, num_classes / len(sequences))
    else:
        test_size = 0.2  # Standard 80/20 split
    
    print(f"Using test_size={test_size}")
    
    try:
        # Try with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=test_size, random_state=42, stratify=labels)
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to non-stratified split")
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=test_size, random_state=42)
        
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_val)} samples")
    
    # Create data loaders
    train_dataset = GaitDataset(X_train, torch.LongTensor(y_train))
    val_dataset = GaitDataset(X_val, torch.LongTensor(y_val))

    # Adjust batch size for small datasets
    batch_size = min(32, len(X_train) // 2)
    batch_size = max(1, batch_size)  # Ensure batch_size is at least 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Using batch size: {batch_size}")
    
    # Initialize model (choose one)
    input_dim = sequences.shape[2]  # Number of features per timestep
    num_classes = len(id_to_name)
    
    # Option 1: ST-GCN (for skeleton data)
    model = create_stgcn_model(num_classes)
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=50,
        lr=0.001,
        device=device
    )
    
    print(f"Training complete! Best model saved to 'best_model.pth'")
    print(f"ID to name mapping: {id_to_name}")

if __name__ == "__main__":
    main()