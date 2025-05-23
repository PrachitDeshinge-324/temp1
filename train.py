import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from models.lstm_model import GaitLSTM, CNNLSTMGait
from data_loader import prepare_data_lstm, create_data_loaders
from utils.train import train_one_epoch, validate, test_model, plot_training_curves

def train_lstm_model(model, train_loader, val_loader, test_loader=None, 
                     criterion=None, num_epochs=100, lr=0.001, weight_decay=1e-5, 
                     device='cuda', save_dir='./models/saved',
                     id_to_name=None, id_to_label=None):
    """
    Train LSTM model with advanced techniques to prevent overfitting
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data (optional)
        criterion: Loss function (if None, CrossEntropyLoss will be used)
        num_epochs: Maximum number of epochs to train
        lr: Initial learning rate
        weight_decay: L2 regularization strength
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model and results
        id_to_name: Dictionary mapping IDs to names
        id_to_label: Dictionary mapping IDs to label indices
    """
    # Create directory to save models and results
    os.makedirs(save_dir, exist_ok=True)
    
    # Create TensorBoard writer
    log_dir = os.path.join(save_dir, 'tensorboard', datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    # Default loss function if none provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training loop with early stopping
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}% | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'* Model saved at epoch {epoch+1} with validation accuracy {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir)
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Test on test set if provided
    if test_loader is not None and id_to_name is not None and id_to_label is not None:
        # Load best model
        checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        # Test model
        test_loss, test_acc, report = test_model(model, test_loader, criterion, device, id_to_name, id_to_label, save_dir)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
        
        # Save test results
        with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
            f.write("Classification Report:\n")
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                f.write(f"{class_name}: Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, "
                        f"F1-Score={metrics['f1-score']:.4f}\n")
    
    return model, train_losses, val_losses, train_accs, val_accs

def preprocess_nan_values(data_path):
    """
    Preprocess NaN values in the gait data by replacing them with
    track-specific averages for each feature.
    
    Args:
        data_path: Path to the numpy data file
        
    Returns:
        Preprocessed numpy array with NaN values replaced
    """
    print("Loading raw data...")
    data = np.load(data_path)
    
    print(f"Raw data shape: {data.shape}")
    print(f"NaN values before preprocessing: {np.isnan(data).sum()}")
    
    # The first column contains track IDs
    track_ids = data[:, 0].astype(int)
    unique_tracks = np.unique(track_ids)
    print(f"Found {len(unique_tracks)} unique tracks")
    
    # Create a copy of the data to avoid modifying the original
    preprocessed_data = data.copy()
    
    # Process each track separately
    for track_id in unique_tracks:
        # Get all rows for this track
        track_mask = track_ids == track_id
        track_data = data[track_mask]
        
        # Skip processing if there are no NaN values in this track
        if not np.isnan(track_data).any():
            continue
            
        print(f"Processing track {track_id} with {np.isnan(track_data).sum()} NaN values")
        
        # Calculate column means for this track, ignoring NaN values
        col_means = np.nanmean(track_data, axis=0)
        
        # For columns where all values are NaN, use zeros instead
        col_means = np.nan_to_num(col_means, nan=0.0)
        
        # Create a mask for NaN values in this track's data
        nan_mask = np.isnan(track_data)
        
        # Create an array with the same shape as track_data, filled with column means
        means_array = np.tile(col_means, (track_data.shape[0], 1))
        
        # Use the mask to replace only NaN values with corresponding means
        track_data_filled = np.where(nan_mask, means_array, track_data)
        
        # Update the preprocessed data for this track
        preprocessed_data[track_mask] = track_data_filled
    
    # Double-check no NaN values remain
    remaining_nans = np.isnan(preprocessed_data).sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} NaN values remain after track-based preprocessing")
        print("Filling remaining NaNs with zeros")
        preprocessed_data = np.nan_to_num(preprocessed_data, nan=0.0)
    else:
        print("Successfully replaced all NaN values with track-specific averages")
    
    return preprocessed_data

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for gait recognition')
    parser.add_argument('--data', type=str, default='results2/industrial_gait_features_flat.npy', 
                        help='Path to numpy data file')
    parser.add_argument('--id_map', type=str, default='results2/id_to_name.json', 
                        help='Path to ID to name mapping file')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--stride', type=int, default=15, help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'cnnlstm'], 
                        help='Model type')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='./models/saved', 
                        help='Directory to save model and results')
    parser.add_argument('--preprocess', action='store_true', default=True,
                        help='Preprocess by replacing NaN values with track-specific averages')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess data if requested
    if args.preprocess:
        print(f"Preprocessing data from {args.data}")
        preprocessed_data = preprocess_nan_values(args.data)
        # Save preprocessed data to a temporary file
        preproc_file = args.data.replace('.npy', '_preprocessed.npy')
        np.save(preproc_file, preprocessed_data)
        print(f"Saved preprocessed data to {preproc_file}")
        # Use preprocessed data for training
        data_file = preproc_file
    else:
        data_file = args.data
    
    # Prepare data
    print(f"Loading and preparing data from {data_file} with sequence length {args.seq_length}")
    sequences, labels, id_to_name, id_to_label = prepare_data_lstm(
        npy_file=data_file,
        id_to_name_file=args.id_map,
        seq_length=args.seq_length,
        stride=args.stride
    )
    
    # Print dataset info
    print(f"Total sequences: {len(sequences)}")
    print(f"Number of classes: {len(id_to_name)}")
    print(f"Classes: {id_to_name}")
    print(f"Sequence shape: {sequences[0].shape}")
    
    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Reverse mapping: label index -> person_id
    label_to_id = {v: k for k, v in id_to_label.items()}
    for label, count in zip(unique_labels, counts):
        person_id = label_to_id.get(label, "Unknown")
        name = id_to_name.get(str(person_id), "Unknown")
        print(f"Class {label} ({person_id}, {name}): {count} samples")
    # Check for invalid track IDs in sequences
    print("Checking for invalid track IDs in sequences...")
    valid_track_ids = set(id_to_label.keys())
    invalid_sequences = []

    for i, seq in enumerate(sequences):
        track_id = int(seq[0][0])
        if track_id not in valid_track_ids:
            invalid_sequences.append(i)
            print(f"Found invalid track_id: {track_id} in sequence {i}")

    # Option 1: Filter out sequences with invalid track IDs
    if invalid_sequences:
        print(f"Filtering out {len(invalid_sequences)} sequences with invalid track IDs")
        valid_indices = [i for i in range(len(sequences)) if i not in invalid_sequences]
        sequences = [sequences[i] for i in valid_indices]
        
    # Generate labels only for valid sequences
    labels = [id_to_label[int(seq[0][0])] for seq in sequences]
    labels = np.array(labels)
    
    sequences = np.array(sequences)
    # Identify which classes remain after filtering
    unique_remaining_labels = np.unique(labels)
    print(f"Final dataset: {len(sequences)} sequences with {len(unique_remaining_labels)} unique classes")
    print(f"Sequences shape: {sequences.shape}")
    
    # Create filtered versions of id_to_name and id_to_label that only include classes present in the data
    filtered_id_to_name = {}
    filtered_id_to_label = {}
    label_to_id = {v: k for k, v in id_to_label.items()}  # Reverse mapping from label to ID
    
    # Create a new label mapping that's consecutive (0, 1, 2, ...)
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_remaining_labels)}
    
    # Update the labels array with the new mapping
    labels = np.array([new_label_mapping[label] for label in labels])
    
    # Create the filtered dictionaries
    for old_label in unique_remaining_labels:
        person_id = label_to_id[old_label]
        new_label = new_label_mapping[old_label]
        name = id_to_name.get(str(person_id), f"Person_{person_id}")
        
        filtered_id_to_name[str(person_id)] = name
        filtered_id_to_label[person_id] = new_label
    
    print(f"Filtered ID to name mapping: {filtered_id_to_name}")
    print(f"Filtered ID to label mapping: {filtered_id_to_label}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        sequences=sequences,
        labels=labels,
        batch_size=args.batch_size
    )
    
    print(f"Train set: {len(train_loader.dataset)} samples")
    print(f"Validation set: {len(val_loader.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    print(f"Class weights: {class_weights}")
    
    # Create model with the CORRECT number of classes (not the original number)
    input_size = sequences.shape[2]  # Feature dimension
    num_classes = len(unique_remaining_labels)  # Use the actual number of classes in filtered data
    
    if args.model == 'lstm':
        model = GaitLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes,  # Using correct number of classes
            dropout=args.dropout
        )
        print(f"Created LSTM model with input_size={input_size}, hidden_size={args.hidden_size}, "
              f"num_layers={args.num_layers}, num_classes={num_classes}")
    else:  # cnnlstm
        model = CNNLSTMGait(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes,  # Using correct number of classes
            dropout=args.dropout
        )
        print(f"Created CNN-LSTM model with input_size={input_size}, hidden_size={args.hidden_size}, "
              f"num_layers={args.num_layers}, num_classes={num_classes}")
    
    # Create output directory
    model_save_dir = os.path.join(args.save_dir, args.model, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=model_save_dir,
        id_to_name=filtered_id_to_name,  # Use filtered dictionaries
        id_to_label=filtered_id_to_label  # Use filtered dictionaries
    )
    
    print(f"Training complete! Best model saved to {model_save_dir}")
    print(f"ID to name mapping: {filtered_id_to_name}")
    print(f"ID to label mapping: {filtered_id_to_label}")

if __name__ == "__main__":
    main()