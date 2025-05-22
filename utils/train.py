import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd  # Add this line
import seaborn as sns
import os

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"Warning: NaN detected in model outputs at batch {batch_idx}")
            continue
            
        loss = criterion(outputs, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at batch {batch_idx}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Avoid division by zero
    if len(train_loader) > 0:
        train_loss = total_loss / len(train_loader)
    else:
        train_loss = 0
        
    train_acc = 100.0 * correct / max(total, 1)  # Avoid division by zero
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc

def test_model(model, test_loader, criterion, device, id_to_name, id_to_label, save_dir=None):
    """Test the model on the test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Find which classes are actually present in the test set
    unique_labels = np.unique(all_labels)
    
    # Create a mapping from original label indices to names
    label_to_id = {v: k for k, v in id_to_label.items()}
    
    # Create target_names that match the classes present in test data
    target_names = []
    for label in unique_labels:
        if label in label_to_id:
            person_id = label_to_id[label]
            name = id_to_name.get(str(person_id), f"Person_{person_id}")
            target_names.append(name)
        else:
            target_names.append(f"Unknown_{label}")
    
    print(f"Classes in test set: {unique_labels}")
    print(f"Target names for report: {target_names}")
    
    # Generate classification report with correct labels parameter
    report = classification_report(
        all_labels, all_preds, 
        labels=unique_labels,
        target_names=target_names,
        output_dict=True
    )
    
    # Save confusion matrix
    if save_dir:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save report as CSV
        report_data = []
        for class_name in target_names:
            if class_name in report:
                report_data.append([
                    class_name,
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
        
        import pandas as pd  # Make sure pandas is imported
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score'])
        report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'), index=False)
    
    return test_loss, test_acc, report

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'training_curves.png'))
    
    plt.show()