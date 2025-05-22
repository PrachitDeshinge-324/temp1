"""
Configuration settings for gait recognition models
"""

# Data configuration
DATA_CONFIG = {
    'npy_file': 'results/industrial_gait_features_flat.npy',
    'id_to_name_file': 'results/id_to_name.json',
    'seq_length': 30,
    'stride': 15,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_state': 42,
}

# LSTM model configuration
LSTM_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
}

# CNN-LSTM model configuration
CNN_LSTM_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'cnn_channels': [64, 128, 256],
    'kernel_sizes': [3, 3, 3],
}

# Output directory
OUTPUT_DIR = 'models/saved'