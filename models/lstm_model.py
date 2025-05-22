import torch
import torch.nn as nn

class GaitLSTM(nn.Module):
    """
    Advanced LSTM architecture for gait recognition with:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Layer normalization
    - Dropout regularization
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=7, dropout=0.3):
        super(GaitLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent exploding/vanishing gradients"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_len, features = x.size()
        
        # Apply input normalization (reshape required)
        x_reshaped = x.reshape(-1, features)
        x_normalized = self.input_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, features)
        
        # Forward propagate LSTM with gradient clipping
        with torch.autograd.set_detect_anomaly(True):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            
            # Apply LSTM
            lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_length, hidden_size*2)
            
            # Apply layer normalization
            lstm_out = self.layer_norm(lstm_out)
            
            # Attention mechanism
            attention_weights = self.attention(lstm_out)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            
            # Dense layers
            out = self.fc1(context_vector)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
        return out
    
# Alternative model variant with CNN feature extraction
class CNNLSTMGait(nn.Module):
    """
    Hybrid CNN-LSTM model for gait recognition:
    - CNN layers extract spatial features from each time step
    - LSTM processes the sequential information
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=7, dropout=0.3):
        super(CNNLSTMGait, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_len, features = x.size()
        
        # Reshape for CNN (combine batch and seq_len dimensions)
        x_reshaped = x.view(batch_size * seq_len, features, 1)
        x_reshaped = x_reshaped.permute(0, 2, 1)  # Change to (batch*seq_len, 1, features)
        
        # Apply CNN feature extraction
        cnn_out = self.conv_layers(x_reshaped)
        cnn_out = cnn_out.mean(dim=2)  # Global average pooling
        
        # Reshape back to sequence format
        lstm_in = cnn_out.view(batch_size, seq_len, -1)
        
        # Apply LSTM
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout and final classification
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        
        return output