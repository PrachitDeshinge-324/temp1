import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(GraphConvolution, self).__init__()
        self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, num_nodes, num_frames)
        A = self.A.to(x.device)
        x = torch.einsum('nctv,vw->nctw', (x, A))
        x = self.conv(x)
        x = self.bn(x)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix, temporal_kernel_size=9):
        super(STGCNBlock, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, adjacency_matrix)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel_size, 1), padding=((temporal_kernel_size-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return F.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_joints, in_channels, adjacency_matrix, num_classes):
        super(STGCN, self).__init__()
        
        # Define your skeleton adjacency matrix
        self.A = adjacency_matrix
        
        # ST-GCN layers
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, self.A),
            STGCNBlock(64, 64, self.A),
            STGCNBlock(64, 64, self.A),
            STGCNBlock(64, 128, self.A),
            STGCNBlock(128, 128, self.A),
            STGCNBlock(128, 256, self.A),
        ])
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, channels, num_joints, sequence_length)
        for st_gcn in self.st_gcn_networks:
            x = st_gcn(x)
            
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x

# Usage example
def create_stgcn_model(num_people):
    # Define adjacency matrix based on human skeleton connections
    # This is a simplified example - you'll need to adjust based on your keypoint format
    num_joints = 17  # COCO format has 17 joints
    adjacency_matrix = np.eye(num_joints)  # Self-connections
    # Add connections between joints (e.g., right shoulder to right elbow)
    # adjacency_matrix[shoulder_idx, elbow_idx] = 1
    
    model = STGCN(
        num_joints=num_joints, 
        in_channels=2,  # x,y coordinates for each joint
        adjacency_matrix=adjacency_matrix,
        num_classes=num_people
    )
    return model