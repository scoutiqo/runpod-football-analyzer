import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TacticalGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(TacticalGNN, self).__init__()
        
        # 1. Graph Convolution Layers
        # These layers "pass messages" between players
        # (e.g. "I am here, my teammate is there, the ball is moving fast")
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)

        # 2. Classification Head
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1: Understand immediate neighbors (Pressure)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2: Understand local structure (Passing triangles)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 3: Understand global shape (Formation)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global Pooling: Aggregate all players into one "Team State" vector
        x = global_mean_pool(x, batch) 

        # Final Prediction (What is happening?)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
