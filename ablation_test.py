import torch
import torch.nn as nn
from main import train_loader  # Dropped X_train to prevent the import error

print("🧪 Starting my ablation study baseline test...")
print("Checking how a basic model performs without our Conv-LSTM layers.")

class BasicStudentModel(nn.Module):
    def __init__(self, input_features_count):
        super(BasicStudentModel, self).__init__()
        self.flat_layer = nn.Linear(input_features_count, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        last_packet_only = x[:, -1, :] 
        return self.activation(self.flat_layer(last_packet_only))

# 2. Automatically detect features from the first batch instead of using X_train
first_batch_samples, _ = next(iter(train_loader))
total_features = first_batch_samples.shape[2]

simple_model = BasicStudentModel(total_features)
loss_calculator = nn.BCELoss()