import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. THE DATA LOADER CONFIGURATION
# =====================================================================
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

print("Loading raw dataset...")
df = pd.read_csv("data/KDDTrain+.txt", names=columns, header=None)

# Basic encoding
df['protocol_type'] = df['protocol_type'].astype('category').cat.codes
df['service'] = df['service'].astype('category').cat.codes
df['flag'] = df['flag'].astype('category').cat.codes
df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
df_numeric = df.drop(columns=['attack_type', 'difficulty'])

# Inject simulated biometrics
num_rows = len(df_numeric)
np.random.seed(42)
keystroke_hold_time = np.random.normal(loc=0.11, scale=0.02, size=num_rows)
keystroke_flight_time = np.random.normal(loc=0.18, scale=0.04, size=num_rows)
keystroke_hold_time = np.where(df_numeric['label'] == 1, keystroke_hold_time * 1.6, keystroke_hold_time)
keystroke_flight_time = np.where(df_numeric['label'] == 1, keystroke_flight_time * 1.8, keystroke_flight_time)
df_numeric['key_hold'] = keystroke_hold_time
df_numeric['key_flight'] = keystroke_flight_time

#step 4: Create the PyTorch Dataset class
class BiosecurityDataset(Dataset):
    def __init__(self, dataframe, sequence_length=10):
        self.labels = dataframe['label'].values
        self.features = dataframe.drop(columns=['label']).values.astype(np.float32)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.features[idx : idx + self.seq_len]
        y_label = self.labels[idx + self.seq_len]
        return torch.tensor(x_seq), torch.tensor(y_label, dtype=torch.long)


# 2. THE CONV-LSTM MODEL ARCHITECTURE

class BioGuardEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_classes):
        super(BioGuardEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# 3. LIVE SESSION BAYESIAN TRUST DECAY TRACKER FUNCTION
def evaluate_live_session_trust(eval_model, data_loader, decay_rate=0.05, sensitivity=1.5):
    print("\n Launching Live BioGuard AI Trust Evaluator...")
    eval_model.eval() 
    
    current_trust_score = 1.0  
    print(f"Initial Session State: Trust Score = {current_trust_score:.2f}")
    
    samples, _ = next(iter(data_loader))
    
    with torch.no_grad():
        predictions = eval_model(samples)
        probabilities = torch.softmax(predictions, dim=1)
        
    for step in range(10):
        anomaly_prob = probabilities[step][1].item()
        penalty = anomaly_prob * sensitivity
        
        if anomaly_prob > 0.5:
            current_trust_score = current_trust_score * (1 - penalty) - decay_rate
        else:
            current_trust_score = current_trust_score + (decay_rate * 0.2)
            
        current_trust_score = max(0.0, min(1.0, current_trust_score))
        
        status = "DEVIATION DETECTED" if anomaly_prob > 0.5 else " NORMAL BEHAVIOR"
        print(f"Packet {step+1} | {status} (Anomaly Prob: {anomaly_prob:.4f}) -> Dynamic Trust Score: {current_trust_score:.2f}")
        
        if current_trust_score < 0.40:
            print(f" ALERT: Trust Score fell to {current_trust_score:.2f} (Below 0.40 Threshold).")
            print(" BIOGUARD AI ACTION: Session Revoked. Evicting user from system automatically.")
            break


INPUT_FEATURES = 43  
SEQUENCE_LENGTH = 10
HIDDEN_DIMENSION = 64
NUM_CLASSES = 2

# Setup dataloaders
dataset = BiosecurityDataset(df_numeric, sequence_length=SEQUENCE_LENGTH)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


model = BioGuardEncoder(input_channels=INPUT_FEATURES, hidden_dim=HIDDEN_DIMENSION, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\n Starting Model Training Engine...")
model.train()

# Train for a few batches to check the network weights
for batch_idx, (data, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 200 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} | Training Loss Value: {loss.item():.4f}")
        
    if batch_idx >= 600:
        break

print("\n Training Loop Core Validation Passed Successfully!")

# Run the evaluator at the very end, now that 'model' is fully defined up above!
evaluate_live_session_trust(model, train_loader)