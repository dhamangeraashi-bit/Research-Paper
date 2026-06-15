import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. DEFINE COLUMNS (Your existing configuration)
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

print("📦 Loading raw data...")
df = pd.read_csv("data/KDDTrain+.txt", names=columns, header=None)

# 2. DATA CLEANING (Baby Steps)
# Neural networks hate text like 'tcp' or 'ftp_data'. Let's convert them to quick numbers.
df['protocol_type'] = df['protocol_type'].astype('category').cat.codes
df['service'] = df['service'].astype('category').cat.codes
df['flag'] = df['flag'].astype('category').cat.codes

# Map target label: 'normal' becomes 0, any attack type becomes 1
df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop text columns we don't need for math vectors anymore
df_numeric = df.drop(columns=['attack_type', 'difficulty'])

# 3. FUSING THE BIOMETRIC MODALITY (Simulation Layer)
print("🧬 Simulating synced biometric stream (Keystroke Dynamics)...")
num_rows = len(df_numeric)

# To mimic real keystroke dynamics, we generate 4 classic features:
# Hold times and flight times between keypresses (measured in milliseconds)
np.random.seed(42)
keystroke_hold_time = np.random.normal(loc=0.11, scale=0.02, size=num_rows)  # Normal user speed
keystroke_flight_time = np.random.normal(loc=0.18, scale=0.04, size=num_rows)

# If the network row is an attack (label == 1), let's spike the keystroke metrics 
# to simulate an attacker's erratic, uncharacteristic typing signature!
keystroke_hold_time = np.where(df_numeric['label'] == 1, keystroke_hold_time * 1.6, keystroke_hold_time)
keystroke_flight_time = np.where(df_numeric['label'] == 1, keystroke_flight_time * 1.8, keystroke_flight_time)

# Add them to our dataframe
df_numeric['key_hold'] = keystroke_hold_time
df_numeric['key_flight'] = keystroke_flight_time

print(f"✅ Multi-modal dataframe created successfully! New Shape: {df_numeric.shape}")

# =====================================================================
# 4. PREPARING DATA FOR THE PYTORCH CONV-LSTM
# =====================================================================
# LSTMs require sequences (windows of time), not just individual rows.
class BiosecurityDataset(Dataset):
    def __init__(self, dataframe, sequence_length=10):
        self.labels = dataframe['label'].values
        # Drop the label to keep only input features
        self.features = dataframe.drop(columns=['label']).values.astype(np.float32)
        self.seq_len = sequence_length

    def __len__(self):
        # We lose the first few elements to form a complete sliding window
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Extract a window of consecutive logs (e.g., 10 packets in a row)
        x_seq = self.features[idx : idx + self.seq_len]
        y_label = self.labels[idx + self.seq_len]
        return torch.tensor(x_seq), torch.tensor(y_label, dtype=torch.long)

# Initialize PyTorch components
target_dataset = BiosecurityDataset(df_numeric, sequence_length=10)
train_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# Test execution inspection
samples, targets = next(iter(train_loader))
print("\n--- Tensor Shapes Passed to Conv-LSTM ---")
print(f"Batch Tensor Shape [Batch Size, Window Length, Features]: {samples.shape}")
print(f"Labels Tensor Shape: {targets.shape}")