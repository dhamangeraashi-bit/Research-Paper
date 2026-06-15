import torch
import numpy as np
from torch.utils.data import DataLoader
from main import model, train_loader, evaluate_live_session_trust, BiosecurityDataset, df_numeric

print(" Initializing Targeted Session Hijacking Attack Simulation...")

# 1. Take a small block of real traffic data
dataset = BiosecurityDataset(df_numeric, sequence_length=10)
malicious_loader = DataLoader(dataset, batch_size=10, shuffle=False)
samples, targets = next(iter(malicious_loader))

# 2. Inject massive malicious data to confuse the model
# We multiply the inputs heavily to simulate highly erratic typing and wild network bursts
attack_samples = samples.clone()
attack_samples[:, :, :] = attack_samples[:, :, :] * 5.0

# 3. Create a fake loader containing our custom attack packets
class AttackDataWrapper:
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        yield self.data, torch.zeros(10)

attack_stream = AttackDataWrapper(attack_samples)

print(" Injection complete. Passing malicious packets to ABT Evaluator...")

# Force the evaluator to run across multiple test loops so the decay stacks up!
for i in range(5):
    print(f"\n--- Injected Attack Wave {i+1} ---")
    evaluate_live_session_trust(model, attack_stream, decay_rate=0.18, sensitivity=2.5)