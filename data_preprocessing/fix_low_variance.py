"""
Fix alibaba.csv: remove low-variance machines that cause StandardScaler explosion.

Problem: 4 machines (m_51, m_59, m_67, m_87) have near-zero variance in the
training set. When StandardScaler normalizes, test values explode to 18,000+,
destroying MSE (127,000 vs paper's 0.003).

Fix: Drop these 4 machines, keep 96 good machines, update enc_in accordingly.
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dataset/alibaba.csv')
n_train = int(len(df) * 0.7)

# Get feature columns (all except 'date')
feature_cols = [c for c in df.columns if c != 'date']
train_data = df[feature_cols].iloc[:n_train]

# Find low-variance machines (std < 1.0 in training set)
stds = train_data.std()
low_var_cols = stds[stds < 1.0].index.tolist()

print(f"Removing {len(low_var_cols)} low-variance machines: {low_var_cols}")
for col in low_var_cols:
    print(f"  {col}: training std = {stds[col]:.6f}")

# Drop them
df = df.drop(columns=low_var_cols)

# Rename remaining columns to m_0, m_1, ..., m_N, OT
feature_cols = [c for c in df.columns if c != 'date']
n_features = len(feature_cols)

# Last column becomes OT (target)
new_names = {col: f"m_{i}" for i, col in enumerate(feature_cols[:-1])}
new_names[feature_cols[-1]] = "OT"
df = df.rename(columns=new_names)

# Verify
print(f"\nCleaned dataset:")
print(f"  Shape: {df.shape}")
print(f"  Features: {n_features} (date excluded)")
print(f"  Columns: {list(df.columns[:4])} ... {list(df.columns[-3:])}")

# Verify no more low-variance issues
train_clean = df.drop(columns=['date']).iloc[:n_train]
min_std = train_clean.std().min()
print(f"  Min training std: {min_std:.4f} (should be >> 1.0)")

# Save
df.to_csv('dataset/alibaba.csv', index=False, float_format='%.4f')
print(f"\nSaved to dataset/alibaba.csv")
print(f"\n*** UPDATE TRAINING COMMAND: use --enc_in {n_features} --dec_in {n_features} --c_out {n_features} ***")
