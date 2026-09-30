"""
Quick diagnostic: check what the model actually outputs vs what the targets look like.
Run this on Colab after training to diagnose the MSE issue.
"""
import numpy as np
import glob
import os

# Find any saved predictions
result_dirs = sorted(glob.glob('results/*alibaba*'))
print(f"Found {len(result_dirs)} result directories")

for d in result_dirs:
    pred_path = os.path.join(d, 'pred.npy')
    true_path = os.path.join(d, 'true.npy')
    metrics_path = os.path.join(d, 'metrics.npy')
    
    if os.path.exists(pred_path) and os.path.exists(true_path):
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        print(f"\n{'='*60}")
        print(f"Directory: {d}")
        print(f"{'='*60}")
        print(f"Pred shape: {pred.shape}")
        print(f"True shape: {true.shape}")
        print()
        print(f"Predictions range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"Predictions mean:  {pred.mean():.4f}")
        print(f"Predictions std:   {pred.std():.4f}")
        print()
        print(f"Targets range:     [{true.min():.4f}, {true.max():.4f}]")
        print(f"Targets mean:      {true.mean():.4f}")
        print(f"Targets std:       {true.std():.4f}")
        print()
        
        # Check if they're in different scales
        pred_scale = max(abs(pred.mean()), pred.std())
        true_scale = max(abs(true.mean()), true.std())
        print(f"Scale ratio (pred/true): {pred_scale/true_scale:.2f}x")
        
        if pred_scale / true_scale > 10:
            print(">>> SCALE MISMATCH DETECTED! Model outputs and targets are in different scales.")
        elif true_scale / pred_scale > 10:
            print(">>> SCALE MISMATCH DETECTED! Targets are much larger than predictions.")
        else:
            print(">>> Scales look comparable.")
        
        # Compute MSE manually
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        print(f"\nManual MSE: {mse:.4f}")
        print(f"Manual MAE: {mae:.4f}")
        
        # First sample comparison
        print(f"\nFirst sample, first 5 channels, first 5 timesteps:")
        print(f"Pred: {pred[0, :5, :5]}")
        print(f"True: {true[0, :5, :5]}")
