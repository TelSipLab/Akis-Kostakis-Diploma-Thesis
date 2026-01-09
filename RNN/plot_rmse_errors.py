import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the RMSE errors CSV (no header)
df = pd.read_csv('rmse_errors.csv', sep='\t', header=None, names=['epoch', 'rmse_rad'])

# Convert radians to degrees
df['rmse_deg'] = np.rad2deg(df['rmse_rad'])

print("RMSE Errors Data:")
print(df)
print(f"\nInitial RMSE (epoch 0): {df.iloc[0]['rmse_rad']:.6f} rad = {df.iloc[0]['rmse_deg']:.3f} deg")
print(f"Final MSE (epoch 1000): {df.iloc[-1]['rmse_rad']:.6f} rad = {df.iloc[-1]['rmse_deg']:.3f} deg")
print(f"Improvement: {((df.iloc[0]['rmse_deg'] - df.iloc[-1]['rmse_deg']) / df.iloc[0]['rmse_deg'] * 100):.1f}%")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: MSE in degrees (log scale)
ax1.plot(df['epoch'], df['rmse_deg'], marker='o', linewidth=2, markersize=5, color='blue')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MSE (degrees)', fontsize=12)
ax1.set_title('Training Loss vs Epoch (Degrees) - Log Scale', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: MSE in radians (log scale)
ax2.plot(df['epoch'], df['rmse_rad'], marker='o', linewidth=2, markersize=5, color='green')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MSE (radians)', fontsize=12)
ax2.set_title('Training Loss vs Epoch (Radians) - Log Scale', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save figure
filename = 'Results/LSTM/mse_training_loss.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {filename}")

plt.show()
