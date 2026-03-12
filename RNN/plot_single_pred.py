import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) == 2:
    sample_index = int(sys.argv[1])
else:
    print("Usage: python3 plot_single_pred.py <sample_index>")
    print("  sample_index: index into the test set (0 = first test sample)")
    sys.exit(1)

# Read the predictions CSV
df = pd.read_csv('Results/lstm_predictions.csv')

# Filter to test set only
df = df[df['set'] == 'test'].reset_index(drop=True)

# Auto-detect number of steps per sample from the data
num_steps = df['step_ahead'].max()
print(f"Test set: {len(df)} rows ({len(df) // num_steps} samples, {num_steps} steps each)")

# Convert radians to degrees
for col in ['roll_pred', 'roll_gt', 'pitch_pred', 'pitch_gt', 'yaw_pred', 'yaw_gt']:
    df[col + '_deg'] = np.rad2deg(df[col])

# Each sample has num_steps rows
start_pos = sample_index * num_steps
if start_pos + num_steps > len(df):
    print(f"Sample index too large! Max: {len(df) // num_steps - 1}")
    sys.exit(1)

df_selection = df.iloc[start_pos:start_pos + num_steps].copy()
x_axis = df_selection['step_ahead'].values

# Compute RMSE for this sample
roll_rmse = np.sqrt(np.mean((df_selection['roll_pred_deg'] - df_selection['roll_gt_deg'])**2))
pitch_rmse = np.sqrt(np.mean((df_selection['pitch_pred_deg'] - df_selection['pitch_gt_deg'])**2))
yaw_rmse = np.sqrt(np.mean((df_selection['yaw_pred_deg'] - df_selection['yaw_gt_deg'])**2))

print(f"Sample {sample_index} | Roll RMSE: {roll_rmse:.3f} | Pitch RMSE: {pitch_rmse:.3f} | Yaw RMSE: {yaw_rmse:.3f}")

# Create 3-row subplot (matching EKF style)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax in axes:
    ax.tick_params(axis='both', labelsize=14)

angles = [
    ('Roll',  'roll_pred_deg',  'roll_gt_deg',  roll_rmse),
    ('Pitch', 'pitch_pred_deg', 'pitch_gt_deg', pitch_rmse),
    ('Yaw',   'yaw_pred_deg',   'yaw_gt_deg',   yaw_rmse),
]

for i, (name, pred_col, gt_col, rmse) in enumerate(angles):
    axes[i].plot(x_axis, df_selection[pred_col].values,
                 label='LSTM Prediction', linestyle='-', linewidth=2, marker='x',
                 markersize=8, color='#ff7f0e')
    axes[i].plot(x_axis, df_selection[gt_col].values,
                 label='Ground Truth', linestyle='--', linewidth=2, marker='o',
                 markersize=6, color='#1f77b4')
    axes[i].set_ylabel(f'{name} Angle (degrees)', fontsize=14)
    axes[i].legend(loc='upper right', fontsize=14)
    axes[i].grid(True, alpha=0.3)

    # RMSE text box
    axes[i].text(0.02, 0.95, f'{name} RMSE: {rmse:.3f} deg',
                 transform=axes[i].transAxes, fontsize=13,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_title(f'LSTM {num_steps}-Step Ahead Prediction - Test Sample {sample_index}', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Step Ahead', fontsize=16)
axes[2].set_xticks(range(1, num_steps + 1))

plt.tight_layout()

filename = f'Results/LSTM/single_sample_prediction_N{num_steps}_test_{sample_index}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {filename}")

plt.show()
