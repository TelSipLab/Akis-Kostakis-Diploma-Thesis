import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Parse command-line arguments for sample range
start_sample = 0
end_sample = 200

if len(sys.argv) >= 3:
    start_sample = int(sys.argv[1])
    end_sample = int(sys.argv[2])
elif len(sys.argv) == 2:
    print("Usage: python3 plot_predictions.py [start_sample] [end_sample]")
    print("Example: python3 plot_predictions.py 0 200")
    sys.exit(1)

# Read the predictions CSV
df = pd.read_csv('Results/lstm_predictions.csv')

# Filter to test set, 1-step ahead only
df = df[(df['set'] == 'test') & (df['step_ahead'] == 1)].reset_index(drop=True)
print(f"Test set, 1-step ahead: {len(df)} samples")

# Convert radians to degrees
for col in ['roll_pred', 'roll_gt', 'pitch_pred', 'pitch_gt', 'yaw_pred', 'yaw_gt']:
    df[col + '_deg'] = np.rad2deg(df[col])

# Filter to sample range
df_filtered = df.iloc[start_sample:end_sample]
x_axis = np.arange(start_sample, start_sample + len(df_filtered))

# Compute RMSE per angle for the plotted range
roll_rmse = np.sqrt(np.mean((df_filtered['roll_pred_deg'] - df_filtered['roll_gt_deg'])**2))
pitch_rmse = np.sqrt(np.mean((df_filtered['pitch_pred_deg'] - df_filtered['pitch_gt_deg'])**2))
yaw_rmse = np.sqrt(np.mean((df_filtered['yaw_pred_deg'] - df_filtered['yaw_gt_deg'])**2))

print(f"Plotting samples {start_sample} to {start_sample + len(df_filtered)}")
print(f"Roll RMSE: {roll_rmse:.3f} deg | Pitch RMSE: {pitch_rmse:.3f} deg | Yaw RMSE: {yaw_rmse:.3f} deg")

# Create 3-row subplot (matching EKF plot style)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax in axes:
    ax.tick_params(axis='both', labelsize=14)

angles = [
    ('Roll',  'roll_pred_deg',  'roll_gt_deg',  roll_rmse),
    ('Pitch', 'pitch_pred_deg', 'pitch_gt_deg', pitch_rmse),
    ('Yaw',   'yaw_pred_deg',   'yaw_gt_deg',   yaw_rmse),
]

for i, (name, pred_col, gt_col, rmse) in enumerate(angles):
    axes[i].plot(x_axis, df_filtered[pred_col].values,
                 label=f'LSTM Prediction', linestyle='-', linewidth=1.5, color='#ff7f0e')
    axes[i].plot(x_axis, df_filtered[gt_col].values,
                 label='Ground Truth', linestyle='--', linewidth=1.5, color='#1f77b4')
    axes[i].set_ylabel(f'{name} Angle (degrees)', fontsize=14)
    axes[i].legend(loc='upper right', fontsize=14)
    axes[i].grid(True, alpha=0.3)

    # RMSE text box
    axes[i].text(0.02, 0.95, f'{name} RMSE: {rmse:.3f} deg',
                 transform=axes[i].transAxes, fontsize=13,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_title('LSTM 1-Step Ahead Prediction vs Ground Truth (Test Set)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Test Sample', fontsize=16)

plt.tight_layout()

filename = f'Results/LSTM/lstm_predictions_test_{start_sample}_{end_sample}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {filename}")

plt.show()
