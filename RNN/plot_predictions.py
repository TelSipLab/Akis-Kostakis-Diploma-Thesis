import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Parse command-line arguments for timestep range
start_timestep = 0
end_timestep = 100

if len(sys.argv) >= 3:
    start_timestep = int(sys.argv[1])
    end_timestep = int(sys.argv[2])
elif len(sys.argv) == 2:
    print("Usage: python plot_predictions.py [start_timestep] [end_timestep]")
    print("Example: python plot_predictions.py 500 700")
    print("Using default: 0 to 100")
    sys.exit(1)

print(f"Plotting timesteps from {start_timestep} to {end_timestep}")

# Read the predictions CSV
df = pd.read_csv('Results/lstm_predictions.csv')

# Convert radians to degrees
df['roll_pred_deg'] = np.rad2deg(df['roll_pred'])
df['roll_gt_deg'] = np.rad2deg(df['roll_gt'])
df['pitch_pred_deg'] = np.rad2deg(df['pitch_pred'])
df['pitch_gt_deg'] = np.rad2deg(df['pitch_gt'])
df['yaw_pred_deg'] = np.rad2deg(df['yaw_pred'])
df['yaw_gt_deg'] = np.rad2deg(df['yaw_gt'])

print(f"Total rows in dataset: {len(df)}")
print(f"Timestep range: {df['timestep'].min()} to {df['timestep'].max()}")

# Since each sample predicts 10 timesteps ahead, we have overlapping predictions
# To avoid clutter, let's use only 1-step ahead predictions
# This means: for timestep T, use the prediction from sample that starts at T-1

# Group by timestep and take the first prediction (which is 1-step ahead)
df_one_step = df.groupby('timestep').first().reset_index()

print(f"\nAfter filtering to 1-step ahead predictions: {len(df_one_step)} rows")

# Filter to specified timestep range
df_filtered = df_one_step[(df_one_step['timestep'] >= start_timestep) &
                          (df_one_step['timestep'] <= end_timestep)]

print(f"Showing timesteps {start_timestep} to {end_timestep}: {len(df_filtered)} rows")

# ============= ROLL ANGLE PLOT =============
plt.figure(figsize=(14, 6))

plt.plot(df_filtered['timestep'], df_filtered['roll_gt_deg'],
         label='Ground Truth', color='blue', linewidth=2, alpha=0.9)
plt.plot(df_filtered['timestep'], df_filtered['roll_pred_deg'],
         label='LSTM Prediction (1-step ahead)', color='red', linewidth=1.5, alpha=0.7, linestyle='--')

plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Roll Angle (degrees)', fontsize=12)
plt.title(f'LSTM Roll Angle: 1-Step Ahead Prediction vs Ground Truth (Timesteps {start_timestep}-{end_timestep})',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

roll_filename = f'Results/roll_prediction_plot_{start_timestep}_{end_timestep}.png'
plt.savefig(roll_filename, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {roll_filename}")

# ============= PITCH ANGLE PLOT =============
plt.figure(figsize=(14, 6))

plt.plot(df_filtered['timestep'], df_filtered['pitch_gt_deg'],
         label='Ground Truth', color='blue', linewidth=2, alpha=0.9)
plt.plot(df_filtered['timestep'], df_filtered['pitch_pred_deg'],
         label='LSTM Prediction (1-step ahead)', color='red', linewidth=1.5, alpha=0.7, linestyle='--')

plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Pitch Angle (degrees)', fontsize=12)
plt.title(f'LSTM Pitch Angle: 1-Step Ahead Prediction vs Ground Truth (Timesteps {start_timestep}-{end_timestep})',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pitch_filename = f'Results/pitch_prediction_plot_{start_timestep}_{end_timestep}.png'
plt.savefig(pitch_filename, dpi=300, bbox_inches='tight')
print(f"Saved plot to {pitch_filename}")

# ============= YAW ANGLE PLOT =============
plt.figure(figsize=(14, 6))

plt.plot(df_filtered['timestep'], df_filtered['yaw_gt_deg'],
         label='Ground Truth', color='blue', linewidth=2, alpha=0.9)
plt.plot(df_filtered['timestep'], df_filtered['yaw_pred_deg'],
         label='LSTM Prediction (1-step ahead)', color='red', linewidth=1.5, alpha=0.7, linestyle='--')

plt.xlabel('Timestep', fontsize=12)
plt.ylabel('Yaw Angle (degrees)', fontsize=12)
plt.title(f'LSTM Yaw Angle: 1-Step Ahead Prediction vs Ground Truth (Timesteps {start_timestep}-{end_timestep})',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

yaw_filename = f'Results/yaw_prediction_plot_{start_timestep}_{end_timestep}.png'
plt.savefig(yaw_filename, dpi=300, bbox_inches='tight')
print(f"Saved plot to {yaw_filename}")

# Show all plots
plt.show()
