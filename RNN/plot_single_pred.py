import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) == 2:
    timestamp_draw = int(sys.argv[1])
else:
    print(f"Wrong usage")
    sys.exit(1)

df = pd.read_csv('Results/lstm_predictions.csv')

# Convert radians to degrees
df['roll_pred_deg'] = np.rad2deg(df['roll_pred'])
df['roll_gt_deg'] = np.rad2deg(df['roll_gt'])
df['pitch_pred_deg'] = np.rad2deg(df['pitch_pred'])
df['pitch_gt_deg'] = np.rad2deg(df['pitch_gt'])
df['yaw_pred_deg'] = np.rad2deg(df['yaw_pred'])
df['yaw_gt_deg'] = np.rad2deg(df['yaw_gt'])


print(f"Total rows in dataset: {len(df)}")


# Find the index of the first row matching both conditions
matching_index = df[(df['timestep'] == timestamp_draw) & (df['step_ahead'] == 1)].index

if len(matching_index) == 0:
    print("No row found with the given timestep and step_ahead == 1")
    sys.exit(1)

start_idx = matching_index[0]  # first matching row
df_selection = df.iloc[start_idx:start_idx + 10]

print(df_selection)



# Plot roll, pitch, yaw predictions vs ground truth
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Roll
axes[0].plot(df_selection['roll_gt_deg'], label='Roll GT', marker='o')
axes[0].plot(df_selection['roll_pred_deg'], label='Roll Pred', marker='x')
axes[0].set_ylabel('Roll (deg)')
axes[0].legend()
axes[0].grid(True)

# Pitch
axes[1].plot(df_selection['pitch_gt_deg'], label='Pitch GT', marker='o')
axes[1].plot(df_selection['pitch_pred_deg'], label='Pitch Pred', marker='x')
axes[1].set_ylabel('Pitch (deg)')
axes[1].legend()
axes[1].grid(True)

# Yaw
axes[2].plot(df_selection['yaw_gt_deg'], label='Yaw GT', marker='o')
axes[2].plot(df_selection['yaw_pred_deg'], label='Yaw Pred', marker='x')
axes[2].set_ylabel('Yaw (deg)')
axes[2].set_xlabel('Row Index')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()