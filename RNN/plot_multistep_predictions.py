import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# =======================
# Parse command-line arguments
# =======================
start_timestep = 0
end_timestep = 50

if len(sys.argv) == 3:
    start_timestep = int(sys.argv[1])
    end_timestep = int(sys.argv[2])
elif len(sys.argv) != 1:
    print("Usage: python plot_multistep_predictions.py [start_timestep] [end_timestep]")
    sys.exit(1)

print(f"Plotting multi-step predictions from timestep {start_timestep} to {end_timestep}")

# =======================
# Load CSV
# =======================
df = pd.read_csv("Results/lstm_predictions.csv")

# Sanity check
required_cols = {
    "timestep", "step_ahead",
    "roll_pred", "pitch_pred", "yaw_pred",
    "roll_gt", "pitch_gt", "yaw_gt"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

print(f"Total rows in dataset: {len(df)}")
print(f"Timestep range in data: {df['timestep'].min()} to {df['timestep'].max()}")

# =======================
# Convert radians to degrees
# =======================
df["roll_pred_deg"]  = np.rad2deg(df["roll_pred"])
df["roll_gt_deg"]    = np.rad2deg(df["roll_gt"])
df["pitch_pred_deg"] = np.rad2deg(df["pitch_pred"])
df["pitch_gt_deg"]   = np.rad2deg(df["pitch_gt"])

# Unwrap yaw before converting to degrees
df["yaw_gt_deg"]   = np.rad2deg(np.unwrap(df["yaw_gt"]))
df["yaw_pred_deg"] = np.rad2deg(np.unwrap(df["yaw_pred"]))

# =======================
# Filter timestep range
# =======================
df_filtered = df[
    (df["timestep"] >= start_timestep) &
    (df["timestep"] <= end_timestep)
]

print(f"Filtered rows: {len(df_filtered)}")

# =======================
# Ground truth (one value per timestep)
# =======================
gt_data = (
    df_filtered
    .sort_values("step_ahead")
    .groupby("timestep")
    .first()
    .reset_index()
)

# =======================
# Horizon-specific predictions
# =======================
step1  = df_filtered[df_filtered["step_ahead"] == 1]
step5  = df_filtered[df_filtered["step_ahead"] == 5]
step10 = df_filtered[df_filtered["step_ahead"] == 10]

# =======================
# Create combined plot with all 3 angles
# =======================
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Roll subplot
ax = axes[0]
ax.plot(gt_data["timestep"], gt_data["roll_gt_deg"],
        label="Ground Truth", color="blue", linewidth=2.5)
ax.plot(step1["timestep"], step1["roll_pred_deg"],
        "--", label="1-step ahead", alpha=0.7)
ax.plot(step5["timestep"], step5["roll_pred_deg"],
        "--", label="5-step ahead", alpha=0.6)
ax.plot(step10["timestep"], step10["roll_pred_deg"],
        "--", label="10-step ahead", alpha=0.5)
ax.set_ylabel("Roll Angle (degrees)", fontsize=12)
ax.set_title(f"Roll Angle: Multi-Step Predictions ({start_timestep}–{end_timestep})",
             fontweight="bold", fontsize=13)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Pitch subplot
ax = axes[1]
ax.plot(gt_data["timestep"], gt_data["pitch_gt_deg"],
        label="Ground Truth", color="blue", linewidth=2.5)
ax.plot(step1["timestep"], step1["pitch_pred_deg"],
        "--", label="1-step ahead", alpha=0.7)
ax.plot(step5["timestep"], step5["pitch_pred_deg"],
        "--", label="5-step ahead", alpha=0.6)
ax.plot(step10["timestep"], step10["pitch_pred_deg"],
        "--", label="10-step ahead", alpha=0.5)
ax.set_ylabel("Pitch Angle (degrees)", fontsize=12)
ax.set_title(f"Pitch Angle: Multi-Step Predictions ({start_timestep}–{end_timestep})",
             fontweight="bold", fontsize=13)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Yaw subplot
ax = axes[2]
ax.plot(gt_data["timestep"], gt_data["yaw_gt_deg"],
        label="Ground Truth", color="blue", linewidth=2.5)
ax.plot(step1["timestep"], step1["yaw_pred_deg"],
        "--", label="1-step ahead", alpha=0.7)
ax.plot(step5["timestep"], step5["yaw_pred_deg"],
        "--", label="5-step ahead", alpha=0.6)
ax.plot(step10["timestep"], step10["yaw_pred_deg"],
        "--", label="10-step ahead", alpha=0.5)
ax.set_xlabel("Timestep", fontsize=12)
ax.set_ylabel("Yaw Angle (degrees)", fontsize=12)
ax.set_title(f"Yaw Angle: Multi-Step Predictions ({start_timestep}–{end_timestep})",
             fontweight="bold", fontsize=13)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f"Results/LSTM/multistep_predictions_{start_timestep}_{end_timestep}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved combined plot to {filename}")
plt.show()

yaw_bias = (step1["yaw_pred_deg"] - step1["yaw_gt_deg"]).mean()
print(f"Mean yaw bias (deg): {yaw_bias:.3f}")

print("Combined plot generated successfully.")
