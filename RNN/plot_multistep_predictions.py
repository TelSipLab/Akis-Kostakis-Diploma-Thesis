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
# Plot helper function
# =======================
def plot_angle(gt, s1, s5, s10, angle_name, ylabel, filename):
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(gt["timestep"], gt[f"{angle_name}_gt_deg"],
            label="Ground Truth", color="blue", linewidth=2.5)

    ax.plot(s1["timestep"], s1[f"{angle_name}_pred_deg"],
            "--", label="1-step ahead", alpha=0.7)

    ax.plot(s5["timestep"], s5[f"{angle_name}_pred_deg"],
            "--", label="5-step ahead", alpha=0.6)

    ax.plot(s10["timestep"], s10[f"{angle_name}_pred_deg"],
            "--", label="10-step ahead", alpha=0.5)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"LSTM {angle_name.capitalize()} Angle: Multi-Step Predictions "
        f"({start_timestep}–{end_timestep})",
        fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    # plt.show()
    plt.close()

# =======================
# Generate plots
# =======================
plot_angle(
    gt_data, step1, step5, step10,
    "roll", "Roll Angle (degrees)",
    f"Results/roll_multistep_plot_{start_timestep}_{end_timestep}.png"
)

plot_angle(
    gt_data, step1, step5, step10,
    "pitch", "Pitch Angle (degrees)",
    f"Results/pitch_multistep_plot_{start_timestep}_{end_timestep}.png"
)

plot_angle(
    gt_data, step1, step5, step10,
    "yaw", "Yaw Angle (degrees)",
    f"Results/yaw_multistep_plot_{start_timestep}_{end_timestep}.png"
)

yaw_bias = (step1["yaw_pred_deg"] - step1["yaw_gt_deg"]).mean()
print("Mean yaw bias (deg):", yaw_bias)

print("All plots generated successfully.")
