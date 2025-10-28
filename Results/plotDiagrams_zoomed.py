import matplotlib.pyplot as plt
import numpy as np

# Roll ground truth file
rollReal = 'Results/ExpectedResults/expected_roll.txt'
rollPredicted = 'Results/Results/EkfRoll.txt'

rollRMSE = 0.298  # Keep it to 3 decimals

# Read data from files
rollData = np.loadtxt(rollPredicted)
rollReal = np.loadtxt(rollReal)

# Define zoom region (change these to zoom into different parts)
start_time = 300
end_time = 700

# Extract zoomed data
rollData_zoom = rollData[start_time:end_time]
rollReal_zoom = rollReal[start_time:end_time]
time_axis = np.arange(start_time, end_time)

# Create a plot with larger figure size
plt.figure(figsize=(12, 8))
plt.plot(time_axis, rollData_zoom, label='Estimated Roll', linestyle='-', linewidth=2, markersize=4, color='#ff7f0e')
plt.plot(time_axis, rollReal_zoom, label='Expected Roll', linestyle='--', linewidth=2, markersize=4, color='#1f77b4')

# Add labels, title, and grid
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Roll Angle (degrees)', fontsize=14)
plt.title(f"Zoomed View: Actual Roll vs EKF Estimation (samples {start_time}-{end_time})", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Dynamically set y-axis limits with some padding
all_data = np.concatenate([rollData_zoom, rollReal_zoom])
y_min, y_max = all_data.min(), all_data.max()
y_margin = (y_max - y_min) * 0.1  # 10% margin
plt.ylim(y_min - y_margin, y_max + y_margin)

# Display RMSE and alpha on the plot
plt.text(0.05, 0.95, f'Roll RMSE (full data): {rollRMSE}',
         transform=plt.gca().transAxes,
         fontsize=12, color='black', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Reduce whitespace margins
plt.tight_layout()

# Save the plot
plt.savefig(f"Results/Figures/EkfRoll_zoomed_{start_time}_{end_time}.png", dpi=150)
print(f"Zoomed EKF plot saved: Results/Figures/EkfRoll_zoomed_{start_time}_{end_time}.png")
