import matplotlib.pyplot as plt
import numpy as np

# Roll ground truth file
rollReal = 'Results/ExpectedResults/expected_roll.txt'
rollPredicted = 'Results/Results/MahonyRoll_kp_9.txt'

rollRMSE = 0.819  # Keep it to 3 decimals

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
plt.xlabel('Time (samples)', fontsize=14)
plt.ylabel('Degrees', fontsize=14)
plt.title(f"Zoomed View: Actual roll vs Mahony Filter Estimation (samples {start_time}-{end_time})", fontsize=14)
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

plt.text(0.05, 0.88, 'kp = 9',
         transform=plt.gca().transAxes,
         fontsize=12, color='black', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Reduce whitespace margins
plt.tight_layout()

# Save the plot
plt.savefig(f"Results/Figures/MahonyRoll_kp_9_zoomed_{start_time}_{end_time}.png", dpi=150)
print(f"Zoomed plot saved: MahonyRoll_kp_9_zoomed_{start_time}_{end_time}.png")
