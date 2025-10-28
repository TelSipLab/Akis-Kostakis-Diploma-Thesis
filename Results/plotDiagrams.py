import matplotlib.pyplot as plt
import numpy as np

# Roll ground truth file
rollReal = 'Results/ExpectedResults/expected_roll.txt' # DONT CHANGE

# Define file names
rollPredicted = 'Results/Results/MahonyRoll_kp_9.txt'


rollRMSE = 0.588 # Keep it to 3 decimals

# Read data from files
rollData = np.loadtxt(rollPredicted)
rollReal = np.loadtxt(rollReal)

# Create a plot with larger figure size
plt.figure(figsize=(12, 8))
plt.plot(rollData, label='Estimated Roll', linestyle='-', linewidth=1, markersize=4)
plt.plot(rollReal, label='Expected Roll', linestyle='--', linewidth=1, markersize=4)

# Add labels, title, and grid
plt.xlabel('Time')
plt.ylabel('Degrees')
plt.title("Actual roll vs Mahony Filter Estimation")
plt.legend()
plt.grid(True)

# Dynamically set y-axis limits with some padding
all_data = np.concatenate([rollData, rollReal])
y_min, y_max = all_data.min(), all_data.max()
y_margin = (y_max - y_min) * 0.1  # 10% margin
plt.ylim(y_min - y_margin, y_max + y_margin)

# Display RMSE and alpha on the plot
plt.text(0.05, 0.95, f'Roll RMSE: {rollRMSE}',
         transform=plt.gca().transAxes,  # use axes coordinates
         fontsize=12, color='black', verticalalignment='top')

plt.text(0.05, 0.91, 'kp = 9',
         transform=plt.gca().transAxes,  # use axes coordinates
         fontsize=12, color='black', verticalalignment='top')

# Reduce whitespace margins
plt.tight_layout()

# Show the plot
# plt.show()

plt.savefig("Results/Figures/MahonyRoll_kp_9.png")
print(f"Plot saved")