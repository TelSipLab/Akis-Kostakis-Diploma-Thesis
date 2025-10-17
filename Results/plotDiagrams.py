import matplotlib.pyplot as plt
import numpy as np

# Define file names
rollPredicted = 'Results/predicted_roll_complementary.txt'
rollReal = 'Results/expected_roll.txt'

rmse = 0.70 # either by cpp code or we can calculate from python

# Read data from files
rollData = np.loadtxt(rollPredicted)
rollReal = np.loadtxt(rollReal)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(rollData, label='Predicted Roll', linestyle='-', linewidth=1, markersize=4)
plt.plot(rollReal, label='Expected Roll', linestyle='--', linewidth=1, markersize=4)

# Add labels, title, and grid
plt.xlabel('Time')
plt.ylabel('Degrees')
plt.title("Actual roll vs the complementary filter calculation")
plt.legend()
plt.grid(True)

# Display RMSE on the plot
plt.text(0.05, 0.95, f'RMSE: {rmse}',
         transform=plt.gca().transAxes,  # use axes coordinates
         fontsize=12, color='black', verticalalignment='top')

# Show the plot
# plt.show()

plt.savefig("Results/Figures/RollRMSEComplementary.png")
