import matplotlib.pyplot as plt
import numpy as np

# Define file names
file1 = 'Results/predicted_roll.txt'
file2 = 'Results/expected_roll.txt'

rmse = 0.81 # either by cpp code or we can calculate from python

# Read data from files
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(data1, label='Predicted Roll', linestyle='-', linewidth=1, markersize=4)
plt.plot(data2, label='Expected Roll', linestyle='-', linewidth=1, markersize=4)

# Add labels, title, and grid
plt.xlabel('Time')
plt.ylabel('Degrees')
plt.title('Comparsion of actual roll with Complemnatary Filter roll')
plt.legend()
plt.grid(True)

# Display RMSE on the plot
plt.text(0.05, 0.95, f'RMSE: {rmse}',
         transform=plt.gca().transAxes,  # use axes coordinates
         fontsize=12, color='black', verticalalignment='top')

# Show the plot
plt.show()
