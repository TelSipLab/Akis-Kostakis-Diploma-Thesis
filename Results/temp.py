import matplotlib.pyplot as plt
import numpy as np

# Load all three datasets
rollComplementary = np.loadtxt('Results/predicted_roll_complementary.txt')
rollEKF = np.loadtxt('Results/predicted_roll_ekf.txt')
rollGroundTruth = np.loadtxt('Results/expected_roll.txt')

# Take a 45% sample from the middle of the dataset to capture interesting dynamics
total_samples = len(rollGroundTruth)
sample_size = int(total_samples * 0.45)
start_idx = int(total_samples * 0.275)  # Start at 27.5% to center the 45% sample
end_idx = start_idx + sample_size

# Extract the sample
rollComp_sample = rollComplementary[start_idx:end_idx]
rollEKF_sample = rollEKF[start_idx:end_idx]
rollTruth_sample = rollGroundTruth[start_idx:end_idx]
time_indices = np.arange(start_idx, end_idx)

# Create comparison plot
plt.figure(figsize=(14, 8))
plt.plot(time_indices, rollTruth_sample, label='Ground Truth', linestyle='-', linewidth=2, color='black', alpha=0.7)
plt.plot(time_indices, rollComp_sample, label='Complementary Filter', linestyle='--', linewidth=1.5, color='blue')
plt.plot(time_indices, rollEKF_sample, label='EKF', linestyle='-.', linewidth=1.5, color='red')

# Add labels, title, and grid
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Roll Angle (degrees)', fontsize=12)
plt.title(f'Roll Angle Comparison: Complementary Filter vs EKF (Sample: {sample_size} points, {int(sample_size/total_samples*100)}% of dataset)', fontsize=13)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)

# Tight y-axis limits to zoom in on differences (5% margin)
all_sample_data = np.concatenate([rollComp_sample, rollEKF_sample, rollTruth_sample])
y_min, y_max = all_sample_data.min(), all_sample_data.max()
y_margin = (y_max - y_min) * 0.05
plt.ylim(y_min - y_margin, y_max + y_margin)

plt.tight_layout()
plt.savefig("Results/Figures/Roll_Comparison_CF_vs_EKF.png", dpi=150)
print(f"Comparison plot saved: Roll_Comparison_CF_vs_EKF.png")
print(f"Sample range: indices {start_idx} to {end_idx} ({sample_size} samples)")
