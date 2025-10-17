import matplotlib.pyplot as plt
import numpy as np

# Load datasets
rollComplementary = np.loadtxt('Results/predicted_roll_complementary.txt')
rollEKF = np.loadtxt('Results/predicted_roll_ekf.txt')
rollGroundTruth = np.loadtxt('Results/expected_roll.txt')

# Load sensor data
gyroData = np.loadtxt('Data/gyro.csv', delimiter=',')
accelData = np.loadtxt('Data/accel.csv', delimiter=',')

# Same sample window as before
total_samples = len(rollGroundTruth)
sample_size = int(total_samples * 0.45)
start_idx = int(total_samples * 0.275)
end_idx = start_idx + sample_size

# Extract samples
rollComp_sample = rollComplementary[start_idx:end_idx]
rollEKF_sample = rollEKF[start_idx:end_idx]
rollTruth_sample = rollGroundTruth[start_idx:end_idx]
time_indices = np.arange(start_idx, end_idx)

# Calculate errors
error_complementary = np.abs(rollComp_sample - rollTruth_sample)
error_ekf = np.abs(rollEKF_sample - rollTruth_sample)

# Extract gyro and accel for this window
gyro_sample = gyroData[start_idx:end_idx, :]
accel_sample = accelData[start_idx:end_idx, :]

# Calculate gyro magnitude (total angular rate)
gyro_magnitude = np.linalg.norm(gyro_sample, axis=1)

# Calculate accel magnitude
accel_magnitude = np.linalg.norm(accel_sample, axis=1)

# Create multi-panel figure
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Roll angle comparison
axes[0].plot(time_indices, rollTruth_sample, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
axes[0].plot(time_indices, rollComp_sample, 'b--', linewidth=1.5, label='Complementary')
axes[0].plot(time_indices, rollEKF_sample, 'r-.', linewidth=1.5, label='EKF')
axes[0].set_ylabel('Roll Angle (deg)', fontsize=11)
axes[0].set_title('Roll Angle Estimation Comparison with Sensor Dynamics', fontsize=13)
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Panel 2: Estimation errors
axes[1].plot(time_indices, error_complementary, 'b-', linewidth=1.5, label='Complementary Error', alpha=0.7)
axes[1].plot(time_indices, error_ekf, 'r-', linewidth=1.5, label='EKF Error', alpha=0.7)
axes[1].set_ylabel('Absolute Error (deg)', fontsize=11)
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Panel 3: Gyro magnitude (angular rate)
axes[2].plot(time_indices, gyro_magnitude, 'g-', linewidth=1.5, label='Gyro Magnitude')
axes[2].set_ylabel('Angular Rate (rad/s)', fontsize=11)
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

# Panel 4: Accel magnitude
axes[3].plot(time_indices, accel_magnitude, 'm-', linewidth=1.5, label='Accel Magnitude')
axes[3].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[3].set_xlabel('Sample Index', fontsize=12)
axes[3].legend(loc='best')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Results/Figures/Roll_Dynamics_Analysis.png", dpi=150)
print(f"Dynamics analysis plot saved: Roll_Dynamics_Analysis.png")

# Calculate statistics
print("\n=== Performance Analysis ===")
print(f"Sample window: {start_idx} to {end_idx} ({sample_size} samples)")
print(f"\nComplementary Filter:")
print(f"  Mean absolute error: {np.mean(error_complementary):.4f} deg")
print(f"  Max error: {np.max(error_complementary):.4f} deg")
print(f"\nEKF:")
print(f"  Mean absolute error: {np.mean(error_ekf):.4f} deg")
print(f"  Max error: {np.max(error_ekf):.4f} deg")

# Identify high-dynamics periods (top 25% gyro magnitude)
high_dynamics_threshold = np.percentile(gyro_magnitude, 75)
high_dynamics_mask = gyro_magnitude > high_dynamics_threshold
low_dynamics_mask = ~high_dynamics_mask

print(f"\n=== High Dynamics vs Low Dynamics Performance ===")
print(f"High dynamics threshold: {high_dynamics_threshold:.4f} rad/s")
print(f"High dynamics samples: {np.sum(high_dynamics_mask)} ({np.sum(high_dynamics_mask)/len(high_dynamics_mask)*100:.1f}%)")
print(f"\nHigh Dynamics Period:")
print(f"  Complementary MAE: {np.mean(error_complementary[high_dynamics_mask]):.4f} deg")
print(f"  EKF MAE: {np.mean(error_ekf[high_dynamics_mask]):.4f} deg")
print(f"  EKF advantage: {np.mean(error_complementary[high_dynamics_mask]) - np.mean(error_ekf[high_dynamics_mask]):.4f} deg")

print(f"\nLow Dynamics Period:")
print(f"  Complementary MAE: {np.mean(error_complementary[low_dynamics_mask]):.4f} deg")
print(f"  EKF MAE: {np.mean(error_ekf[low_dynamics_mask]):.4f} deg")
print(f"  EKF advantage: {np.mean(error_complementary[low_dynamics_mask]) - np.mean(error_ekf[low_dynamics_mask]):.4f} deg")
