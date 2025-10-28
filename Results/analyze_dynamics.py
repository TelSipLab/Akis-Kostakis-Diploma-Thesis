import matplotlib.pyplot as plt
import numpy as np

# Load datasets
rollComplementary = np.loadtxt('Results/Results/ComplementaryRoll_a_0_79.txt')
rollMahony = np.loadtxt('Results/Results/MahonyRoll_kp_9.txt')
rollEKF = np.loadtxt('Results/Results/EkfRoll.txt')
rollGroundTruth = np.loadtxt('Results/ExpectedResults/expected_roll.txt')

# Load sensor data
gyroData = np.loadtxt('Data/gyro.csv', delimiter=',')
accelData = np.loadtxt('Data/accel.csv', delimiter=',')

# Same sample window as before
total_samples = len(rollGroundTruth)
sample_size = total_samples
start_idx = 0
end_idx = total_samples

# Extract samples
rollComp_sample = rollComplementary[start_idx:end_idx]
rollMahony_sample = rollMahony[start_idx:end_idx]
rollEKF_sample = rollEKF[start_idx:end_idx]
rollTruth_sample = rollGroundTruth[start_idx:end_idx]
time_indices = np.arange(start_idx, end_idx)

# Calculate errors (squared for RMSE calculation)
error_complementary = rollComp_sample - rollTruth_sample
error_mahony = rollMahony_sample - rollTruth_sample
error_ekf = rollEKF_sample - rollTruth_sample

# Absolute errors for plotting
abs_error_complementary = np.abs(error_complementary)
abs_error_mahony = np.abs(error_mahony)
abs_error_ekf = np.abs(error_ekf)

# Extract gyro and accel for this window
gyro_sample = gyroData[start_idx:end_idx, :]
accel_sample = accelData[start_idx:end_idx, :]

# Calculate gyro magnitude (total angular rate)
gyro_magnitude = np.linalg.norm(gyro_sample, axis=1)

# Calculate accel magnitude
accel_magnitude = np.linalg.norm(accel_sample, axis=1)

# Calculate RMSE for legend labels
rmse_complementary = np.sqrt(np.mean(error_complementary**2))
rmse_mahony = np.sqrt(np.mean(error_mahony**2))
rmse_ekf = np.sqrt(np.mean(error_ekf**2))

# Create multi-panel figure
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Roll angle comparison
axes[0].plot(time_indices, rollTruth_sample, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
axes[0].plot(time_indices, rollComp_sample, 'b--', linewidth=1.5, label='Complementary', alpha=0.8)
axes[0].plot(time_indices, rollMahony_sample, 'r-.', linewidth=1.5, label='Mahony', alpha=0.8)
axes[0].plot(time_indices, rollEKF_sample, 'g:', linewidth=2, label='EKF', alpha=0.8)
axes[0].set_ylabel('Roll Angle (deg)', fontsize=11)
axes[0].set_title('Roll Angle Estimation Comparison with Sensor Dynamics', fontsize=13)
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Panel 2: Estimation errors
axes[1].plot(time_indices, abs_error_complementary, 'b-', linewidth=1.5, label=f'Complementary - RMSE: {rmse_complementary:.3f} deg', alpha=0.7)
axes[1].plot(time_indices, abs_error_mahony, 'r-', linewidth=1.5, label=f'Mahony - RMSE: {rmse_mahony:.3f} deg', alpha=0.7)
axes[1].plot(time_indices, abs_error_ekf, 'g-', linewidth=1.5, label=f'EKF - RMSE: {rmse_ekf:.3f} deg', alpha=0.7)
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

# Print RMSE statistics (already calculated above for legend)
print("\n=== Performance Analysis ===")
print(f"Sample window: {start_idx} to {end_idx} ({sample_size} samples)")
print(f"\nComplementary Filter:")
print(f"  RMSE: {rmse_complementary:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_complementary):.4f} deg")
print(f"\nMahony Filter:")
print(f"  RMSE: {rmse_mahony:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_mahony):.4f} deg")
print(f"\nEKF:")
print(f"  RMSE: {rmse_ekf:.4f} deg")
print(f"  Max absolute error: {np.max(abs_error_ekf):.4f} deg")

# Identify high-dynamics periods (top 25% gyro magnitude)
high_dynamics_threshold = np.percentile(gyro_magnitude, 75)
high_dynamics_mask = gyro_magnitude > high_dynamics_threshold
low_dynamics_mask = ~high_dynamics_mask

# Calculate RMSE for high/low dynamics periods
rmse_comp_high = np.sqrt(np.mean(error_complementary[high_dynamics_mask]**2))
rmse_mahony_high = np.sqrt(np.mean(error_mahony[high_dynamics_mask]**2))
rmse_ekf_high = np.sqrt(np.mean(error_ekf[high_dynamics_mask]**2))
rmse_comp_low = np.sqrt(np.mean(error_complementary[low_dynamics_mask]**2))
rmse_mahony_low = np.sqrt(np.mean(error_mahony[low_dynamics_mask]**2))
rmse_ekf_low = np.sqrt(np.mean(error_ekf[low_dynamics_mask]**2))

print(f"\n=== High Dynamics vs Low Dynamics Performance ===")
print(f"High dynamics threshold: {high_dynamics_threshold:.4f} rad/s")
print(f"High dynamics samples: {np.sum(high_dynamics_mask)} ({np.sum(high_dynamics_mask)/len(high_dynamics_mask)*100:.1f}%)")
print(f"\nHigh Dynamics Period:")
print(f"  Complementary RMSE: {rmse_comp_high:.4f} deg")
print(f"  Mahony RMSE: {rmse_mahony_high:.4f} deg")
print(f"  EKF RMSE: {rmse_ekf_high:.4f} deg")
print(f"  Best performer: ", end="")
if rmse_ekf_high < rmse_mahony_high and rmse_ekf_high < rmse_comp_high:
    print(f"EKF (advantage: {min(rmse_comp_high, rmse_mahony_high) - rmse_ekf_high:.4f} deg)")
elif rmse_mahony_high < rmse_comp_high:
    print(f"Mahony (advantage: {rmse_comp_high - rmse_mahony_high:.4f} deg)")
else:
    print(f"Complementary")

print(f"\nLow Dynamics Period:")
print(f"  Complementary RMSE: {rmse_comp_low:.4f} deg")
print(f"  Mahony RMSE: {rmse_mahony_low:.4f} deg")
print(f"  EKF RMSE: {rmse_ekf_low:.4f} deg")
print(f"  Best performer: ", end="")
if rmse_ekf_low < rmse_mahony_low and rmse_ekf_low < rmse_comp_low:
    print(f"EKF (advantage: {min(rmse_comp_low, rmse_mahony_low) - rmse_ekf_low:.4f} deg)")
elif rmse_mahony_low < rmse_comp_low:
    print(f"Mahony (advantage: {rmse_comp_low - rmse_mahony_low:.4f} deg)")
else:
    print(f"Complementary")
