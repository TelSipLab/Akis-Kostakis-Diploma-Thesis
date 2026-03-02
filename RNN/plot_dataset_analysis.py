import csv
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Read dataset
rows = []
with open('Data/dataset_1.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append([float(x) for x in row])

data = np.array(rows)
S = len(rows)
Ts = 0.03
time = np.arange(S) * Ts  # Time in seconds

names_short = [r'$\phi$', r'$\theta$', r'$\psi$',
               r'$\omega_\phi$', r'$\omega_\theta$', r'$\omega_\psi$',
               r'$\tau_\phi$', r'$\tau_\theta$', r'$\tau_\psi$']
names_full = ['Roll', 'Pitch', 'Yaw',
              r'$\omega$ Roll', r'$\omega$ Pitch', r'$\omega$ Yaw',
              r'$\tau$ Roll', r'$\tau$ Pitch', r'$\tau$ Yaw']
units = ['rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'Nm', 'Nm', 'Nm']

# ── Figure 1: Dataset overview - all 9 features ────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Dataset Overview: 3397 Samples at 33.3 Hz', fontsize=14, fontweight='bold')

colors_groups = [
    ['#2980B9', '#3498DB', '#85C1E9'],  # Angles - blues
    ['#E67E22', '#F39C12', '#F5B041'],  # Gyro - oranges
    ['#27AE60', '#2ECC71', '#82E0AA'],  # Torques - greens
]
group_labels = ['Attitude Angles (rad)', 'Angular Velocities (rad/s)', 'Control Torques (Nm)']
group_cols = [(0,1,2), (3,4,5), (6,7,8)]
angle_names = [r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Yaw)']
gyro_names = [r'$\omega_\phi$', r'$\omega_\theta$', r'$\omega_\psi$']
torque_names = [r'$\tau_\phi$', r'$\tau_\theta$', r'$\tau_\psi$']
all_names = [angle_names, gyro_names, torque_names]

for ax_idx, (ax, cols, colors, ylabel, lnames) in enumerate(
    zip(axes, group_cols, colors_groups, group_labels, all_names)):
    for i, (col, color, name) in enumerate(zip(cols, colors, lnames)):
        ax.plot(time, data[:, col], color=color, linewidth=0.8, label=name, alpha=0.9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

axes[-1].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
plt.savefig('RNN/dataset_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('RNN/dataset_overview.pdf', bbox_inches='tight', facecolor='white')
print("Saved: RNN/dataset_overview.png and .pdf")

# ── Figure 2: Angular velocity magnitude - shows dynamic segments ──
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig2.suptitle('Flight Dynamics Analysis', fontsize=14, fontweight='bold')

omega_mag = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)

ax1.plot(time, omega_mag, color='#E74C3C', linewidth=0.7, alpha=0.8)
ax1.fill_between(time, 0, omega_mag, color='#E74C3C', alpha=0.15)
ax1.set_ylabel(r'$|\omega|$ (rad/s)', fontsize=11)
ax1.set_title('Angular Velocity Magnitude', fontsize=11)
ax1.grid(True, alpha=0.3)

# Highlight dynamic segments
ax1.axhline(y=0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.text(time[-1] + 0.5, 0.5, 'threshold', fontsize=8, color='gray', va='center')

# Roll angle in degrees
roll_deg = data[:, 0] * 180.0 / math.pi
pitch_deg = data[:, 1] * 180.0 / math.pi

ax2.plot(time, roll_deg, color='#2980B9', linewidth=0.8, label=r'Roll $\phi$', alpha=0.9)
ax2.plot(time, pitch_deg, color='#E67E22', linewidth=0.8, label=r'Pitch $\theta$', alpha=0.9)
ax2.set_ylabel('Angle (degrees)', fontsize=11)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_title('Attitude Angles', fontsize=11)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('RNN/flight_dynamics.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('RNN/flight_dynamics.pdf', bbox_inches='tight', facecolor='white')
print("Saved: RNN/flight_dynamics.png and .pdf")

# ── Print table for thesis (LaTeX format) ──────────────────────────
print()
print("LaTeX table for thesis:")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Dataset feature statistics (3397 samples)}")
print(r"\begin{tabular}{llrrrr}")
print(r"\hline")
print(r"Feature & Unit & Min & Max & Mean & Std \\")
print(r"\hline")
names_latex = [r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Yaw)',
               r'$\omega_\phi$', r'$\omega_\theta$', r'$\omega_\psi$',
               r'$\tau_\phi$', r'$\tau_\theta$', r'$\tau_\psi$']
for j in range(9):
    col = data[:, j]
    mn, mx, mean, std = col.min(), col.max(), col.mean(), col.std()
    print(f"{names_latex[j]} & {units[j]} & {mn:.4f} & {mx:.4f} & {mean:.4f} & {std:.4f} \\\\")
    if j == 2 or j == 5:
        print(r"\hline")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

plt.show()
