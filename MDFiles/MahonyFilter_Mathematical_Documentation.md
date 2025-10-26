# Mahony Filter Mathematical Documentation

## Complete Mathematical Reference for Passive Complementary Filter on SO(3)

**Author:** Based on implementation in `MahonyFilter.hpp` and `MahonyFilter.cpp`
**Date:** October 2025
**Primary Reference:** Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). "Nonlinear Complementary Filters on the Special Orthogonal Group." *IEEE Transactions on Automatic Control*, 53(5), 1203-1218.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm Description](#algorithm-description)
4. [Implementation Details](#implementation-details)
5. [Code Walkthrough](#code-walkthrough)
6. [Parameter Tuning](#parameter-tuning)
7. [Symbol Glossary](#symbol-glossary)
8. [References](#references)

---

## Overview

The **Mahony Filter** is a **passive complementary filter** that estimates attitude (orientation) by fusing:
- **Gyroscope measurements** (angular velocity)
- **Accelerometer measurements** (gravity direction)

**Key Features:**
- Operates directly on **SO(3)** (Special Orthogonal Group) - the space of 3D rotation matrices
- No gimbal lock issues (unlike Euler angles)
- Computationally efficient (no quaternion operations)
- Globally asymptotically stable under certain conditions
- Single tunable gain parameter (kₚ)

**Advantages over Complementary Filter:**
- Works with full rotation matrices (preserves all orientation information)
- More principled mathematical foundation (Lie group theory)
- Better handling of large rotations

**Comparison with EKF:**
- Simpler: no covariance propagation, no Kalman gain computation
- Faster: fewer matrix operations
- Less adaptive: fixed gain vs. time-varying Kalman gain
- No explicit uncertainty quantification

---

## Mathematical Foundation

### Special Orthogonal Group SO(3)

The set of all 3×3 rotation matrices forms the **Special Orthogonal Group**:

```math
\text{SO}(3) = \{ \mathbf{R} \in \mathbb{R}^{3 \times 3} \mid \mathbf{R}^T \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1 \}
```

**Properties:**
- **Orthogonality**: R^T R = I
- **Determinant**: det(R) = 1
- **Inverse**: R^(-1) = R^T

### Lie Algebra so(3)

The tangent space at identity of SO(3) is the **Lie algebra so(3)**, consisting of all 3×3 skew-symmetric matrices:

```math
\text{so}(3) = \{ \boldsymbol{\Omega} \in \mathbb{R}^{3 \times 3} \mid \boldsymbol{\Omega}^T = -\boldsymbol{\Omega} \}
```

Any skew-symmetric matrix can be represented by a 3D vector via the **skew operator**:

```math
\text{skew}(\boldsymbol{\omega}) = \boldsymbol{\Omega} = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
```

**Properties:**
- skew(ω) · **v** = ω × **v** (cross product)
- For small dt: exp(skew(ω)·dt) ≈ I + skew(ω)·dt (first-order approximation)

### Vex Operator

The inverse of the skew operator is the **vex operator**, which extracts a vector from a skew-symmetric matrix:

```math
\text{vex}(\boldsymbol{\Omega}) = \boldsymbol{\omega} = \begin{bmatrix} \Omega_{32} \\ \Omega_{13} \\ \Omega_{21} \end{bmatrix}
```

**Code Reference:**
```cpp
// MahonyFilter.hpp:40-46
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d s;
    s << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return s;
}

// MahonyFilter.hpp:50-52
static Eigen::Vector3d vex(const Eigen::Matrix3d& M) {
    return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
}
```

### Projection onto Skew-Symmetric Space

For any matrix **M**, the **skew-symmetric projection** is:

```math
P_a(\mathbf{M}) = \frac{1}{2}(\mathbf{M} - \mathbf{M}^T)
```

This extracts the skew-symmetric part of **M**.

---

## Algorithm Description

### Problem Formulation

**Goal:** Estimate the rotation matrix R̂(t) that transforms from body frame to inertial frame.

**Measurements:**
- **ω_y**(t): Gyroscope measurement (angular velocity in body frame, rad/s)
- **R_y**(t): Rotation matrix derived from accelerometer (assuming gravity is the only force)

**State:**
- **R̂**(t) ∈ SO(3): Estimated rotation matrix

### Filter Equations

The Mahony filter consists of the following steps at each time step:

---

#### **Step 1: Compute Rotation Error (Equation 1)**

```math
\tilde{\mathbf{R}} = \hat{\mathbf{R}}^T \mathbf{R}_y
```

Where:
- **R̂**: Current rotation estimate
- **R_y**: Rotation matrix from accelerometer measurement
- **R̃**: Relative rotation error (how much R̂ differs from R_y)

**Interpretation:** If R̂ = R_y (perfect estimate), then R̃ = I (identity).

**Code Reference:**
```cpp
// MahonyFilter.cpp:23
Eigen::Matrix3d rTilda = rHat.transpose() * R_y;
```

---

#### **Step 2: Extract Correction Term (Equation 7)**

```math
\boldsymbol{\omega}_{\text{mes}} = \text{vex}\left( P_a(\tilde{\mathbf{R}}) \right) = \text{vex}\left( \frac{1}{2}(\tilde{\mathbf{R}} - \tilde{\mathbf{R}}^T) \right)
```

Where:
- P_a(·): Skew-symmetric projection operator
- **ω_mes**: Measurement-derived angular velocity correction

**Interpretation:**
- If R̃ ∈ SO(3), then R̃^T R̃ = I, so P_a(R̃) ≠ 0 indicates R̃ is not a pure rotation
- The vex operator extracts the "rotation axis" that would align R̂ with R_y
- This represents the instantaneous correction needed based on accelerometer

**Code Reference:**
```cpp
// MahonyFilter.cpp:26-27
Eigen::Matrix3d Pa_R_tilde = 0.5 * (rTilda - rTilda.transpose());
Eigen::Vector3d omega_mes = vex(Pa_R_tilde);
```

---

#### **Step 3: Combine Angular Velocities (Equation 10)**

```math
\boldsymbol{\omega}_{\text{total}} = \boldsymbol{\omega}_y + k_p \cdot \boldsymbol{\omega}_{\text{mes}}
```

Where:
- **ω_y**: Gyroscope measurement (rad/s)
- k_p: Proportional gain (tuning parameter)
- **ω_mes**: Correction term from accelerometer
- **ω_total**: Fused angular velocity estimate

**Interpretation:**
- **High k_p**: Trust accelerometer more (faster correction, but more sensitive to linear accelerations)
- **Low k_p**: Trust gyroscope more (smoother, but drifts over time)
- This is the **complementary** aspect: gyro for high-frequency, accel for low-frequency

**Code Reference:**
```cpp
// MahonyFilter.cpp:30-31
Eigen::Vector3d omega_total = omega_y + kp * omega_mes;
Eigen::Matrix3d Omega_skew = skew(omega_total);
```

---

#### **Step 4: Propagate Rotation Matrix**

```math
\dot{\hat{\mathbf{R}}} = \hat{\mathbf{R}} \cdot \text{skew}(\boldsymbol{\omega}_{\text{total}})
```

Discrete-time integration (Euler method):

```math
\hat{\mathbf{R}}_{k+1} = \hat{\mathbf{R}}_k + \hat{\mathbf{R}}_k \cdot \text{skew}(\boldsymbol{\omega}_{\text{total}}) \cdot \Delta t
```

Where:
- Δt: Time step (sampling period)
- This is the **kinematic equation** on SO(3)

**Code Reference:**
```cpp
// MahonyFilter.cpp:33
rHat = rHat + rHat * Omega_skew * dt;
```

---

#### **Step 5: Orthonormalization**

After numerical integration, R̂ may drift from SO(3) due to:
- Finite precision arithmetic
- Euler integration approximation

**Re-orthonormalization** using SVD ensures R̂ ∈ SO(3):

```math
\hat{\mathbf{R}} = \mathbf{U} \mathbf{V}^T
```

Where **U**, **V** come from SVD: R̂ = U Σ V^T

**Determinant correction:**
If det(R̂) = -1 (reflection instead of rotation), flip the sign of the last column of **U**.

**Code Reference:**
```cpp
// MahonyFilter.hpp:55-66
void orthonormalize() {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rHat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rHat = svd.matrixU() * svd.matrixV().transpose();

    // Ensure det(R̂) = 1
    if (rHat.determinant() < 0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1;
        rHat = U * svd.matrixV().transpose();
    }
}
```

---

## Implementation Details

### Accelerometer-Based Rotation Matrix

The accelerometer measures specific force (gravity + linear acceleration). Assuming **no linear acceleration**, the measurement points opposite to gravity:

```math
\mathbf{a}_{\text{body}} = \mathbf{R}^T \mathbf{g}_{\text{inertial}}
```

Where:
- **g_inertial** = [0, 0, -9.81]^T (gravity in inertial frame, assuming Z-up)
- **a_body**: Accelerometer reading

**Roll and Pitch from Accelerometer:**

```math
\phi_{\text{accel}} = \arctan2(a_y, a_z) \quad \text{(roll)}
```

```math
\theta_{\text{accel}} = \arctan2(-a_x, \sqrt{a_y^2 + a_z^2}) \quad \text{(pitch)}
```

**Construct Rotation Matrix R_y:**

```math
\mathbf{R}_y = \mathbf{R}_z(0) \cdot \mathbf{R}_y(\theta_{\text{accel}}) \cdot \mathbf{R}_x(\phi_{\text{accel}})
```

Note: Yaw (ψ) is set to 0 because accelerometer cannot observe yaw.

**Code Reference:**
```cpp
// mahonyFilterMain.cpp:61-70
double roll_meas = atan2(accel_norm.y(), accel_norm.z());
double pitch_meas = atan2(-accel_norm.x(),
    sqrt(accel_norm.y() * accel_norm.y() +
        accel_norm.z() * accel_norm.z()));

Eigen::Matrix3d R_y =
    (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *           // yaw = 0
        Eigen::AngleAxisd(pitch_meas, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll_meas, Eigen::Vector3d::UnitX())).toRotationMatrix();
```

---

### Extracting Euler Angles from R̂

Given rotation matrix R̂, extract **ZYX Euler angles** (yaw-pitch-roll):

```math
\phi = \arctan2(R_{32}, R_{33}) \quad \text{(roll)}
```

```math
\theta = \arctan2(-R_{31}, \sqrt{R_{32}^2 + R_{33}^2}) \quad \text{(pitch)}
```

```math
\psi = \arctan2(R_{21}, R_{11}) \quad \text{(yaw)}
```

**Code Reference:**
```cpp
// mahonyFilterMain.cpp:79-82
double rollEst = atan2(R_hat(2, 1), R_hat(2, 2)) * 180.0 / M_PI;
double pitchEst = atan2(-R_hat(2, 0),
    sqrt(R_hat(2, 1) * R_hat(2, 1) +
        R_hat(2, 2) * R_hat(2, 2))) * 180.0 / M_PI;
```

---

## Code Walkthrough

### Class Structure

**Header: `MahonyFilter.hpp`**

```cpp
class MahonyFilter {
public:
    MahonyFilter(double dt, double kp);
    void setData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);
    void update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y);
    void initialize(const Eigen::Matrix3d& R_init);
    Eigen::Vector3d getEulerAngles() const;

    Eigen::Matrix3d rHat;  // Estimated rotation matrix

private:
    double dt;   // Time step (sampling period)
    double kp;   // Proportional gain

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v);
    static Eigen::Vector3d vex(const Eigen::Matrix3d& M);
    void orthonormalize();
};
```

---

### Constructor

**File: `MahonyFilter.cpp:3-8`**

```cpp
MahonyFilter::MahonyFilter(double dt, double kp) :
    rHat(Eigen::Matrix3d::Identity()),  // Initialize to identity (no rotation)
    dt(dt),
    kp(kp) {
}
```

**Initialization:**
- R̂ = I₃ (identity matrix)
- This assumes the IMU starts at zero orientation (aligned with inertial frame)
- Can be changed with `initialize()` method if initial orientation is known

---

### Update Function

**File: `MahonyFilter.cpp:20-36`**

```cpp
void MahonyFilter::update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y) {
    // Step 1: Compute rotation error
    Eigen::Matrix3d rTilda = rHat.transpose() * R_y;

    // Step 2: Extract correction term
    Eigen::Matrix3d Pa_R_tilde = 0.5 * (rTilda - rTilda.transpose());
    Eigen::Vector3d omega_mes = vex(Pa_R_tilde);

    // Step 3: Combine angular velocities
    Eigen::Vector3d omega_total = omega_y + kp * omega_mes;
    Eigen::Matrix3d Omega_skew = skew(omega_total);

    // Step 4: Propagate rotation matrix
    rHat = rHat + rHat * Omega_skew * dt;

    // Step 5: Orthonormalize
    orthonormalize();
}
```

---

### Main Loop

**File: `mahonyFilterMain.cpp:52-98`**

```cpp
for (int i = 0; i < numSamples; i++) {
    // Read sensor data
    Eigen::Vector3d gyroReading = gyroMeasurements.row(i).transpose();
    Eigen::Vector3d accelReading = accelMeasurements.row(i).transpose();
    accelReading(0) = -accelReading(0);  // Sign correction

    // Normalize accelerometer
    Eigen::Vector3d accel_norm = accelReading.normalized();

    // Compute roll/pitch from accelerometer
    double roll_meas = atan2(accel_norm.y(), accel_norm.z());
    double pitch_meas = atan2(-accel_norm.x(),
        sqrt(accel_norm.y() * accel_norm.y() + accel_norm.z() * accel_norm.z()));

    // Construct R_y from accelerometer
    Eigen::Matrix3d R_y = /* ... */;

    // Update filter
    mahony.update(gyroReading, R_y);

    // Extract Euler angles from R̂
    double rollEst = atan2(R_hat(2, 1), R_hat(2, 2)) * 180.0 / M_PI;
    double pitchEst = atan2(-R_hat(2, 0), /* ... */) * 180.0 / M_PI;
}
```

---

## Parameter Tuning

### Proportional Gain (kₚ)

**Physical Meaning:**
- How strongly the accelerometer correction influences the estimate
- Units: dimensionless (scales the correction term ω_mes)

**Tuning Guidelines:**

| kₚ Value | Behavior | Use Case |
|----------|----------|----------|
| **0** | Pure gyro integration (no correction) | Drifts indefinitely |
| **Low (1-10)** | Slow correction, smooth output | Slow-moving systems, high vibration |
| **Medium (10-50)** | Balanced response | General-purpose IMU fusion |
| **High (50-200)** | Fast correction, sensitive to disturbances | Fast dynamics, low vibration |
| **Very High (>200)** | Nearly pure accelerometer | Gyro-free systems (not recommended) |

**Example from code:**
```cpp
// mahonyFilterMain.cpp:12
const double kp = 50.0;  // Aggressive correction
```

**Practical Tuning:**
1. Start with kₚ = 10
2. Increase if drift is observed
3. Decrease if output is too noisy or oscillates
4. Compare RMSE metrics with different values

---

### Time Step (dt)

**Physical Meaning:**
- Sampling period of the IMU (seconds)
- Determines numerical integration accuracy

**Guidelines:**
- Match IMU sampling rate (e.g., 100 Hz → dt = 0.01 s)
- Smaller dt → more accurate integration
- If dt is too large, orthonormalization becomes critical

**Example from code:**
```cpp
// mahonyFilterMain.cpp:11
const double dt = 0.02;  // 50 Hz sampling rate
```

---

## Symbol Glossary

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| **R̂** | 3×3 | Estimated rotation matrix (state) |
| **R_y** | 3×3 | Rotation matrix from accelerometer measurement |
| **R̃** | 3×3 | Rotation error (R̂^T R_y) |
| **ω_y** | 3×1 | Gyroscope measurement (rad/s) |
| **ω_mes** | 3×1 | Measurement-derived correction (rad/s) |
| **ω_total** | 3×1 | Fused angular velocity (rad/s) |
| **Ω** | 3×3 | Skew-symmetric matrix of angular velocity |
| **P_a(·)** | 3×3 → 3×3 | Skew-symmetric projection operator |
| **k_p** | scalar | Proportional gain (tuning parameter) |
| **dt** | scalar | Time step (s) |
| **φ** (phi) | scalar | Roll angle (rad) |
| **θ** (theta) | scalar | Pitch angle (rad) |
| **ψ** (psi) | scalar | Yaw angle (rad) |
| **SO(3)** | - | Special Orthogonal Group (rotation matrices) |
| **so(3)** | - | Lie algebra (skew-symmetric matrices) |

---

## References

### Primary Paper
**Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008).**
*"Nonlinear Complementary Filters on the Special Orthogonal Group."*
IEEE Transactions on Automatic Control, 53(5), 1203-1218.
DOI: [10.1109/TAC.2008.923738](https://doi.org/10.1109/TAC.2008.923738)

**Key Contributions:**
- Rigorous stability analysis on SO(3)
- Extension to integral gain (PI controller on SO(3))
- Comparison with quaternion-based approaches

---

### Related Work

**Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).**
*"Estimation of IMU and MARG Orientation Using a Gradient Descent Algorithm."*
IEEE International Conference on Rehabilitation Robotics.

- Alternative gradient-descent approach
- Similar computational complexity
- Quaternion-based instead of rotation matrices

**Sabatini, A. M. (2006).**
*"Quaternion-Based Extended Kalman Filter for Determining Orientation by Inertial and Magnetic Sensing."*
IEEE Transactions on Biomedical Engineering, 53(7), 1346-1356.

- EKF approach (implemented in this project as `EKF2.cpp`)
- Adaptive gain vs. fixed gain (Mahony)
- Includes gyro bias estimation

---

### Code Files

- **Header:** `include/MahonyFilter.hpp`
- **Implementation:** `src/MahonyFilter.cpp`
- **Main:** `mahonyFilterMain.cpp`
- **Utilities:** `include/Utils.hpp` (RMSE, MEA calculations)

---

## Appendix: Rotation Matrix Conventions

### Body Frame vs. Inertial Frame

**Inertial Frame (World):**
- Fixed reference frame (e.g., North-East-Down or X-Y-Z)
- Gravity vector: **g** = [0, 0, -9.81]^T (assuming Z-up)

**Body Frame (IMU):**
- Attached to the sensor
- X: forward, Y: right, Z: down (typical convention)

**Rotation Matrix R:**
- R transforms vectors from body → inertial
- R^T transforms vectors from inertial → body

**Example:**
```math
\mathbf{v}_{\text{inertial}} = \mathbf{R} \cdot \mathbf{v}_{\text{body}}
```

---

### Sign Conventions

**Accelerometer Sign Correction:**
```cpp
accelReading(0) = -accelReading(0);  // Flip X-axis
```

This accounts for sensor-specific mounting or calibration. Verify with your IMU datasheet.

---

## Performance Characteristics

**Computational Complexity (per update):**
- Matrix multiplication: O(n³) where n=3 → ~54 FLOPs
- SVD orthonormalization: O(n³) → ~150 FLOPs
- Total: ~200-300 FLOPs per update

**Comparison:**
- EKF: ~1000-2000 FLOPs (covariance updates)
- Quaternion Mahony: ~100 FLOPs (no SVD needed)

**Memory:**
- State: 9 floats (3×3 matrix)
- Parameters: 2 floats (dt, kp)

---

**End of Documentation**
