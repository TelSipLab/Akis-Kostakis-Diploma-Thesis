# Extended Kalman Filter: Complete Mathematical Reference
## Quaternion-Based IMU Attitude Estimation

---

## Table of Contents
1. [Overview](#overview)
2. [State Representation](#state-representation)
3. [Mathematical Foundations](#mathematical-foundations)
4. [EKF Algorithm](#ekf-algorithm)
5. [Detailed Mathematics](#detailed-mathematics)
6. [Implementation Details](#implementation-details)
7. [Tuning Parameters](#tuning-parameters)

---

## Overview

### What is the Extended Kalman Filter?

The **Extended Kalman Filter (EKF)** is a recursive state estimator for **nonlinear systems**. It extends the standard Kalman Filter (which works only for linear systems) by **linearizing** the nonlinear dynamics and measurement models at each time step using **Jacobian matrices**.

### Why Use EKF for IMU Attitude Estimation?

- **Sensor Fusion**: Combines gyroscope (angular velocity) and accelerometer (gravity direction) measurements
- **Bias Estimation**: Automatically estimates and compensates for gyroscope bias drift
- **Optimal Filtering**: Provides statistically optimal estimates under Gaussian noise assumptions
- **Quaternion Representation**: Avoids gimbal lock and singularities inherent in Euler angles

### System Overview

```
INPUTS:
├── Gyroscope: ω = [ωx, ωy, ωz]ᵀ (rad/s) - Angular velocity in body frame
└── Accelerometer: a = [ax, ay, az]ᵀ (m/s²) - Specific force in body frame

STATE:
└── x = [q0, q1, q2, q3, bx, by, bz]ᵀ
    ├── q = [q0, q1, q2, q3]ᵀ - Unit quaternion (orientation)
    └── b = [bx, by, bz]ᵀ - Gyroscope bias (rad/s)

OUTPUTS:
├── Roll (φ): Rotation about x-axis
├── Pitch (θ): Rotation about y-axis
└── Yaw (ψ): Rotation about z-axis (not observable with only IMU)
```

---

## State Representation

### State Vector (7D)

$$
\mathbf{x} = \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ b_x \\ b_y \\ b_z \end{bmatrix} \in \mathbb{R}^7
$$

**Components:**

1. **Quaternion** $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$
   - Represents orientation from navigation frame to body frame
   - Unit norm constraint: $\|\mathbf{q}\| = \sqrt{q_0^2 + q_1^2 + q_2^2 + q_3^2} = 1$
   - $q_0$: Scalar (real) part
   - $q_1, q_2, q_3$: Vector (imaginary) parts

2. **Gyroscope Bias** $\mathbf{b} = [b_x, b_y, b_z]^T$
   - Systematic error in gyroscope measurements
   - Slowly time-varying (modeled as random walk)
   - Units: rad/s

### Coordinate Frames

- **Navigation Frame (n)**: Fixed inertial frame
  - Gravity points in +z direction: $\mathbf{g}_n = [0, 0, 1]^T$

- **Body Frame (b)**: Attached to the IMU
  - Rotates with the device
  - Sensors measure in this frame

---

## Mathematical Foundations

### 1. Quaternion Basics

#### Definition
A quaternion represents a rotation in 3D space:

$$
\mathbf{q} = q_0 + q_1\mathbf{i} + q_2\mathbf{j} + q_3\mathbf{k}
$$

where $\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1$

#### Quaternion Multiplication
For two quaternions $\mathbf{q}_1 = [q_{10}, q_{11}, q_{12}, q_{13}]^T$ and $\mathbf{q}_2 = [q_{20}, q_{21}, q_{22}, q_{23}]^T$:

$$
\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
q_{10}q_{20} - q_{11}q_{21} - q_{12}q_{22} - q_{13}q_{23} \\
q_{10}q_{21} + q_{11}q_{20} + q_{12}q_{23} - q_{13}q_{22} \\
q_{10}q_{22} - q_{11}q_{23} + q_{12}q_{20} + q_{13}q_{21} \\
q_{10}q_{23} + q_{11}q_{22} - q_{12}q_{21} + q_{13}q_{20}
\end{bmatrix}
$$

#### Quaternion Kinematics
The time derivative of a quaternion due to angular velocity $\boldsymbol{\omega}$:

$$
\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}
$$

where $\boldsymbol{\Omega}(\boldsymbol{\omega})$ is the **skew-symmetric matrix**:

$$
\boldsymbol{\Omega}(\boldsymbol{\omega}) = \begin{bmatrix}
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
$$

**Physical Interpretation:**
- If you rotate with angular velocity $\boldsymbol{\omega}$ for time $dt$, the quaternion changes by $\dot{\mathbf{q}} \cdot dt$
- The factor $\frac{1}{2}$ comes from quaternion algebra

### 2. Quaternion to Rotation Matrix

A quaternion $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$ converts to rotation matrix:

$$
\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1q_2 - q_0q_3) & 2(q_1q_3 + q_0q_2) \\
2(q_1q_2 + q_0q_3) & q_0^2 - q_1^2 + q_2^2 - q_3^2 & 2(q_2q_3 - q_0q_1) \\
2(q_1q_3 - q_0q_2) & 2(q_2q_3 + q_0q_1) & q_0^2 - q_1^2 - q_2^2 + q_3^2
\end{bmatrix}
$$

**Purpose:** Rotates vectors from navigation frame to body frame
- $\mathbf{v}_b = \mathbf{R}(\mathbf{q}) \mathbf{v}_n$
- $\mathbf{v}_n = \mathbf{R}(\mathbf{q})^T \mathbf{v}_b$ (since $\mathbf{R}^T = \mathbf{R}^{-1}$ for rotation matrices)

### 3. Quaternion to Euler Angles

From quaternion to roll (φ), pitch (θ), yaw (ψ):

$$
\phi = \text{atan2}\left(2(q_0q_1 + q_2q_3), 1 - 2(q_1^2 + q_2^2)\right)
$$

$$
\theta = \text{asin}\left(2(q_0q_2 - q_3q_1)\right)
$$

$$
\psi = \text{atan2}\left(2(q_0q_3 + q_1q_2), 1 - 2(q_2^2 + q_3^2)\right)
$$

**Note:** Pitch has gimbal lock at $\theta = \pm 90°$, handled by checking if $|2(q_0q_2 - q_3q_1)| \geq 1$

---

## EKF Algorithm

The EKF operates in two steps that repeat at each time sample:

### Algorithm Flow

```
FOR each time step k:
  ┌─────────────────────────────────────┐
  │ 1. PREDICT STEP (using gyroscope)   │
  ├─────────────────────────────────────┤
  │ a) Propagate state: x̂⁻ = f(x̂, ω)   │
  │ b) Compute Jacobian: F              │
  │ c) Propagate covariance: P⁻ = FPFᵀ+Q│
  └─────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────┐
  │ 2. UPDATE STEP (using accelerometer)│
  ├─────────────────────────────────────┤
  │ a) Compute innovation: y = z - h(x̂⁻)│
  │ b) Compute Jacobian: H              │
  │ c) Compute Kalman gain: K           │
  │ d) Update state: x̂ = x̂⁻ + Ky       │
  │ e) Update covariance: P = (I-KH)P⁻  │
  └─────────────────────────────────────┘
           ↓
  [Normalize quaternion to maintain unit norm]
END FOR
```

---

## Detailed Mathematics

### PREDICT STEP

#### Step 1: State Prediction

**Process Model:**

$$
\mathbf{x}_k^- = f(\mathbf{x}_{k-1}, \boldsymbol{\omega}_k, \Delta t)
$$

**Detailed Equations:**

1. **Correct gyroscope measurement with bias:**
   $$
   \boldsymbol{\omega}_{\text{corrected}} = \boldsymbol{\omega}_{\text{measured}} - \mathbf{b}_{k-1}
   $$

2. **Propagate quaternion using quaternion kinematics:**
   $$
   \mathbf{q}_k^- = \mathbf{q}_{k-1} + \frac{1}{2}\Delta t \cdot \boldsymbol{\Omega}(\boldsymbol{\omega}_{\text{corrected}}) \mathbf{q}_{k-1}
   $$

   Then normalize: $\mathbf{q}_k^- \leftarrow \frac{\mathbf{q}_k^-}{\|\mathbf{q}_k^-\|}$

3. **Propagate bias (constant model):**
   $$
   \mathbf{b}_k^- = \mathbf{b}_{k-1}
   $$

**Combined state prediction:**
$$
\mathbf{x}_k^- = \begin{bmatrix} \mathbf{q}_k^- \\ \mathbf{b}_k^- \end{bmatrix}
$$

---

#### Step 2: Compute State Transition Jacobian $\mathbf{F}$

The Jacobian $\mathbf{F} = \frac{\partial f}{\partial \mathbf{x}}$ is a 7×7 matrix:

$$
\mathbf{F} = \begin{bmatrix}
\mathbf{F}_{qq} & \mathbf{F}_{qb} \\
\mathbf{0}_{3 \times 4} & \mathbf{I}_{3 \times 3}
\end{bmatrix}
$$

**Block Components:**

1. **$\mathbf{F}_{qq}$ (4×4)**: Derivative of quaternion w.r.t. quaternion
   $$
   \mathbf{F}_{qq} = \mathbf{I}_{4 \times 4} + \frac{1}{2}\Delta t \cdot \boldsymbol{\Omega}(\boldsymbol{\omega}_{\text{corrected}})
   $$

2. **$\mathbf{F}_{qb}$ (4×3)**: Derivative of quaternion w.r.t. bias

   Since $\frac{\partial \mathbf{q}}{\partial \mathbf{b}} = -\frac{1}{2}\Delta t \cdot \frac{\partial \boldsymbol{\Omega}}{\partial \boldsymbol{\omega}} \mathbf{q}$:

   $$
   \mathbf{F}_{qb} = \frac{1}{2}\Delta t \begin{bmatrix}
   -q_1 & -q_2 & -q_3 \\
   q_0 & -q_3 & q_2 \\
   q_3 & q_0 & -q_1 \\
   -q_2 & q_1 & q_0
   \end{bmatrix}
   $$

   **Explanation:** Each column shows how quaternion changes when corresponding bias component changes

3. **$\mathbf{0}_{3 \times 4}$**: Bias doesn't depend on quaternion

4. **$\mathbf{I}_{3 \times 3}$**: Bias propagates as constant (identity mapping)

---

#### Step 3: Covariance Prediction

$$
\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}
$$

**Where:**
- $\mathbf{P}_k^-$: Predicted state covariance (7×7) - uncertainty in state estimate
- $\mathbf{F}$: State transition Jacobian (7×7) - how state evolves
- $\mathbf{P}_{k-1}$: Previous state covariance (7×7)
- $\mathbf{Q}$: Process noise covariance (7×7) - models system uncertainty

**Process Noise Matrix $\mathbf{Q}$:**

$$
\mathbf{Q} = \begin{bmatrix}
\sigma_q^2 \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 3} \\
\mathbf{0}_{3 \times 4} & \sigma_b^2 \mathbf{I}_{3 \times 3}
\end{bmatrix}
$$

- $\sigma_q^2$: Quaternion process noise variance (typically $\sim 0.001$)
- $\sigma_b^2$: Bias random walk variance (typically $\sim 0.0001$)

---

### UPDATE STEP

#### Step 1: Measurement Model

**Accelerometer Measurement:**
The accelerometer measures the gravity vector in the body frame (assuming no external accelerations):

$$
\mathbf{z}_k = \mathbf{a}_{\text{measured}} = \mathbf{R}(\mathbf{q})^T \mathbf{g}_n + \mathbf{v}_k
$$

where:
- $\mathbf{g}_n = [0, 0, 1]^T$: Normalized gravity in navigation frame (pointing up)
- $\mathbf{v}_k$: Measurement noise $\sim \mathcal{N}(0, \mathbf{R})$

**Expected Measurement (Predicted):**

$$
\mathbf{h}(\mathbf{x}_k^-) = \mathbf{R}(\mathbf{q}_k^-)^T \mathbf{g}_n
$$

This is what we *expect* the accelerometer to read given our predicted orientation.

---

#### Step 2: Innovation (Measurement Residual)

$$
\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\mathbf{x}_k^-)
$$

where $\mathbf{z}_k = \frac{\mathbf{a}_{\text{measured}}}{\|\mathbf{a}_{\text{measured}}\|}$ (normalized)

**Physical Interpretation:**
- $\mathbf{y}_k$ is the difference between what we measured and what we predicted
- Tells us how much to correct our state estimate

---

#### Step 3: Compute Measurement Jacobian $\mathbf{H}$

The Jacobian $\mathbf{H} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}$ is a 3×7 matrix:

$$
\mathbf{H} = \begin{bmatrix}
\mathbf{H}_q & \mathbf{0}_{3 \times 3}
\end{bmatrix}
$$

**Block Components:**

1. **$\mathbf{H}_q$ (3×4)**: Derivative of measurement w.r.t. quaternion

   Since $\mathbf{h} = \mathbf{R}(\mathbf{q})^T \mathbf{g}_n$ and $\mathbf{g}_n = [0, 0, 1]^T$:

   $$
   \mathbf{H}_q = \frac{\partial}{\partial \mathbf{q}} \left( \mathbf{R}(\mathbf{q})^T \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \right)
   $$

   This equals the derivative of the third column of $\mathbf{R}^T$ w.r.t. quaternion:

   $$
   \mathbf{H}_q = 2 \begin{bmatrix}
   q_0 g_x + q_3 g_y - q_2 g_z & q_1 g_x + q_2 g_y + q_3 g_z & -q_2 g_x + q_1 g_y - q_0 g_z & -q_3 g_x + q_0 g_y + q_1 g_z \\
   -q_3 g_x + q_0 g_y + q_1 g_z & q_2 g_x - q_1 g_y + q_0 g_z & q_1 g_x + q_2 g_y + q_3 g_z & -q_0 g_x - q_3 g_y + q_2 g_z \\
   q_2 g_x - q_1 g_y + q_0 g_z & q_3 g_x - q_0 g_y - q_1 g_z & q_0 g_x + q_3 g_y - q_2 g_z & q_1 g_x + q_2 g_y + q_3 g_z
   \end{bmatrix}
   $$

   With $\mathbf{g}_n = [0, 0, 1]^T$, this simplifies to:

   $$
   \mathbf{H}_q = 2 \begin{bmatrix}
   -q_2 & q_3 & -q_0 & q_1 \\
   q_1 & q_0 & q_3 & q_2 \\
   q_0 & -q_1 & -q_2 & q_3
   \end{bmatrix}
   $$

2. **$\mathbf{0}_{3 \times 3}$**: Measurement doesn't depend on bias

---

#### Step 4: Innovation Covariance

$$
\mathbf{S}_k = \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R}
$$

**Where:**
- $\mathbf{S}_k$: Innovation covariance (3×3) - uncertainty in innovation
- $\mathbf{R}$: Measurement noise covariance (3×3)

**Measurement Noise Matrix $\mathbf{R}$:**

$$
\mathbf{R} = \sigma_a^2 \mathbf{I}_{3 \times 3}
$$

- $\sigma_a^2$: Accelerometer noise variance (typically $\sim 0.1$)

---

#### Step 5: Kalman Gain

$$
\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \mathbf{S}_k^{-1}
$$

**Physical Interpretation:**
- $\mathbf{K}_k$ is a 7×3 matrix
- Determines how much to trust the measurement vs. the prediction
- High measurement noise → Low gain → Trust prediction more
- High prediction uncertainty → High gain → Trust measurement more

---

#### Step 6: State Update

$$
\mathbf{x}_k = \mathbf{x}_k^- + \mathbf{K}_k \mathbf{y}_k
$$

**Correction:**
- Add weighted innovation to predicted state
- Weighted by Kalman gain (optimal weighting)

**Then normalize quaternion:**
$$
\mathbf{q}_k \leftarrow \frac{\mathbf{q}_k}{\|\mathbf{q}_k\|}
$$

---

#### Step 7: Covariance Update

$$
\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^-
$$

**Alternative (Joseph form - numerically stable):**
$$
\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^- (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T
$$

**Physical Interpretation:**
- Reduces uncertainty after incorporating measurement
- Covariance always decreases (we gain information)

---

## Implementation Details

### Initialization

**1. Initial Quaternion from Accelerometer:**

Assuming the device starts stationary, the accelerometer measures only gravity:

$$
\phi_0 = \text{atan2}(a_y, a_z)
$$

$$
\theta_0 = \text{atan2}(-a_x, \sqrt{a_y^2 + a_z^2})
$$

$$
\psi_0 = 0
$$

Then convert Euler angles to quaternion:

$$
q_0 = \cos(\phi_0/2)\cos(\theta_0/2)\cos(\psi_0/2) + \sin(\phi_0/2)\sin(\theta_0/2)\sin(\psi_0/2)
$$
$$
q_1 = \sin(\phi_0/2)\cos(\theta_0/2)\cos(\psi_0/2) - \cos(\phi_0/2)\sin(\theta_0/2)\sin(\psi_0/2)
$$
$$
q_2 = \cos(\phi_0/2)\sin(\theta_0/2)\cos(\psi_0/2) + \sin(\phi_0/2)\cos(\theta_0/2)\sin(\psi_0/2)
$$
$$
q_3 = \cos(\phi_0/2)\cos(\theta_0/2)\sin(\psi_0/2) - \sin(\phi_0/2)\sin(\theta_0/2)\cos(\psi_0/2)
$$

**2. Initial Covariance:**

$$
\mathbf{P}_0 = \begin{bmatrix}
0.1 \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 3} \\
\mathbf{0}_{3 \times 4} & 0.01 \mathbf{I}_{3 \times 3}
\end{bmatrix}
$$

- Higher uncertainty in quaternion (0.1) - less confident about initial orientation
- Lower uncertainty in bias (0.01) - assume bias starts near zero

---

### Discrete-Time Integration

The continuous quaternion kinematics:
$$
\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}) \mathbf{q}
$$

**Euler Integration (first-order):**
$$
\mathbf{q}_{k+1} = \mathbf{q}_k + \Delta t \cdot \dot{\mathbf{q}}_k
$$
$$
\mathbf{q}_{k+1} = \mathbf{q}_k + \frac{\Delta t}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_k) \mathbf{q}_k
$$

**Better alternatives (not currently implemented):**
- **RK4 (Runge-Kutta 4th order)**: More accurate for large $\Delta t$
- **Exponential map**: Preserves quaternion constraints better

---

### Numerical Stability

**1. Quaternion Normalization:**
- After every predict and update step
- Prevents numerical drift from unit norm constraint
- Simple normalization: $\mathbf{q} \leftarrow \mathbf{q} / \|\mathbf{q}\|$

**2. Covariance Symmetry:**
- Ensure $\mathbf{P}$ remains symmetric: $\mathbf{P} = \frac{1}{2}(\mathbf{P} + \mathbf{P}^T)$
- Use Joseph form for update to maintain positive definiteness

**3. Matrix Inversion:**
- Inverse of $\mathbf{S}_k$ in Kalman gain
- Use Cholesky decomposition or pseudo-inverse for numerical stability

---

## Tuning Parameters

### Process Noise Covariance $\mathbf{Q}$

**Quaternion Process Noise ($\sigma_q^2$):**
- **Higher value**: Trusts gyroscope less, adapts faster to changes
- **Lower value**: Trusts gyroscope more, smoother estimates
- **Typical range**: 0.0001 to 0.01
- **Current**: 0.001

**Bias Process Noise ($\sigma_b^2$):**
- **Higher value**: Allows bias to change faster (more adaptive)
- **Lower value**: Assumes bias is very stable
- **Typical range**: 0.00001 to 0.001
- **Current**: 0.0001

### Measurement Noise Covariance $\mathbf{R}$

**Accelerometer Noise ($\sigma_a^2$):**
- **Higher value**: Trusts accelerometer less (useful during dynamic motion)
- **Lower value**: Trusts accelerometer more (better for static/quasi-static)
- **Typical range**: 0.01 to 1.0
- **Current**: 0.1

### Initial Covariance $\mathbf{P}_0$

**Quaternion Initial Uncertainty:**
- **Higher value**: Less confident in initial orientation
- **Current**: 0.1

**Bias Initial Uncertainty:**
- **Higher value**: Less confident that bias starts at zero
- **Current**: 0.01

### Tuning Strategy

1. **Start with conservative values** (mid-range)
2. **Test under different conditions:**
   - Static (no movement)
   - Slow rotation
   - Fast rotation
   - External accelerations
3. **Adjust based on performance:**
   - **High steady-state error** → Increase $\mathbf{Q}$
   - **Oscillations** → Decrease $\mathbf{Q}$ or increase $\mathbf{R}$
   - **Slow convergence** → Increase $\mathbf{Q}$ or decrease $\mathbf{P}_0$
   - **Sensitive to vibration** → Increase $\mathbf{R}$

---

## Summary of Key Equations

### Predict Step
1. $\boldsymbol{\omega}_{\text{corrected}} = \boldsymbol{\omega}_{\text{measured}} - \mathbf{b}$
2. $\mathbf{q}_k^- = \mathbf{q}_{k-1} + \frac{\Delta t}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_{\text{corrected}}) \mathbf{q}_{k-1}$
3. $\mathbf{F} = \begin{bmatrix} \mathbf{I} + \frac{\Delta t}{2}\boldsymbol{\Omega} & \mathbf{F}_{qb} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}$
4. $\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$

### Update Step
1. $\mathbf{z}_k = \mathbf{a}_{\text{measured}} / \|\mathbf{a}_{\text{measured}}\|$
2. $\mathbf{h}(\mathbf{x}_k^-) = \mathbf{R}(\mathbf{q}_k^-)^T [0, 0, 1]^T$
3. $\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\mathbf{x}_k^-)$
4. $\mathbf{H} = [\mathbf{H}_q \quad \mathbf{0}]$
5. $\mathbf{S}_k = \mathbf{H} \mathbf{P}_k^- \mathbf{H}^T + \mathbf{R}$
6. $\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}^T \mathbf{S}_k^{-1}$
7. $\mathbf{x}_k = \mathbf{x}_k^- + \mathbf{K}_k \mathbf{y}_k$
8. $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_k^-$

---

## Common Issues and Solutions

### Issue 1: Quaternion Norm Drift
**Symptom:** Quaternion norm deviates from 1.0 over time
**Solution:** Normalize quaternion after every predict and update step

### Issue 2: Covariance Matrix Not Positive Definite
**Symptom:** Matrix inversion fails, negative eigenvalues
**Solution:**
- Use Joseph form for covariance update
- Add small regularization: $\mathbf{P} = \mathbf{P} + \epsilon \mathbf{I}$ (where $\epsilon \sim 10^{-9}$)

### Issue 3: Filter Divergence
**Symptom:** Estimates drift away from truth
**Solution:**
- Increase process noise $\mathbf{Q}$
- Check for bugs in Jacobian computation
- Verify measurement model correctness

### Issue 4: Oscillations
**Symptom:** Estimates oscillate around true value
**Solution:**
- Decrease process noise $\mathbf{Q}$
- Increase measurement noise $\mathbf{R}$
- Check time step $\Delta t$ (may be too large)

### Issue 5: Poor Performance During Dynamic Motion
**Symptom:** Large errors when accelerometer experiences non-gravitational accelerations
**Solution:**
- Detect high dynamics (check gyro magnitude or accel magnitude deviation from g)
- Adaptively increase $\mathbf{R}$ during detected dynamic periods
- Consider switching to gyro-only mode temporarily

---

## References

1. **Primary Reference:**
   - Sabatini (2011): "Quaternion-based extended Kalman filter for determining orientation by inertial and magnetic sensing"

2. **Quaternion Math:**
   - Kuipers, J. B. (1999): "Quaternions and Rotation Sequences"

3. **Kalman Filtering:**
   - Welch & Bishop: "An Introduction to the Kalman Filter"
   - Simon, D. (2006): "Optimal State Estimation"

---

## Appendix: Matrix Dimensions Summary

| Variable | Dimensions | Description |
|----------|-----------|-------------|
| $\mathbf{x}$ | 7×1 | State vector |
| $\mathbf{q}$ | 4×1 | Quaternion |
| $\mathbf{b}$ | 3×1 | Gyro bias |
| $\mathbf{P}$ | 7×7 | State covariance |
| $\mathbf{Q}$ | 7×7 | Process noise covariance |
| $\mathbf{R}$ | 3×3 | Measurement noise covariance |
| $\mathbf{F}$ | 7×7 | State transition Jacobian |
| $\mathbf{H}$ | 3×7 | Measurement Jacobian |
| $\mathbf{K}$ | 7×3 | Kalman gain |
| $\mathbf{S}$ | 3×3 | Innovation covariance |
| $\mathbf{y}$ | 3×1 | Innovation (residual) |
| $\mathbf{z}$ | 3×1 | Measurement |
| $\boldsymbol{\omega}$ | 3×1 | Angular velocity |
| $\mathbf{R}(\mathbf{q})$ | 3×3 | Rotation matrix |
| $\boldsymbol{\Omega}$ | 4×4 | Quaternion derivative matrix |

---

**Document Version:** 1.0
**Date:** 2025-10-28
**Status:** Complete Mathematical Reference
