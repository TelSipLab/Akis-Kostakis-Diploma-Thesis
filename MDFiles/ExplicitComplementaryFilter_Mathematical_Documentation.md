# Explicit Complementary Filter Mathematical Documentation

## Complete Mathematical Reference for Explicit Complementary Filter on SO(3)

**Author:** Based on implementation in `ExplicitComplementaryFilter.hpp` and `ExplicitComplementaryFilter.cpp`
**Date:** January 2025
**Primary Reference:** Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). "Nonlinear Complementary Filters on the Special Orthogonal Group." *IEEE Transactions on Automatic Control*, 53(5), 1203-1218. **Section V (pages 8-11)**

---

## Table of Contents

1. [Overview](#overview)
2. [Why "Explicit"?](#why-explicit)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Filter Equations](#filter-equations)
5. [Single Vector Case (Accelerometer Only)](#single-vector-case)
6. [Bias Estimation Theory](#bias-estimation-theory)
7. [Implementation Details](#implementation-details)
8. [Code Walkthrough](#code-walkthrough)
9. [Parameter Tuning](#parameter-tuning)
10. [Comparison with Other Filters](#comparison-with-other-filters)
11. [Symbol Glossary](#symbol-glossary)
12. [References](#references)

---

## Overview

The **Explicit Complementary Filter** is an advanced attitude estimation algorithm that fuses:
- **Gyroscope measurements** (angular velocity with bias)
- **Accelerometer measurements** (gravity direction)
- **Optional: Magnetometer measurements** (magnetic field direction)

**Key Innovation:** Works directly with vectorial measurements without requiring full attitude reconstruction.

### Key Features

✅ **No Attitude Reconstruction Required** - Uses raw sensor vectors directly
✅ **Gyro Bias Estimation** - Estimates bias in all 3 axes
✅ **Works with Single Vector** - Accelerometer-only operation supported
✅ **Almost Global Stability** - Proven convergence for almost all initial conditions
✅ **Low Computational Cost** - Ideal for embedded systems
✅ **No Magnetometer Required** - Can estimate yaw bias without magnetometer

### Advantages over Passive Mahony Filter

| Feature | Passive Mahony | Explicit CF |
|---------|----------------|-------------|
| Attitude reconstruction | Required (R<sub>y</sub>) | ❌ Not required ✅ |
| Computational cost | Medium | ✅ Lower |
| Works without magnetometer | No (yaw drifts) | ✅ Yes |
| Bias estimation | All axes | ✅ All axes |
| Code complexity | Medium | ✅ Simpler |

### Advantages over EKF

| Feature | EKF | Explicit CF |
|---------|-----|-------------|
| Optimality | Optimal (Gaussian) | Suboptimal |
| Tuning | Complex (Q, R) | ✅ Simple (k<sub>p</sub>, k<sub>i</sub>) |
| Computational cost | High (7×7 covariance) | ✅ Very low |
| Robustness | Sensitive | ✅ Robust |
| Convergence | Local | ✅ Almost global |

---

## Why "Explicit"?

### The Problem with Traditional Approach

**Traditional complementary filters (including Passive Mahony) require:**

```
Step 1: Reconstruct attitude from sensor measurements
        a = accelerometer reading
        m = magnetometer reading
        ↓
        φ = atan2(ay, az)                    (roll)
        θ = atan2(-ax, √(ay² + az²))         (pitch)
        ↓
        Ry = rotation_matrix(φ, θ, ψ)        (reconstructed attitude)

Step 2: Use Ry in filter equations
```

**Problems with this approach:**
- ❌ Computational overhead (trigonometry + matrix construction)
- ❌ Loss of information when converting to Euler angles
- ❌ Poor error characterization
- ❌ Numerical issues near singularities (gimbal lock)
- ❌ What if magnetometer is unavailable? Cannot get full R<sub>y</sub>
- ❌ Noise in R<sub>y</sub> propagates into filter

### The Explicit Solution

**Instead of reconstructing R<sub>y</sub>, work with directional measurements:**

```
Inertial frame:   v₀ = [0, 0, 1]ᵀ         (gravity points up)

Measured:         v = Rᵀ v₀               (gravity in body frame)

Estimated:        v̂ = R̂ᵀ v₀              (estimated gravity in body frame)

Error:            ωₘₑₛ = v × v̂          (correction direction!)
```

**Why this works:**
- ✅ Cross product `v × v̂` gives rotation axis needed to align them
- ✅ Magnitude `|v × v̂| = sin(θ)` where θ is angle between vectors
- ✅ Automatically zero when aligned (no overcorrection)
- ✅ Clean geometric interpretation
- ✅ Numerically stable

---

## Mathematical Foundation

### Coordinate Frames

- **{A}** = Inertial frame (fixed in space)
- **{B}** = Body frame (attached to IMU)
- **R** ∈ SO(3) = True rotation matrix from {B} to {A}
- **R̂** ∈ SO(3) = Estimated rotation matrix

### Sensor Models

**Gyroscope:**
```
Ωʸ = Ω + b + μ

where:
  Ω   = true angular velocity (rad/s)
  b   = slowly varying bias (rad/s)
  μ   = zero-mean noise (rad/s)
```

**Accelerometer:**
```
a = Rᵀ(v̇ - g₀) + bₐ + μₐ

For low-frequency motion (v̇ ≈ 0):
  v = a/|a| ≈ -Rᵀ e₃

where:
  g₀  = [0, 0, -9.81]ᵀ m/s² (gravity in inertial frame)
  e₃  = [0, 0, 1]ᵀ (upward unit vector)
  v   = normalized gravity direction in body frame
```

### Cost Function (Section V, Equation 30)

Instead of minimizing attitude error, minimize **directional error**:

```
Eₘₑₛ = Σᵢ kᵢ Eᵢ

where Eᵢ = 1 - ⟨vᵢ, v̂ᵢ⟩ = 1 - vᵢᵀ v̂ᵢ
```

**Geometric meaning:**
- `⟨v, v̂⟩ = cos(θ)` where θ is angle between vectors
- `E = 1 - cos(θ)` is zero when aligned, maximum (=2) when opposite
- For small angles: `E ≈ θ²/2` (quadratic cost)

**For single vector (gravity only):**
```
Eₘₑₛ = k(1 - vᵀv̂)
```

where:
- `v₀ = [0, 0, 1]ᵀ` (gravity direction in inertial frame)
- `v = Rᵀv₀` (measured gravity in body frame, normalized)
- `v̂ = R̂ᵀv₀` (estimated gravity in body frame)

---

## Filter Equations

### The Three Core Equations (Equation 32)

From paper Section V, Equation (32):

```
(32a)  Ṙ̂ = R̂ [Ωʸ - b̂ + kₚ ωₘₑₛ]ₓ

(32b)  ḃ̂ = -kᵢ ωₘₑₛ

(32c)  ωₘₑₛ = Σᵢ kᵢ (vᵢ × v̂ᵢ)
```

### Explanation of Each Term

**Equation (32c) - Measurement Correction:**
```
ωₘₑₛ = v × v̂
```
- Cross product gives **rotation axis** perpendicular to both vectors
- This is exactly the axis needed to rotate v̂ toward v
- Magnitude: `|v × v̂| = sin(θ)` provides proportional correction
- When aligned (v = v̂), correction is zero

**Equation (32a) - Rotation Update:**
```
Ṙ̂ = R̂ [Ωʸ - b̂ + kₚ ωₘₑₛ]ₓ
```
- `Ωʸ`: Gyro measurement (prediction, high frequency)
- `-b̂`: Bias correction (remove drift)
- `kₚ ωₘₑₛ`: Proportional correction (align with measurements, low frequency)
- `[·]ₓ`: Skew-symmetric matrix operator

**Equation (32b) - Bias Update:**
```
ḃ̂ = -kᵢ ωₘₑₛ
```
- Integral correction for bias estimation
- Drives ωₘₑₛ → 0 over time
- Ensures b̂ → b (true bias)

### Discrete-Time Implementation

For numerical integration with time step Δt:

```cpp
// Step 1: Compute estimated gravity direction in body frame
v̂ = R̂ᵀ v₀

// Step 2: Compute measurement correction
ωₘₑₛ = v × v̂

// Step 3: Update bias estimate
b̂(k+1) = b̂(k) - kᵢ ωₘₑₛ Δt

// Step 4: Compute total angular velocity
ω_total = Ωʸ - b̂ + kₚ ωₘₑₛ

// Step 5: Update rotation matrix
R̂(k+1) = R̂(k) + R̂(k) [ω_total]ₓ Δt

// Step 6: Project back to SO(3)
R̂(k+1) = orthonormalize(R̂(k+1))
```

---

## Single Vector Case

### Setup (Corollary 5.2, Page 11)

**Given:**
- Only accelerometer: `v = Rᵀ v₀` where `v₀ = [0, 0, 1]ᵀ`
- No magnetometer (yaw is unobservable from single measurement)

**Question:** Can we still estimate gyro bias in all 3 axes, including yaw?

**Answer:** ✅ **YES!** (Under certain conditions)

### Filter Equations for Single Vector

```
ωₘₑₛ = v × v̂

Ṙ̂ = R̂ [(Ωʸ - b̂ + kₚ ωₘₑₛ)]ₓ

ḃ̂ = -kᵢ ωₘₑₛ
```

### Why Yaw Bias Can Be Estimated

**Key Insight from paper proof (page 11):**

Even though yaw angle cannot be directly measured from gravity alone, the **yaw bias** can still be estimated if the system rotates over time.

**Intuitive Explanation:**

1. **Yaw bias causes drift** around vertical axis (Z)
2. **When system pitches/rolls**, this Z-drift couples into X/Y axes
3. **Accelerometer detects this coupled error** in measurable axes
4. **Filter learns yaw bias** from the coupling over time

**Mathematical Condition (Theorem 5.1):**

The signals Ω(t) and v(t) must be **"asymptotically independent"**

**Practical meaning:**
> "The IMU must experience varied motion - not just hovering still"

Normal vehicle/robot operation automatically satisfies this!

### Practical Example

```
Scenario: Quadcopter with only accelerometer + gyro

t = 0-10s:   Hovering still
             → Roll/pitch accurate ✅
             → Yaw bias NOT observable ❌

t = 10-20s:  Pitch forward, roll left, yaw turns, etc.
             → Yaw bias becomes observable through coupling ✅
             → b̂ₓ, b̂ᵧ, b̂ᵧ all converge to true values ✅

t = 20-30s:  Hovering again
             → Yaw doesn't drift anymore! ✅
             → Bias already estimated ✅
```

### Experimental Validation (Figure 9, Page 13)

From paper's HoverEye UAV experiment:

**Without bias estimation:**
- Yaw drift: ~30-40° over 2 minutes ❌

**With Explicit Complementary Filter:**
- Yaw drift: **< 5°** over 2 minutes ✅
- Bias convergence time: ~10-20 seconds ✅
- All three axes (roll, pitch, yaw bias) converged ✅

---

## Bias Estimation Theory

### Lyapunov Stability Analysis (Theorem 5.1)

**Error variables:**
```
R̃ = R̂ᵀ R        (rotation error)
b̃ = b - b̂        (bias error)
```

**Lyapunov function:**
```
V = Eₘₑₛ + (1/kᵢ)|b̃|²

  = Σᵢ kᵢ(1 - vᵢᵀv̂ᵢ) + (1/kᵢ)|b̃|²
```

**Time derivative along filter trajectories:**
```
V̇ = -kₚ |ωₘₑₛ|² ≤ 0
```

**What this proves:**

1. `V̇ ≤ 0` → System is **Lyapunov stable**
2. `V̇ = 0` only when `ωₘₑₛ = 0`
3. `ωₘₑₛ = 0` only when:
   - `v = v̂` (measurements aligned) AND
   - `b̂ = b` (bias correct)
4. Therefore: **(R̂, b̂) → (R, b)** asymptotically ✅

### Convergence Properties (from Theorem 5.1)

**1. Local Exponential Stability**
- Small errors decay exponentially fast
- Linearization around (R̃, b̃) = (I, 0) is stable
- Eigenvalues of linearization have negative real parts

**2. Almost Global Convergence**
- For almost all initial conditions: (R̂, b̂) → (R, b)
- Exception: measure-zero set of "bad" initial conditions
- In practice: **always converges** ✅

**3. Unstable Equilibria**
- Unstable points exist at tr(R̃) = -1 (180° rotations), b̃ = 0
- These are **repelling** (system moves away from them)
- Never reached in practice

### Error Dynamics (Equations 37-38)

From the paper, the error dynamics are:

```
R̃˙ = [R̃, Ωₓ] - kₚ ωₘₑₛₓ R̃ - b̃ₓ R̃

b̃˙ = kᵢ ωₘₑₛ
```

where `[A, B] = AB - BA` is the matrix commutator.

**Key Property - Passive Coupling:**
```
tr([R̃, Ωₓ]) = 0
```

The gyro term doesn't contribute to Lyapunov derivative!
The filter is **"passive"** to gyro perturbations.

---

## Implementation Details

### Orthonormalization

After each update, R̂ drifts from SO(3) due to numerical errors.

**Why needed:**
```
Discrete integration: R̂ ← R̂ + R̂[ω]ₓ Δt

This doesn't preserve: RᵀR = I (orthonormality)
                       det(R) = 1 (orientation)
```

**Solution: SVD Projection**
```cpp
// Singular Value Decomposition
R̂ = U Σ Vᵀ

// Nearest rotation matrix (set all singular values to 1)
R̂_ortho = U Vᵀ

// Ensure proper rotation (det = +1, not -1)
if (det(R̂_ortho) < 0) {
    U[:, 2] *= -1  // Flip last column
    R̂_ortho = U Vᵀ
}
```

**When to apply:**
- Every step (recommended for safety) ✅
- Every 10 steps (faster, usually okay)
- When |det(R̂) - 1| > threshold

### Initial Conditions

**Recommended initialization:**

```cpp
// 1. Collect first accelerometer sample
v_init = accel_first_sample.normalized()

// 2. Initialize rotation to align Z-axis with -gravity
//    (This gives correct roll/pitch, arbitrary yaw)
R̂₀ = rotationAlignZ(v_init)

// 3. Initialize bias to zero
b̂₀ = [0, 0, 0]ᵀ
```

**Alternative (faster convergence):**
```cpp
// Collect N samples while IMU is stationary
// Average gyro readings to estimate initial bias
b̂₀ = mean(gyro_samples[0:N])

// Still initialize R̂₀ from accelerometer
R̂₀ = rotationAlignZ(v_init)
```

### Handling Multiple Vectors

If both accelerometer and magnetometer are available:

```cpp
// Reference vectors (inertial frame)
v₀_gravity = [0, 0, 1]ᵀ
v₀_magnetic = [cos(declination), 0, sin(declination)]ᵀ

// Measured vectors (body frame, normalized)
v_accel = accelerometer.normalized()
v_mag = magnetometer.normalized()

// Estimated vectors
v̂_accel = R̂ᵀ * v₀_gravity
v̂_mag = R̂ᵀ * v₀_magnetic

// Weighted correction
ωₘₑₛ = k₁(v_accel × v̂_accel) + k₂(v_mag × v̂_mag)
```

**Weight selection guidelines:**

| Situation | k₁ (accel) | k₂ (mag) | Reason |
|-----------|-----------|----------|---------|
| Normal operation | 1.0 | 1.0 | Equal trust |
| High acceleration | 0.1 | 2.0 | Accel unreliable |
| Magnetic disturbance | 2.0 | 0.1 | Mag unreliable |
| Indoor (no mag) | 1.0 | 0.0 | Mag unavailable |

---

## Code Walkthrough

### Constructor

```cpp
ExplicitComplementaryFilter::ExplicitComplementaryFilter(
    double dt, double kp, double ki)
    : dt(dt), kp(kp), ki(ki),
      rHat(Eigen::Matrix3d::Identity()),
      biasEstimate(Eigen::Vector3d::Zero()) {

    // Initialize gravity reference vector (inertial frame)
    v0_gravity << 0.0, 0.0, 1.0;  // e₃ = [0,0,1]ᵀ
}
```

**Initialization:**
- R̂ starts at identity (no rotation)
- b̂ starts at zero (no bias)
- v₀ = [0, 0, 1]ᵀ (gravity points up in inertial frame)

### Main Update Loop

```cpp
void ExplicitComplementaryFilter::predictForAllData() {
    for(int i = 0; i < gyroData.rows(); i++) {
        // Read sensor data
        Eigen::Vector3d gyroReading = gyroData.row(i).transpose();
        Eigen::Vector3d accelReading = accelerometerData.row(i).transpose();

        // Sign correction (sensor-specific)
        accelReading(0) = -accelReading(0);

        // Normalize accelerometer → gravity direction
        Eigen::Vector3d v_measured = accelReading.normalized();

        // Run filter update
        update(gyroReading, v_measured);

        // Extract Euler angles from R̂
        double roll = atan2(rHat(2,1), rHat(2,2));
        double pitch = atan2(-rHat(2,0), sqrt(rHat(2,1)² + rHat(2,2)²));

        // Store results
        rollEstimation(i) = roll;
        pitchEstimation(i) = pitch;
    }
}
```

### Core Update Function

```cpp
void ExplicitComplementaryFilter::update(
    const Eigen::Vector3d& omega_y,      // Gyro measurement
    const Eigen::Vector3d& v_measured) { // Normalized accel

    // STEP 1: Compute estimated gravity in body frame
    //         v̂ = R̂ᵀ v₀
    Eigen::Vector3d v_estimated = rHat.transpose() * v0_gravity;

    // STEP 2: Compute measurement correction (Eq 32c)
    //         ωₘₑₛ = v × v̂
    Eigen::Vector3d omega_mes = v_measured.cross(v_estimated);

    // STEP 3: Update bias estimate (Eq 32b)
    //         ḃ̂ = -kᵢ ωₘₑₛ
    biasEstimate = biasEstimate - ki * omega_mes * dt;

    // STEP 4: Compute total angular velocity (Eq 32a)
    //         ω = Ωʸ - b̂ + kₚ ωₘₑₛ
    Eigen::Vector3d omega_total = omega_y - biasEstimate + kp * omega_mes;

    // STEP 5: Update rotation matrix (Eq 32a)
    //         Ṙ̂ = R̂ [ω]ₓ
    Eigen::Matrix3d omega_skew = Utils::skewMatrixFromVector(omega_total);
    rHat = rHat + rHat * omega_skew * dt;

    // STEP 6: Project back to SO(3)
    orthonormalize();
}
```

### Measurement Correction Computation

```cpp
Eigen::Vector3d ExplicitComplementaryFilter::computeOmegaMeasurement(
    const Eigen::Vector3d& v_measured,   // Measured gravity direction
    const Eigen::Vector3d& v_estimated)  // Estimated gravity direction
    const {

    // Equation (32c): ωₘₑₛ = Σᵢ kᵢ(vᵢ × v̂ᵢ)
    // For single vector (gravity only):
    //     ωₘₑₛ = v × v̂

    return v_measured.cross(v_estimated);
}
```

**Geometric interpretation:**
- `v × v̂` gives axis perpendicular to both vectors
- This is the rotation axis needed to align v̂ with v
- Magnitude: `|v × v̂| = |v||v̂|sin(θ) = sin(θ)` (both normalized)
- When aligned: `v × v̂ = 0` (no correction needed)

---

## Parameter Tuning

### Recommended Starting Values

From paper Section VI (experimental results):

```
kₚ = 1.0 rad/s     (proportional gain)
kᵢ = 0.3 rad/s     (integral gain)
```

### Tuning Guidelines

| Gain | Effect if too LOW | Effect if too HIGH |
|------|-------------------|-------------------|
| k<sub>p</sub> | • Slow correction<br>• More drift<br>• Large transient errors | • Noise amplification<br>• Oscillations<br>• Reduced stability margin |
| k<sub>i</sub> | • Slow bias convergence<br>• Residual drift | • Overshoot<br>• Potential instability<br>• Noise in bias estimate |

### Tuning Procedure

**Step 1: Start with recommended values**
```cpp
kp = 1.0;
ki = 0.3;
```

**Step 2: Adjust k<sub>p</sub> based on noise**
```
If output is noisy:     Decrease kp to 0.5 - 0.7
If output is sluggish:  Increase kp to 1.5 - 2.0
```

**Step 3: Adjust k<sub>i</sub> based on convergence**
```
If bias converges slowly:  Increase ki to 0.5 - 0.8
If bias oscillates:        Decrease ki to 0.1 - 0.2
```

**Step 4: Verify stability**
```
Check that: kp > 0, ki > 0
Typical range: 0.5 ≤ kp ≤ 2.0
               0.1 ≤ ki ≤ 0.8
```

### Rule of Thumb

**Relationship between gains:**
```
ki ≈ 0.3 * kp

This provides good balance between:
- Fast attitude correction (kp)
- Stable bias estimation (ki)
```

**Sampling time dependence:**
```
For faster sampling (dt smaller):
  → Can use larger gains (faster convergence)

For slower sampling (dt larger):
  → Must use smaller gains (avoid instability)

Guideline: kp * dt < 0.1
```

---

## Comparison with Other Filters

### vs Passive Mahony Filter

| Feature | Passive Mahony | Explicit CF | Winner |
|---------|----------------|-------------|--------|
| Attitude reconstruction | Required (R<sub>y</sub>) | Not required | ✅ Explicit |
| Computational cost | Medium | Lower | ✅ Explicit |
| Magnetometer required | Yes (for yaw) | No | ✅ Explicit |
| Bias estimation | All axes | All axes | ✅ Tie |
| Stability | Almost global | Almost global | ✅ Tie |
| Code complexity | Medium | Lower | ✅ Explicit |
| Noise sensitivity | Medium | Lower | ✅ Explicit |

**When to use Explicit instead of Passive:**
- ✅ Magnetometer unavailable or unreliable
- ✅ Embedded system (limited CPU)
- ✅ Prefer simpler code
- ✅ Want to avoid attitude reconstruction errors

### vs Extended Kalman Filter (EKF)

| Feature | EKF | Explicit CF | Winner |
|---------|-----|-------------|--------|
| Optimality | Optimal (if Gaussian) | Suboptimal | ⚖️ EKF (theoretically) |
| Tuning complexity | High (Q, R matrices) | Low (k<sub>p</sub>, k<sub>i</sub>) | ✅ Explicit |
| Computational cost | High (7×7 covariance) | Very low | ✅ Explicit |
| Memory usage | High | Low | ✅ Explicit |
| Robustness | Sensitive to tuning | Robust | ✅ Explicit |
| Convergence | Local | Almost global | ✅ Explicit |
| Uncertainty estimate | Yes (covariance) | No | ✅ EKF |

**When to use Explicit instead of EKF:**
- ✅ Embedded system (limited CPU/memory)
- ✅ Want simple, robust filter
- ✅ Don't need uncertainty quantification
- ✅ Prefer guaranteed global convergence

**When to use EKF instead of Explicit:**
- ✅ Need optimal estimation
- ✅ Need uncertainty bounds
- ✅ Have good noise models
- ✅ Have CPU resources for 7×7 matrix operations

### vs Linear Complementary Filter

| Feature | Linear CF | Explicit CF | Winner |
|---------|-----------|-------------|--------|
| State space | Euclidean (angles) | SO(3) | ✅ Explicit |
| Large angles | Poor (linearization) | Excellent | ✅ Explicit |
| Bias estimation | Limited (1D per axis) | Full 3D | ✅ Explicit |
| Gimbal lock | Yes | No | ✅ Explicit |
| Simplicity | Very simple | Simple | ⚖️ Linear |

---

## Symbol Glossary

### Rotation and Frames

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| {A} | Inertial (world) reference frame | - |
| {B} | Body-fixed reference frame (IMU) | - |
| R | True rotation matrix from {B} to {A} | 3×3 |
| R̂ | Estimated rotation matrix | 3×3 |
| R̃ | Rotation error: R̃ = R̂ᵀR | 3×3 |
| SO(3) | Special Orthogonal Group (3D rotations) | - |

### Measurements and Estimates

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| Ω | True angular velocity (body frame) | 3×1 rad/s |
| Ωʸ | Measured angular velocity (gyro) | 3×1 rad/s |
| b | True gyro bias | 3×1 rad/s |
| b̂ | Estimated gyro bias | 3×1 rad/s |
| b̃ | Bias error: b̃ = b - b̂ | 3×1 rad/s |
| v₀ | Reference vector (inertial frame) | 3×1 |
| v | Measured vector (body frame) | 3×1 |
| v̂ | Estimated vector (body frame) | 3×1 |

### Filter Parameters

| Symbol | Meaning | Typical Value | Units |
|--------|---------|---------------|-------|
| k<sub>p</sub> | Proportional gain | 1.0 | rad/s |
| k<sub>i</sub> | Integral gain | 0.3 | rad/s |
| Δt | Sample time | 0.02 | s |
| ωₘₑₛ | Measurement correction | - | rad/s |

### Mathematical Operators

| Symbol | Meaning | Example |
|--------|---------|---------|
| [·]ₓ | Skew-symmetric matrix | [ω]ₓ v = ω × v |
| vex(·) | Inverse of [·]ₓ | vex([ω]ₓ) = ω |
| ⟨·,·⟩ | Inner product | ⟨v, v̂⟩ = vᵀv̂ |
| × | Cross product | v × v̂ |
| tr(·) | Matrix trace | tr(A) = Σᵢ Aᵢᵢ |
| det(·) | Determinant | det(R) = 1 for R ∈ SO(3) |
| [A,B] | Matrix commutator | [A,B] = AB - BA |

### Special Vectors

| Symbol | Value | Meaning |
|--------|-------|---------|
| e₃ | [0, 0, 1]ᵀ | Unit vector (Z-axis up) |
| g₀ | [0, 0, -9.81]ᵀ m/s² | Gravity (inertial frame) |
| v₀_gravity | [0, 0, 1]ᵀ | Gravity direction (normalized) |

---

## References

### Primary Paper

**Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008)**
"Nonlinear Complementary Filters on the Special Orthogonal Group"
*IEEE Transactions on Automatic Control*, Vol. 53, No. 5, pp. 1203-1218
DOI: 10.1109/TAC.2008.923738

**Key Sections for Explicit CF:**
- Section V: Explicit error formulation (pages 8-11)
- Theorem 5.1: Multiple vector case (page 9)
- Corollary 5.2: Single vector case (page 11) ⭐
- Section VI: Experimental results (pages 12-13)
- Figure 9: Bias estimation validation (page 13)

### Related Documentation

- `MDFiles/MahonyFilter_Mathematical_Documentation.md` - Passive Mahony filter
- `MDFiles/EKF_Complete_Mathematical_Reference.md` - Extended Kalman filter
- `MDFiles/Accelerometer_Explanation.md` - Sensor fundamentals

### Implementation Files

- `include/ExplicitComplementaryFilter.hpp` - Header file
- `src/ExplicitComplementaryFilter.cpp` - Implementation
- `include/Utils.hpp` - Mathematical utilities

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Author:** Based on Mahony et al. (2008) implementation

---

**End of Document**
