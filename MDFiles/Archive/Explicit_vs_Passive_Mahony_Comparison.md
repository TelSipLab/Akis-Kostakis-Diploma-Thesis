# Explicit vs Passive Mahony Filter - Detailed Comparison

## Overview

This document compares the **Passive Mahony Filter** and **Explicit Complementary Filter** implementations, highlighting their key differences in approach, implementation, and performance.

---

## 1. Conceptual Differences

### Passive Mahony Filter (Equation 13 in paper)

**Core Idea:** Uses **reconstructed attitude matrix** from sensors to compute correction

```
Input:  Gyro + Accelerometer → Reconstruct Ry → Compute error R̃ = R̂ᵀRy
        ↓
Output: Corrected attitude estimate R̂
```

**Process Flow:**
1. Measure accelerometer → Calculate roll/pitch angles
2. Build rotation matrix Ry from angles
3. Compute rotation error: R̃ = R̂ᵀRy
4. Extract correction vector: ω = vex(Pa(R̃))
5. Update: R̂˙ = R̂[Ωʸ + kpω]×

### Explicit Complementary Filter (Equation 32 in paper)

**Core Idea:** Uses **vectorial measurements directly** without attitude reconstruction

```
Input:  Gyro + Accelerometer (normalized) → Direct vector comparison
        ↓
Output: Corrected attitude estimate R̂ + Bias estimate b̂
```

**Process Flow:**
1. Measure accelerometer → Normalize to get gravity direction v
2. Estimate gravity direction: v̂ = R̂ᵀv₀
3. Compute correction: ωₘₑₛ = v × v̂
4. Update attitude: R̂˙ = R̂[Ωʸ - b̂ + kpωₘₑₛ]×
5. Update bias: ḃ̂ = -kiωₘₑₛ

---

## 2. Key Mathematical Differences

### Error Representation

| Aspect | Passive Mahony | Explicit CF |
|--------|----------------|-------------|
| **Error Type** | Rotation matrix error | Vectorial error |
| **Error Variable** | R̃ = R̂ᵀRy ∈ SO(3) | v - v̂ ∈ ℝ³ |
| **Correction Term** | ω = vex(Pa(R̃)) | ωₘₑₛ = v × v̂ |
| **Geometric Meaning** | Rotation needed to align R̂ with Ry | Rotation axis to align v̂ with v |

### Filter Equations Side-by-Side

**Passive Mahony (No Bias):**
```
R̃ = R̂ᵀRy                          (rotation error)
ω = vex(Pa(R̃))                    (correction term)
R̂˙ = R̂[Ωʸ + kpω]×                 (attitude update)
```

**Explicit CF (With Bias):**
```
v̂ = R̂ᵀv₀                          (estimated gravity)
ωₘₑₛ = v × v̂                       (correction term)
R̂˙ = R̂[Ωʸ - b̂ + kpωₘₑₛ]×          (attitude update)
ḃ̂ = -kiωₘₑₛ                        (bias update)
```

### Equivalence Relationship

From the paper (Section V, page 9):

When Ry is constructed from a single vector measurement v:
```
Pa(R̂ᵀRy) ≈ (v × v̂)×
```

Therefore:
```
vex(Pa(R̃)) ≈ v × v̂
```

**They compute the same correction**, but Explicit does it without constructing Ry!

---

## 3. Implementation Differences

### Input Processing

**Passive Mahony:**
```cpp
// Step 1: Reconstruct full rotation matrix from accelerometer
Eigen::Vector3d accel = accelReading.normalized();

double roll = atan2(accel.y(), accel.z());
double pitch = atan2(-accel.x(), sqrt(accel.y()² + accel.z()²));

Eigen::Matrix3d Ry = rotationMatrixFromRollPitch(roll, pitch);

// Step 2: Use Ry in filter
update(gyroReading, Ry);
```

**Explicit CF:**
```cpp
// Step 1: Use accelerometer directly as gravity direction
Eigen::Vector3d v_measured = accelReading.normalized();

// Step 2: Use vector directly in filter
update(gyroReading, v_measured);
```

### Correction Computation

**Passive Mahony:**
```cpp
void MahonyFilter::update(const Eigen::Vector3d& omega_y,
                          const Eigen::Matrix3d& R_y) {
    // Compute rotation error
    Eigen::Matrix3d R_tilde = rHat.transpose() * R_y;

    // Extract correction vector
    Eigen::Matrix3d Pa_R_tilde = 0.5 * (R_tilde - R_tilde.transpose());
    Eigen::Vector3d omega_mes = Utils::vexFromSkewMatrix(Pa_R_tilde);

    // Update rotation (no bias)
    Eigen::Vector3d omega_total = omega_y + kp * omega_mes;
    rHat = rHat + rHat * Utils::skewMatrixFromVector(omega_total) * dt;
}
```

**Explicit CF:**
```cpp
void ExplicitComplementaryFilter::update(const Eigen::Vector3d& omega_y,
                                          const Eigen::Vector3d& v_measured) {
    // Compute estimated gravity direction
    Eigen::Vector3d v_estimated = rHat.transpose() * v0_gravity;

    // Compute correction vector (simple cross product!)
    Eigen::Vector3d omega_mes = v_measured.cross(v_estimated);

    // Update bias estimate
    biasEstimate = biasEstimate - ki * omega_mes * dt;

    // Update rotation (with bias correction)
    Eigen::Vector3d omega_total = omega_y - biasEstimate + kp * omega_mes;
    rHat = rHat + rHat * Utils::skewMatrixFromVector(omega_total) * dt;
}
```

---

## 4. Computational Complexity

### Operations Count (per update)

| Operation | Passive Mahony | Explicit CF |
|-----------|----------------|-------------|
| **Trigonometric functions** | 4 (atan2, sqrt, sin, cos) | 0 ✅ |
| **Matrix construction** | 1 (build Ry) | 0 ✅ |
| **Matrix multiplication** | 2 (R̂ᵀRy, Pa operation) | 1 (R̂ᵀv₀) ✅ |
| **Cross product** | 0 | 1 |
| **Bias update** | ❌ No | ✅ Yes |
| **Total complexity** | O(n) higher | O(n) lower ✅ |

**Winner:** Explicit CF is **computationally cheaper**

---

## 5. Advantages & Disadvantages

### Passive Mahony Filter

**✅ Advantages:**
- Simple to understand (uses familiar roll/pitch angles)
- Well-established in literature
- No bias estimation → simpler tuning (only kp)
- Proven stable in practice

**❌ Disadvantages:**
- Requires attitude reconstruction (computational overhead)
- Sensitive to noise in reconstructed Ry
- No gyro bias estimation → drift over time
- Cannot handle partial measurements (needs full Ry)
- Trigonometric operations (slower on embedded systems)

### Explicit Complementary Filter

**✅ Advantages:**
- **No attitude reconstruction needed** (major advantage!)
- Works directly with sensor vectors
- **Gyro bias estimation in all 3 axes** (prevents drift)
- Lower computational cost (no trig functions)
- Can work with partial measurements (accelerometer only)
- Better numerical properties (no angle wrapping issues)
- Cleaner geometric interpretation (cross product)

**❌ Disadvantages:**
- Two parameters to tune (kp and ki)
- Requires understanding of bias dynamics
- Less intuitive for those familiar with Euler angles
- Bias convergence depends on motion (needs excitation)

---

## 6. Performance Comparison (Your Results)

### Error Metrics

| Metric | Passive Mahony (kp=11) | Explicit CF (kp=11, ki=0.05) | Improvement |
|--------|------------------------|------------------------------|-------------|
| **Roll RMSE** | 0.613° | **0.553°** | ✅ **9.8% better** |
| **Pitch RMSE** | 0.755° | **0.751°** | ✅ 0.5% better |
| **Combined RMSE** | 0.684° | **0.652°** | ✅ **4.7% better** |
| **Bias Estimation** | ❌ No | ✅ **Yes (all 3 axes)** | Major advantage |

### Tuning Results

**Passive Mahony:**
- kp = 11.0
- ki = N/A (no bias estimation)
- **1 parameter to tune**

**Explicit CF:**
- kp = 11.0
- ki = 0.05
- **2 parameters to tune** (but better performance)

---

## 7. Practical Considerations

### When to Use Passive Mahony

✅ **Use when:**
- You want simplest possible implementation
- Only one parameter to tune (kp)
- Gyro drift is acceptable for your application
- You have magnetometer available for full attitude
- Computational cost is not critical

### When to Use Explicit CF

✅ **Use when:**
- You want best performance from complementary filters
- Gyro bias estimation is important (long-term drift prevention)
- No magnetometer available (accelerometer only)
- Embedded system with limited CPU (no trig operations)
- You want to avoid attitude reconstruction errors
- You need to handle partial sensor failures gracefully

---

## 8. Code Structure Comparison

### File Organization

**Passive Mahony:**
```
include/MahonyFilter.hpp       - Header
src/MahonyFilter.cpp           - Implementation
mahonyFilterMain.cpp           - Test program

Key methods:
- setIMUData(gyro, accel)
- predictForAllData()
- update(omega_y, R_y)
```

**Explicit CF:**
```
include/ExplicitComplementaryFilter.hpp  - Header
src/ExplicitComplementaryFilter.cpp      - Implementation
explicitComplementaryFilterMain.cpp      - Test program

Key methods:
- setIMUData(gyro, accel)
- predictForAllData()
- update(omega_y, v_measured)
- computeOmegaMeasurement(v, v̂)
- getBiasEstimation()         ← Extra feature!
```

### State Variables

**Passive Mahony:**
```cpp
class MahonyFilter {
private:
    double dt;                      // Sample time
    double kp;                      // Proportional gain
    Eigen::Matrix3d rHat;           // Rotation estimate

    // No bias estimation!
};
```

**Explicit CF:**
```cpp
class ExplicitComplementaryFilter {
private:
    double dt;                      // Sample time
    double kp;                      // Proportional gain
    double ki;                      // Integral gain
    Eigen::Matrix3d rHat;           // Rotation estimate
    Eigen::Vector3d biasEstimate;   // ← Gyro bias estimate!
    Eigen::Vector3d v0_gravity;     // Reference gravity vector
};
```

---

## 9. Stability Analysis

### Lyapunov Function

**Passive Mahony:**
```
V = (1/2) tr(I - R̃)

V̇ = -kp |ω|²  ≤ 0
```
- Stable for kp > 0
- No bias convergence guarantee

**Explicit CF:**
```
V = Σ ki(1 - vᵢᵀv̂ᵢ) + (1/ki)|b̃|²

V̇ = -kp |ωₘₑₛ|²  ≤ 0
```
- Stable for kp, ki > 0
- **Guarantees bias convergence** (Theorem 5.1)

### Convergence Properties

| Property | Passive Mahony | Explicit CF |
|----------|----------------|-------------|
| **Attitude convergence** | ✅ Almost global | ✅ Almost global |
| **Bias convergence** | ❌ N/A | ✅ Almost global* |
| **Equilibrium** | (R̂, Ω̂) → (R, Ω) | (R̂, b̂) → (R, b) |

*Requires asymptotic independence of Ω(t) and v(t) (i.e., motion excitation)

---

## 10. Summary Table

| Feature | Passive Mahony | Explicit CF | Winner |
|---------|----------------|-------------|--------|
| **Computational Cost** | Higher (trig, matrix ops) | Lower (cross product) | ✅ Explicit |
| **Memory Usage** | Lower (no bias state) | Slightly higher | ⚖️ Mahony |
| **Attitude Reconstruction** | Required | Not required | ✅ Explicit |
| **Gyro Bias Estimation** | ❌ No | ✅ Yes (all 3 axes) | ✅ Explicit |
| **Parameters to Tune** | 1 (kp) | 2 (kp, ki) | ⚖️ Mahony |
| **Roll RMSE** | 0.613° | **0.553°** | ✅ Explicit |
| **Pitch RMSE** | 0.755° | **0.751°** | ✅ Explicit |
| **Code Complexity** | Medium | Medium | ⚖️ Tie |
| **Ease of Understanding** | Higher (uses angles) | Lower (uses vectors) | ⚖️ Mahony |
| **Magnetometer Required** | Recommended | Optional | ✅ Explicit |
| **Long-term Drift** | Present (no bias correction) | Minimal (bias estimated) | ✅ Explicit |
| **Partial Sensor Failure** | Cannot handle | Can handle | ✅ Explicit |
| **Embedded Suitability** | Good | **Excellent** | ✅ Explicit |

---

## 11. Conclusion

### For This Project (Your Thesis)

**Passive Mahony:**
- Serves as a good baseline
- Simpler to explain conceptually
- Well-known in literature

**Explicit Complementary Filter:**
- **Recommended for practical use**
- Better performance (0.553° vs 0.613° roll RMSE)
- Gyro bias estimation prevents long-term drift
- More efficient computationally
- Main contribution of Mahony et al. 2008 paper (Section V)

### Recommendation

For your thesis, **emphasize the Explicit CF** as:
1. ✅ More advanced theoretical contribution
2. ✅ Better practical performance
3. ✅ Demonstrates understanding of modern filter design
4. ✅ Shows you can work directly on manifolds (SO(3))
5. ✅ Proves you understand bias estimation theory

The Passive Mahony serves as a good stepping stone to explain the Explicit version.

---

## References

**Primary Source:**
- Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). "Nonlinear Complementary Filters on the Special Orthogonal Group." *IEEE TAC*, 53(5), 1203-1218.
  - Section III: Passive Complementary Filter (Equation 13)
  - Section V: Explicit Complementary Filter (Equation 32)

**Implementation Files:**
- Passive Mahony: `src/MahonyFilter.cpp`, `include/MahonyFilter.hpp`
- Explicit CF: `src/ExplicitComplementaryFilter.cpp`, `include/ExplicitComplementaryFilter.hpp`

---

**Document Version:** 1.0
**Date:** January 2025
