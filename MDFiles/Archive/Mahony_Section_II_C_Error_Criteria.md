# Mahony Filter - Section II-C: Error Criteria for Estimation on SO(3)

## How Do We Measure Rotation Error?

This section answers a fundamental question: **"How do we measure how wrong our estimate R̂ is compared to the true rotation R?"**

---

## **The Problem: You Can't Just Subtract Rotation Matrices**

With regular vectors, error is simple:
```
error = x̂ - x  ← Easy!
```

But with rotations on SO(3), you **cannot** do:
```
error = R̂ - R  ← WRONG! This doesn't respect SO(3) structure
```

**Why not?**
- R̂ - R is NOT a rotation matrix
- Doesn't respect the geometry of SO(3) (it's a manifold, not a vector space)
- Doesn't handle the circular nature of rotations (e.g., 0° = 360°)

---

## **The Solution: Relative Rotation Error**

Instead, we define the **rotation error matrix**:

$$\tilde{R} = \hat{R}^T R$$

Or sometimes (equivalent):

$$\tilde{R} = R^T \hat{R}$$

**Physical meaning:** R̃ represents the rotation needed to go from R̂ to R.

**Notation:**
- R̃ (R-tilde): Rotation error matrix
- R̂ (R-hat): Estimated rotation matrix
- R: True rotation matrix

---

## **Interpretation: "How do I correct my estimate?"**

Think of it this way:
- **R̂**: My current estimate (where I think I am)
- **R**: True orientation (where I actually am)
- **R̃**: The rotation I need to apply to R̂ to get to R

$$R = \hat{R} \cdot \tilde{R}$$

**Perfect estimate:** If R̂ = R, then R̃ = I (identity matrix)

**Small error:** If R̂ is close to R, then R̃ is close to I

---

## **Example: Numerical Understanding**

Let's say:
- True orientation: R = 45° rotation around Z-axis
- Estimated orientation: R̂ = 40° rotation around Z-axis
- Error: 5° off

Then:

$$\tilde{R} = \hat{R}^T R = (40°)^T \times (45°) = 5° \text{ rotation around Z}$$

R̃ tells us: "You need to rotate 5° more around Z to be correct."

---

## **Error Metrics on SO(3)**

Now that we have R̃, how do we quantify "how far from identity" it is?

### **Metric 1: Trace-Based Error**

$$\text{error}_{\text{trace}} = \text{trace}(I - \tilde{R})$$

**Properties:**
- If R̃ = I (perfect): error = trace(I - I) = 0
- If R̃ is far from I: error is large
- Range: [0, 4] (since trace of 3×3 matrix is at most 3, and I has trace 3)

**Equivalent form:**

$$\text{error}_{\text{trace}} = 3 - \text{trace}(\tilde{R})$$

### **Metric 2: Rotation Angle Error**

Every rotation matrix can be represented as rotation by angle θ around some axis.

$$\text{trace}(\tilde{R}) = 1 + 2\cos(\theta)$$

So the rotation angle is:

$$\theta = \arccos\left(\frac{\text{trace}(\tilde{R}) - 1}{2}\right)$$

This θ is the **angle of the error rotation**.

**Physical meaning:** "My estimate is off by θ degrees around some axis."

---

## **The Skew-Symmetric Projection $P_a$**

This is the **key tool** used in the Mahony filter!

### **Definition:**

For any matrix M, the skew-symmetric projection is:

$$P_a(M) = \frac{1}{2}(M - M^T)$$

**What it does:** Extracts the skew-symmetric (anti-symmetric) part of M.

---

### **Why is this useful for error?**

If R̃ is close to identity, we can write:

$$\tilde{R} \approx I + \epsilon \Omega$$

Where:
- ε is small
- Ω is a skew-symmetric matrix

**Computing $P_a(\tilde{R})$:**

$$P_a(\tilde{R}) = \frac{1}{2}(\tilde{R} - \tilde{R}^T)$$

If R̃ ∈ SO(3) (perfect rotation), then $\tilde{R}^T \tilde{R} = I$, so:

$$\tilde{R}^T = \tilde{R}^{-1}$$

But if there's error, $\tilde{R}^T \neq \tilde{R}^{-1}$, and $P_a(\tilde{R}) \neq 0$!

---

### **The Magic: Extracting the Error Vector**

Apply the **vex** operator to get a 3D error vector:

$$\omega_{\text{error}} = \text{vex}(P_a(\tilde{R}))$$

**This vector represents the instantaneous angular velocity correction needed!**

---

## **In the Mahony Filter (Equation 7)**

This is exactly what the filter does:

```cpp
// MahonyFilter.cpp:23-27

// Step 1: Compute rotation error
Eigen::Matrix3d rTilda = rHat.transpose() * R_y;
                      // R̃ = R̂^T R_y

// Step 2: Extract correction via projection
Eigen::Matrix3d Pa_R_tilde = 0.5 * (rTilda - rTilda.transpose());
                          // P_a(R̃)

Eigen::Vector3d omega_mes = vex(Pa_R_tilde);
                          // ω_mes = vex(P_a(R̃))
```

**ω_mes is the measurement-derived correction term!**

---

## **Why $P_a$ Instead of Just $(R̃ - I)$?**

You might ask: "Why not just use R̃ - I as the error?"

**Answer:** Because we need the error in the **Lie algebra so(3)**, not in matrix space.

- **R̃ - I**: Just a matrix (not skew-symmetric, not in so(3))
- **$P_a(\tilde{R})$**: A skew-symmetric matrix (in so(3))
- **$\text{vex}(P_a(\tilde{R}))$**: A vector in ℝ³ representing angular velocity

The projection $P_a$ ensures we get a **valid angular velocity correction**.

---

## **Complete Error Flow Diagram**

```
True rotation (R)              Estimated rotation (R̂)
      ↓                                 ↓
      └─────────→ R̃ = R̂^T R ←─────────┘
                      ↓
              P_a(R̃) = ½(R̃ - R̃^T)  (skew-symmetric matrix)
                      ↓
              ω_error = vex(P_a(R̃))  (3D vector)
                      ↓
              Apply correction with gain k_p
                      ↓
              ω_total = ω_gyro + k_p · ω_error
```

---

## **Mathematical Properties**

### **Property 1: Zero Error ⟺ Identity**

If R̂ = R (perfect estimate):

$$\tilde{R} = \hat{R}^T R = R^T R = I$$

$$P_a(I) = \frac{1}{2}(I - I^T) = \frac{1}{2}(I - I) = 0$$

$$\text{vex}(0) = 0$$

No correction needed! ✓

### **Property 2: Small Angle Approximation**

For small rotation errors (angle θ ≪ 1):

$$\tilde{R} \approx I + \theta \cdot \Omega \quad \text{(where } \Omega \text{ is skew-symmetric)}$$

$$P_a(\tilde{R}) \approx \frac{1}{2}\theta \cdot \Omega$$

$$\text{vex}(P_a(\tilde{R})) \approx \frac{1}{2}\theta \cdot \omega$$

The correction is proportional to the error angle!

### **Property 3: SO(3) Invariance**

The error metric works regardless of:
- Reference frame orientation
- Order of rotations
- Sign conventions

This is because it respects the group structure of SO(3).

---

## **Comparison of Error Metrics**

| Method | Formula | Units | Properties |
|--------|---------|-------|------------|
| **Naive subtraction** | R̂ - R | Matrix | ✗ Not in SO(3), meaningless |
| **Relative rotation** | $\tilde{R} = \hat{R}^T R$ | Matrix | ✓ In SO(3), but still a matrix |
| **Trace error** | $3 - \text{trace}(\tilde{R})$ | Scalar | ✓ Single number, hard to correct |
| **Rotation angle** | $\arccos((\text{trace}(\tilde{R})-1)/2)$ | Radians | ✓ Physical meaning, no axis info |
| **$P_a$ projection** | $\text{vex}(P_a(\tilde{R}))$ | rad/s | ✓ Vector, directly usable for correction |

**Winner for control:** $P_a$ projection → gives us a correction vector!

---

## **Geometric Intuition**

Think of SO(3) as a curved 3D manifold (like the surface of a sphere, but more complex).

- **Vector subtraction (R̂ - R)**: Tries to draw a straight line in curved space → nonsense
- **$P_a(\hat{R}^T R)$**: Finds the tangent vector at R̂ pointing toward R → makes sense!

```
      SO(3) manifold (curved)
         ___
       /     \
      |   R   |  ← True rotation
      |   ↑   |
      |   |   |  ← Tangent vector (error direction)
      |   R̂   |  ← Estimated rotation
       \_____/
```

$P_a$ gives us the tangent vector in the Lie algebra that points from R̂ toward R.

---

## **Detailed Mathematical Breakdown**

### Why is $P_a(\tilde{R})$ the "error"?

For a **perfect rotation matrix** R̃ ∈ SO(3), we have:

$$\tilde{R}^T \tilde{R} = I \quad \Rightarrow \quad \tilde{R}^T = \tilde{R}^{-1}$$

This means:

$$\tilde{R} - \tilde{R}^T = \tilde{R} - \tilde{R}^{-1}$$

For small deviations from identity ($\tilde{R} = I + \epsilon \Omega + O(\epsilon^2)$):

$$\tilde{R}^{-1} = I - \epsilon \Omega + O(\epsilon^2)$$

So:

$$\tilde{R} - \tilde{R}^T = (I + \epsilon \Omega) - (I - \epsilon \Omega) = 2\epsilon \Omega$$

Therefore:

$$P_a(\tilde{R}) = \frac{1}{2}(\tilde{R} - \tilde{R}^T) = \epsilon \Omega$$

**This is exactly the skew-symmetric error matrix!**

---

## **Summary: Section II-C Key Points**

1. **Error on SO(3)** is defined as $\tilde{R} = \hat{R}^T R$ (relative rotation)
2. **Perfect estimate** means R̃ = I
3. **$P_a$ projection** extracts the skew-symmetric part: $P_a(\tilde{R}) = \frac{1}{2}(\tilde{R} - \tilde{R}^T)$
4. **vex operator** converts to 3D vector: $\omega_{\text{error}} = \text{vex}(P_a(\tilde{R}))$
5. This $\omega_{\text{error}}$ is the **correction term** used in the Mahony filter

**The Mahony filter's genius:** It uses this mathematically rigorous error metric to drive the correction!

---

## **Connection to Mahony Filter Equations**

### Equation (1) in the paper:

$$\tilde{R} = \hat{R}^T R_y$$

Where:
- R̂: Current rotation estimate
- $R_y$: Rotation from measurement (accelerometer)
- R̃: Error between them

### Equation (7) in the paper:

$$\omega_{\text{mes}} = \text{vex}(P_a(\tilde{R}))$$

This extracts the correction angular velocity from the error matrix.

### Equation (10) in the paper:

$$\omega_{\text{total}} = \omega_y + k_p \omega_{\text{mes}}$$

Combines gyroscope measurement with the correction term.

---

**End of Document**
