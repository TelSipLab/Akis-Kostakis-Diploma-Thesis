# Accelerometer Deep Dive

## Understanding What Accelerometers Actually Measure

---

## **What Does an Accelerometer ACTUALLY Measure?**

### The Counterintuitive Truth:

**An accelerometer does NOT measure acceleration!**

It measures **specific force** (also called **proper acceleration**).

---

## **Specific Force vs. True Acceleration**

### Thought Experiment 1: Free Fall

Imagine you drop your IMU from a building:

**What's happening:**
- True acceleration: **a = 9.81 m/s²** downward (gravity is accelerating it)
- Accelerometer reading: **a = 0 m/s²** (zero!)

**Why?** In free fall, there's no force acting on the internal sensing element - it's weightless!

---

### Thought Experiment 2: Sitting on a Table

Your IMU is sitting still on a table:

**What's happening:**
- True acceleration: **a = 0 m/s²** (not moving)
- Accelerometer reading: **a = 9.81 m/s²** upward (!)

**Why?** The table is pushing up on the sensor, preventing it from falling.

---

## **The Physics: How Accelerometers Work**

Inside an accelerometer is a tiny **proof mass** (a small weight) attached to springs:

```
        ┌─────────┐
        │  Case   │
    ════╪═  ⬜  ═╪════  ← Springs
        │  Mass  │
        └─────────┘
```

**When the sensor accelerates:**
1. The case moves with the sensor
2. The proof mass **lags behind** (inertia)
3. Springs compress/extend
4. Sensor measures spring displacement → reports as "acceleration"

**What the accelerometer actually measures:**
```
a_measured = Forces acting on the mass / mass
           = (Spring force - Gravity) / mass
```

This is **specific force** = all forces EXCEPT gravity.

---

## **The Key Equation**

```
a_accelerometer = a_true - g
```

Where:
- **a_true**: True kinematic acceleration (d²x/dt²)
- **g**: Gravitational acceleration vector

**Rearranged:**
```
a_true = a_accelerometer + g
```

---

## **Why This is USEFUL for Orientation**

### When the IMU is Stationary (a_true = 0):

```
a_accelerometer = -g
```

**The accelerometer points opposite to gravity!**

If we know:
- **g** in the inertial frame = [0, 0, -9.81]^T (assuming Z-up)
- **a** in the body frame = what the sensor reads

Then we can find the rotation matrix **R** that relates them!

```
a_body = R^T × g_inertial
```

This gives us **roll and pitch** (the orientation of the sensor relative to gravity).

---

## **Extracting Roll and Pitch from Accelerometer**

### Step 1: Read the Accelerometer

```cpp
Eigen::Vector3d accelReading = [aₓ, a_y, a_z]^T;
```

This is the gravity vector expressed in the **body frame**.

### Step 2: Normalize (Remove Magnitude)

```cpp
Eigen::Vector3d accel_norm = accelReading.normalized();
```

We only care about **direction** (not magnitude), because we know gravity is always 9.81 m/s².

### Step 3: Calculate Roll (Rotation Around X-axis)

```cpp
double roll = atan2(a_y, a_z);
```

**Intuition:**
- If sensor is level: a_y ≈ 0, a_z ≈ -9.81 → roll ≈ 0
- If tilted right: a_y > 0 → roll > 0
- If tilted left: a_y < 0 → roll < 0

**Geometric meaning:**
```
     Z
     ↑
     |
     |___→ Y
    /
   ↓
  gravity
```
Roll = angle in the Y-Z plane

### Step 4: Calculate Pitch (Rotation Around Y-axis)

```cpp
double pitch = atan2(-aₓ, sqrt(a_y² + a_z²));
```

**Intuition:**
- If sensor is level: aₓ ≈ 0 → pitch ≈ 0
- If nose up: aₓ < 0 → pitch > 0
- If nose down: aₓ > 0 → pitch < 0

**Geometric meaning:**
```
     Z
     ↑
     |
     |___→ X
    /
   ↓
  gravity
```
Pitch = angle in the X-Z plane

---

## **Why Can't We Measure Yaw?**

**Yaw** = rotation around the **vertical axis** (Z-axis, aligned with gravity)

**Problem:** Gravity is vertical!

Rotating around the vertical axis doesn't change how gravity appears in the sensor:

```
Top view (looking down Z-axis):

  Yaw = 0°          Yaw = 45°         Yaw = 90°
      Y                Y'                Y''
      ↑                ↗                 →
      |              /                   |
  ←───┼───→ X    ←─┼──→ X'          ←───┼
      |           /                      |
                                         X''

Gravity (Z) is into the page for all → Same accelerometer Z reading!
```

**Bottom line:** You need a magnetometer (measures Earth's magnetic field) to get yaw.

---

## **The Problem: Linear Acceleration**

### What Happens During Motion?

If the IMU is accelerating (e.g., in a moving car):

```
a_measured = a_gravity + a_linear
```

The accelerometer **cannot distinguish** between:
- Tilting (gravity direction changes)
- Linear acceleration (car speeds up/turns)

**Example:** Car accelerates forward:
```
a_measured = [a_forward, 0, -9.81]
                  ↑
              Linear acceleration (not gravity!)
```

The filter would **incorrectly** interpret this as a pitch angle!

**Solution:** Use complementary filtering!
- **Gyroscope**: Unaffected by linear acceleration → trust for short-term
- **Accelerometer**: Wrong during motion, but correct on average → trust for long-term drift correction

---

## **In Your Code (mahonyFilterMain.cpp)**

```cpp
// mahonyFilterMain.cpp:53-73

// Read accelerometer
Eigen::Vector3d accelReading = accelMeasurements.row(i).transpose();
accelReading(0) = -accelReading(0);  // Sensor-specific sign correction

// Normalize (only care about direction)
Eigen::Vector3d accel_norm = accelReading.normalized();

// Extract roll and pitch
double roll_meas = atan2(accel_norm.y(), accel_norm.z());
double pitch_meas = atan2(-accel_norm.x(),
    sqrt(accel_norm.y() * accel_norm.y() +
         accel_norm.z() * accel_norm.z()));

// Build rotation matrix R_y from accelerometer
Eigen::Matrix3d R_y =
    (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *      // yaw = 0 (unknown)
     Eigen::AngleAxisd(pitch_meas, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(roll_meas, Eigen::Vector3d::UnitX())).toRotationMatrix();

// Feed to Mahony filter
mahony.update(gyroReading, R_y);
```

---

## **Summary Table**

| Scenario | True Accel | Accel Reading | Interpretation |
|----------|------------|---------------|----------------|
| **Stationary on table** | 0 m/s² | +9.81 m/s² up | Table pushing up against gravity |
| **Free fall** | -9.81 m/s² down | 0 m/s² | Weightless! |
| **Elevator accelerating up** | +2 m/s² up | +11.81 m/s² up | Feels heavier |
| **Car braking** | -5 m/s² forward | -5 m/s² forward + 9.81 m/s² up | False pitch angle! |
| **Tilted 45° (stationary)** | 0 m/s² | Components of gravity | True orientation |

---

## **Comparison: Gyroscope vs Accelerometer**

| Property | Gyroscope | Accelerometer |
|----------|-----------|---------------|
| **Measures** | Angular velocity (ω) | Specific force (a) |
| **Units** | rad/s | m/s² |
| **Provides** | Rate of rotation | Gravity direction |
| **Frequency** | High-frequency, smooth | Low-frequency (noisy) |
| **Drift** | Yes (integrates to angle) | No |
| **Absolute Reference** | No | Yes (gravity) |
| **Can measure** | All 3 angles (rate) | Roll & Pitch only |
| **Affected by linear motion** | No | Yes (major problem) |

---

## **Key Takeaways**

1. **Accelerometers measure specific force**, not true acceleration
2. When stationary, they measure **-g** (opposite to gravity)
3. This gives us **roll and pitch** (but not yaw)
4. **Linear acceleration corrupts** the measurement (looks like tilt)
5. This is why we need **sensor fusion** with gyroscopes

---

## **Why Sensor Fusion?**

**Problem with gyro alone:**
```
θ(t) = θ(0) + ∫ω(t)dt
       ↑
   Initial angle unknown + integration drift → unusable after ~10 seconds
```

**Problem with accel alone:**
- Noisy
- Wrong during motion (linear acceleration ≠ 0)
- No yaw information

**Solution: Complementary Filtering (Mahony, EKF, etc.)**
- Use **gyro** for high-frequency changes (smooth, responsive)
- Use **accel** for low-frequency correction (prevent drift)

```
Estimated angle = α × (gyro integration) + (1-α) × (accel measurement)
                  ↑ high-frequency        ↑ low-frequency
```

---

**End of Document**
