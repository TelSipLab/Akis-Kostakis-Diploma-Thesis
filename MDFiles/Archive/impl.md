# EKF Implementation Architecture & Algorithm

## State Representation (7D Vector)

```
state = [q0, q1, q2, q3, bx, by, bz]
         └─ quaternion ─┘  └─ gyro bias ─┘
```

- **q0, q1, q2, q3**: Quaternion representing orientation (unit norm constraint)
- **bx, by, bz**: Gyroscope bias (systematic error in gyro measurements)

## Data Flow Architecture

### 1. INITIALIZATION (once)
```
├── Read CSV files (gyro.csv, accel.csv, angles.csv)
├── Create EKF with initial quaternion [1,0,0,0] (identity = no rotation)
│   ├── Initialize state vector (7x1)
│   ├── Initialize covariance P (7x7)
│   ├── Set process noise Q (7x7)
│   └── Set measurement noise R (3x3)
└── Set sensor data matrices
```

### 2. PROCESSING LOOP (for each timestep i = 0 to N)

#### PREDICT STEP
```
├── Input: gyro reading [gx, gy, gz] at time i
├── Correct gyro: ω_corrected = ω_raw - bias
├── Propagate quaternion using quaternion kinematics:
│   q̇ = 0.5 * Ω(ω_corrected) * q
│   where Ω is skew-symmetric matrix from angular velocity
├── Propagate state: x̂⁻ = f(x, gyro, dt)
│   ├── Update quaternion using quaternion derivative
│   └── Bias assumed constant (ḃ = 0)
├── Compute State Transition Jacobian F (7x7)
│   ├── ∂q/∂q: Quaternion evolution w.r.t. quaternion
│   ├── ∂q/∂b: Quaternion evolution w.r.t. bias
│   └── ∂b/∂b: Identity (bias is constant)
└── Update covariance: P⁻ = F*P*Fᵀ + Q
```

#### UPDATE STEP
```
├── Input: accel reading [ax, ay, az] at time i
├── Normalize accel (gravity direction measurement)
│   a_norm = a / ||a||
├── Predicted measurement:
│   ẑ = R(q)ᵀ * [0,0,1]
│   (rotate gravity vector from world to body frame)
├── Innovation (measurement residual):
│   y = a_norm - ẑ
├── Compute Measurement Jacobian H (3x7)
│   ├── ∂h/∂q: How measurement changes with quaternion
│   └── ∂h/∂b: Zero (measurement independent of bias)
├── Innovation covariance: S = H*P⁻*Hᵀ + R
├── Kalman gain: K = P⁻*Hᵀ*S⁻¹
├── Update state: x̂ = x̂⁻ + K*y
├── Update covariance: P = (I - K*H)*P⁻
└── Normalize quaternion to maintain unit norm constraint
```

### 3. OUTPUT
```
├── Extract quaternion from state (first 4 elements)
├── Convert quaternion → rotation matrix → Euler angles
│   ├── roll = atan2(R(2,1), R(2,2))
│   ├── pitch = atan2(-R(2,0), sqrt(R(2,1)² + R(2,2)²))
│   └── yaw = atan2(R(1,0), R(0,0))
├── Compare with ground truth angles
└── Calculate error metrics (RMSE/MEA)
```

## Key Algorithm Components

### Quaternion Kinematics (Predict)

The quaternion derivative is:
```
q̇ = 0.5 * Ω(ω) * q

where Ω(ω) = [  0   -ωx  -ωy  -ωz ]
              [ ωx    0    ωz  -ωy ]
              [ ωy  -ωz    0    ωx ]
              [ ωz   ωy  -ωx    0  ]
```

Discrete update (Euler integration):
```
q_new = q_old + dt * q̇
q_new = q_new / ||q_new||  // normalize
```

### Accelerometer Measurement Model (Update)

The accelerometer measures gravity direction in body frame:
```
Expected measurement: h(x) = R(q)ᵀ * g_world
where g_world = [0, 0, 1]ᵀ (gravity pointing up in world frame)
```

The rotation matrix R(q) rotates world frame to body frame.

### Jacobian Matrices

#### State Transition Jacobian F (7x7)
```
F = [ ∂f_q/∂q   ∂f_q/∂b ]  (4x4)  (4x3)
    [ ∂f_b/∂q   ∂f_b/∂b ]  (3x4)  (3x3)

where:
- ∂f_q/∂q: Linearization of quaternion kinematics
- ∂f_q/∂b: How bias affects quaternion propagation
- ∂f_b/∂q: Zero (bias independent of quaternion)
- ∂f_b/∂b: Identity (bias constant model)
```

#### Measurement Jacobian H (3x7)
```
H = [ ∂h/∂q   ∂h/∂b ]  (3x4)  (3x3)

where:
- ∂h/∂q: How measurement changes with quaternion
- ∂h/∂b: Zero (measurement independent of bias)
```

## Noise Matrices

### Process Noise Q (7x7)
```
Q = [ Q_quaternion    0      ]  (4x4)  (4x3)
    [      0       Q_bias    ]  (3x4)  (3x3)
```
- Models uncertainty in quaternion evolution
- Models gyro bias random walk

### Measurement Noise R (3x3)
```
R = σ_accel² * I₃
```
- Models accelerometer sensor noise
- Diagonal matrix (assumes independent noise on each axis)

## Integration with Current Codebase

### Usage Pattern (similar to ComplementaryFilter)

```cpp
// In main.cpp
double dt = 0.02;  // 50 Hz sampling rate
Eigen::Vector4d initialQuaternion(1.0, 0.0, 0.0, 0.0);  // Identity quaternion

ExtendedKalmanFilter ekf(dt, initialQuaternion);
ekf.setGyroData(gyroData.getEigenData());
ekf.setAccelData(accelData.getEigenData());
ekf.processAllData();  // Runs predict-update loop for all timesteps

// Extract results
auto eulerAngles = ekf.getEulerAngles();  // Returns [roll, pitch, yaw]
// Compare with ground truth and calculate RMSE/MEA
```

### Data Format
- **gyro.csv**: Angular velocity [rad/s] (3 columns: gx, gy, gz)
- **accel.csv**: Acceleration [m/s²] (3 columns: ax, ay, az)
- **angles.csv**: Ground truth Euler angles [rad] (3 columns: roll, pitch, yaw)

## Implementation Status

### ✅ Completed
- Constructor with state/covariance initialization
- Quaternion utility functions (normalize, toRotationMatrix, toEuler)
- Data setter functions
- Getter functions for quaternion and Euler angles

### ⏳ To Implement
1. **predict() function**
   - Quaternion propagation with gyro input
   - State transition Jacobian computation
   - Covariance prediction

2. **update() function**
   - Accelerometer measurement model
   - Measurement Jacobian computation
   - Kalman gain calculation
   - State and covariance update

3. **processAllData() function**
   - Loop through all timesteps
   - Call predict() with gyro data
   - Call update() with accel data
   - Store intermediate results

4. **Jacobian computation functions**
   - getStateTransitionJacobian()
   - getMeasurementJacobian()

## Expected Output

After implementation, the EKF should:
1. Estimate roll and pitch angles from IMU data
2. Output results to `Results/` directory
3. Provide comparable or better accuracy than ComplementaryFilter
4. Handle gyroscope bias estimation automatically
