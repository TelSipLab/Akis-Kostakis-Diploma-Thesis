# Extended Kalman Filter (EKF) Equations Reference

## Overview
The Extended Kalman Filter is a non-linear extension of the classical Kalman filter that handles non-linear system dynamics and measurement models by linearizing them around the current state estimate.

## Mathematical Notation

### State Variables
- **x̂(k|k-1)**: State estimate at time k given observations up to time k-1 (prediction)
- **x̂(k|k)**: State estimate at time k given observations up to time k (corrected estimate)
- **x̂(k-1|k-1)**: Previous corrected state estimate at time k-1
- **u(k-1)**: Control input at time k-1
- **z(k)**: Measurement/observation vector at time k

### Covariance Matrices
- **P(k|k-1)**: Predicted error covariance matrix at time k
- **P(k|k)**: Corrected error covariance matrix at time k
- **P(k-1|k-1)**: Previous corrected error covariance matrix
- **Q(k-1)**: Process noise covariance matrix
- **R(k)**: Measurement noise covariance matrix

### Functions and Jacobians
- **f(x, u)**: Non-linear state transition function
- **h(x)**: Non-linear measurement function
- **F(k-1)**: Jacobian matrix of f with respect to x, evaluated at x̂(k-1|k-1)
- **H(k)**: Jacobian matrix of h with respect to x, evaluated at x̂(k|k-1)

### Other Variables
- **K(k)**: Kalman gain matrix at time k
- **I**: Identity matrix
- **T**: Matrix transpose operation
- **⁻¹**: Matrix inverse operation

---

## EKF Algorithm Steps

### Step 1: Prediction Phase

#### 1.1 State Prediction
```
x̂(k|k-1) = f(x̂(k-1|k-1), u(k-1))
```

**Description**: Predicts the state at time k using the non-linear state transition function f() applied to the previous corrected state estimate and control input.

**Physical Meaning**: This step propagates the system state forward in time based on the system dynamics model.

#### 1.2 Error Covariance Prediction
```
P(k|k-1) = F(k-1) × P(k-1|k-1) × F(k-1)ᵀ + Q(k-1)
```

**Description**: Predicts the error covariance matrix by linearizing the state transition function and adding process noise.

**Components**:
- **F(k-1)**: Jacobian matrix representing the linearized system dynamics
- **Q(k-1)**: Accounts for uncertainty in the process model

---

### Step 2: Update Phase

#### 2.1 Kalman Gain Calculation
```
K(k) = P(k|k-1) × H(k)ᵀ × [H(k) × P(k|k-1) × H(k)ᵀ + R(k)]⁻¹
```

**Description**: Computes the optimal weighting factor that balances between predicted state and new measurement.

**Components**:
- **H(k)**: Jacobian of measurement function (how measurements relate to states)
- **R(k)**: Measurement noise covariance (uncertainty in sensors)
- The term in brackets is called the "innovation covariance"

#### 2.2 State Update (Correction)
```
x̂(k|k) = x̂(k|k-1) + K(k) × [z(k) - h(x̂(k|k-1))]
```

**Description**: Corrects the predicted state using the new measurement and Kalman gain.

**Components**:
- **z(k) - h(x̂(k|k-1))**: Innovation or residual (difference between actual and predicted measurements)
- **K(k)**: Determines how much to trust the measurement vs. the prediction

#### 2.3 Error Covariance Update
```
P(k|k) = [I - K(k) × H(k)] × P(k|k-1)
```

**Description**: Updates the error covariance to reflect the reduction in uncertainty after incorporating the measurement.

---

## Jacobian Matrices

### State Transition Jacobian F(k-1)
```
F(k-1) = ∂f/∂x |_{x=x̂(k-1|k-1)}
```

**Description**: Partial derivatives of the state transition function with respect to each state variable, evaluated at the previous state estimate.

### Measurement Jacobian H(k)
```
H(k) = ∂h/∂x |_{x=x̂(k|k-1)}
```

**Description**: Partial derivatives of the measurement function with respect to each state variable, evaluated at the predicted state.

---

## Application to Attitude Estimation

For quaternion-based attitude estimation using IMU data:

### State Vector
```
x = [q₀, q₁, q₂, q₃, ωₓ, ωᵧ, ωᵤ]ᵀ
```
Where:
- **q₀, q₁, q₂, q₃**: Quaternion components representing orientation
- **ωₓ, ωᵧ, ωᵤ**: Angular velocities (gyroscope bias states)

### Process Model f(x, u)
Quaternion kinematics:
```
q̇ = ½ × Ω(ω) × q
```
Where Ω(ω) is the quaternion multiplication matrix.

### Measurement Model h(x)
Relates quaternion to expected accelerometer and magnetometer readings in body frame.

### Key Non-linearities
1. **Quaternion multiplication** in dynamics
2. **Rotation matrix conversions** from quaternion to DCM
3. **Trigonometric functions** in attitude representations

---

## Advantages of EKF over Linear Kalman Filter

1. **Handles non-linear dynamics**: Can work with complex system models
2. **Real-time capability**: Computationally efficient compared to particle filters
3. **Optimal for weakly non-linear systems**: Provides good estimates when linearization is valid

## Limitations

1. **Linearization errors**: Accuracy depends on how well linear approximation holds
2. **Jacobian computation**: Requires analytical or numerical differentiation
3. **Divergence risk**: Can become unstable for highly non-linear systems

---

## Implementation Notes

### Numerical Considerations
- Ensure positive definiteness of covariance matrices
- Use numerically stable matrix operations
- Consider using Joseph form for covariance update to improve numerical stability

### Tuning Parameters
- **Q**: Process noise - represents uncertainty in model
- **R**: Measurement noise - represents sensor uncertainty
- Initial conditions for P₀ and x₀

---

*This reference document provides the mathematical foundation for implementing Extended Kalman Filtering in attitude estimation applications.*