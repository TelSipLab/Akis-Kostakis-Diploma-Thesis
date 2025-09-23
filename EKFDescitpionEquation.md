# Extended Kalman Filter: Algorithm and Parameters

The Extended Kalman Filter (EKF) is the nonlinear version of the Kalman filter, adapted for systems with nonlinear process and measurement models. The EKF linearizes the dynamics and measurement equations at each time step.

## Time Update (“Predict”)

### 1. Predict State

$$
\hat{x}^-_k = f(\hat{x}_{k-1}, u_{k-1}, 0)
$$

- **$\hat{x}^-_k$**: Predicted (a priori) state estimate at time $k$, using the nonlinear process model.
- **$f(\cdot)$**: Nonlinear state transition function. Calculates the next state based on the previous estimate, control input, and (typically) zero process noise for prediction.
- **$\hat{x}_{k-1}$**: Previous (a posteriori) state estimate, after the last measurement update.
- **$u_{k-1}$**: Control input at the previous time step.


### 2. Predict Error Covariance

$$
P^-_k = A_k P_{k-1} A_k^T + W_k Q_{k-1} W_k^T
$$

- **$P^-_k$**: Predicted (a priori) error covariance, quantifies uncertainty in the predicted state.
- **$A_k$**: Jacobian of the state transition function $f$ w.r.t. the state, evaluated at $(\hat{x}_{k-1}, u_{k-1}, 0)$.
- **$P_{k-1}$**: Previous error covariance.
- **$W_k$**: Jacobian of the process function $f$ w.r.t. the process noise, evaluated at $(\hat{x}_{k-1}, u_{k-1}, 0)$.
- **$Q_{k-1}$**: Process noise covariance matrix at the previous step.

***

## Measurement Update (“Correct”)

### 1. Compute Kalman Gain

$$
K_k = P^-_k H_k^T \left( H_k P^-_k H_k^T + V_k R_k V_k^T \right)^{-1}
$$

- **$K_k$**: Kalman gain, reflects the weight given to measurement versus prediction for the update.
- **$H_k$**: Jacobian of the measurement function $h$ w.r.t. the state, evaluated at $(\hat{x}^-_k, 0)$.
- **$V_k$**: Jacobian of the measurement function $h$ w.r.t. the measurement noise, evaluated at $(\hat{x}^-_k, 0)$.
- **$R_k$**: Measurement noise covariance at step $k$.


### 2. Update State Estimate

$$
\hat{x}_k = \hat{x}^-_k + K_k \left(z_k - h(\hat{x}^-_k, 0)\right)
$$

- **$\hat{x}_k$**: Updated (a posteriori) state estimate after incorporating measurement $z_k$.
- **$z_k$**: Measurement received at time step $k$.
- **$h(\cdot)$**: Nonlinear measurement function, maps the predicted state to measurement space.


### 3. Update Error Covariance

$$
P_k = (I - K_k H_k) P^-_k
$$

- **$P_k$**: Updated (a posteriori) error covariance.
- **$I$**: Identity matrix.

***

## EKF Parameters Table

| Symbol | Description |
| :-- | :-- |
| \$ \hat{x}_k \$ | Estimated system state at step $k$ |
| $P_k$ | Error covariance of estimate at step $k$ |
| $f$ | Nonlinear state transition function |
| $A_k$ | Jacobian of $f$ w.r.t. state; evaluates system dynamics sensitivity |
| $W_k$ | Jacobian of $f$ w.r.t. process noise; process noise sensitivity |
| $Q_{k-1}$ | Process noise covariance matrix for previous step |
| $u_k$ | Control input at step $k$ |
| $z_k$ | Measurement at step $k$ |
| $h$ | Nonlinear measurement function |
| $H_k$ | Jacobian of $h$ w.r.t. state; measurement model sensitivity |
| $V_k$ | Jacobian of $h$ w.r.t. measurement noise; sensitivity |
| $R_k$ | Measurement noise covariance matrix |
| $K_k$ | Kalman gain |
| $I$ | Identity matrix |


***

### Common Simplification

In many practical implementations (including quaternion EKF), $W_k = I$ and $V_k = I$, simplifying to:

$$
P^-_k = A_k P_{k-1} A_k^T + Q
$$

$$
K_k = P^-_k H_k^T \left( H_k P^-_k H_k^T + R \right)^{-1}
$$

***

### EKF Process Description

- **Prediction step** uses the system's nonlinear model to forecast the next state and its uncertainty, employing appropriate linearizations via the Jacobian matrices.
- **Update step** incorporates the latest measurement by linearizing the measurement function and optimally combining it with the prediction using the Kalman gain.