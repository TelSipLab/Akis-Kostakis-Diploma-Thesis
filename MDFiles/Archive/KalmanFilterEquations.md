# This is the equations for kalman filtering (not extended).

Can you create an MD file description what each means/does and explain each parameter

Here is a detailed Markdown (MD) description of the standard Kalman filter equations from the provided image. This will outline the meaning and function of each step, as well as an explanation of every parameter involved.

***

# Kalman Filter: Algorithm and Parameters

The Kalman filter is a recursive algorithm for estimating the evolving state of a linear dynamic system from indirect and noisy measurements. It consists of two main steps: **Time Update (Predict)** and **Measurement Update (Correct)**.

## Time Update (“Predict”)

The **predict** step uses the system’s dynamics to forecast the next state estimate and its uncertainty.

### 1. Predict State

$$
\hat{x}^-_k = A \hat{x}_{k-1} + B u_{k-1}
$$

- **$\hat{x}^-_k$**: Predicted (a priori) state estimate at time $k$. This is what the model predicts before considering the new measurement.
- **$A$**: State transition matrix. Describes how the state evolves from one step to the next in the absence of noise and control input.
- **$\hat{x}_{k-1}$**: Previous (a posteriori) state estimate, after the last measurement update.
- **$B$**: Control input matrix. How the external control input affects the state.
- **$u_{k-1}$**: Control input applied in the previous step.


### 2. Predict Error Covariance

$$
P^-_k = A P_{k-1} A^T + Q
$$

- **$P^-_k$**: Predicted (a priori) error covariance. Quantifies the predicted uncertainty in the state estimate.
- **$P_{k-1}$**: Previous (a posteriori) error covariance, after the last measurement update.
- **$Q$**: Process noise covariance matrix. Models the uncertainty in the process, i.e., the randomness in system evolution.

***

## Measurement Update (“Correct”)

The **correct** step incorporates the new measurement to improve the prior state prediction.

### 1. Kalman Gain

$$
K_k = P^-_k H^T (H P^-_k H^T + R)^{-1}
$$

- **$K_k$**: Kalman gain. Determines the weight given to the measurement versus the prediction.
- **$H$**: Measurement matrix. Maps the true state space into the observed space.
- **$R$**: Measurement noise covariance matrix. Models the uncertainty in the sensor measurement.


### 2. Update State Estimate

$$
\hat{x}_k = \hat{x}^-_k + K_k (z_k - H\hat{x}^-_k)
$$

- **$\hat{x}_k$**: Updated (a posteriori) state estimate after measurement $z_k$.
- **$z_k$**: Measurement at step $k$.
- **$(z_k - H\hat{x}^-_k)$**: Measurement residual (innovation). The difference between actual measurement and predicted measurement.


### 3. Update Error Covariance

$$
P_k = (I - K_k H) P^-_k
$$

- **$P_k$**: Updated (a posteriori) error covariance.
- **$I$**: Identity matrix (of appropriate dimensions).

***

## Summary of Parameters

| Symbol | Description |
| :-- | :-- |
| \$ \hat{x}_k \$ | Estimated system state at step $k$ |
| $P_k$ | Error covariance of estimate at step $k$ |
| $A$ | State transition matrix |
| $B$ | Control input matrix |
| $u_k$ | Control input at step $k$ |
| $Q$ | Process noise covariance matrix |
| $z_k$ | Measurement at step $k$ |
| $H$ | Measurement matrix |
| $R$ | Measurement noise covariance matrix |
| $K_k$ | Kalman gain |
| $I$ | Identity matrix |


***

### Short Description of the Process

- The **predict** step advances the previous estimate using the system's model and process dynamics.
- The **correct** step blends the model prediction with sensor measurement, providing an optimally weighted estimate and uncertainty assessment at each step.
