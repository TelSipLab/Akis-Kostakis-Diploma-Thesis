#include "ExtendedKalmanFilter.hpp"
#include <cmath>

ExtendedKalmanFilter::ExtendedKalmanFilter(double dt, const Eigen::Vector4d &initialQuaternion) : dt(dt)
{

    // Initialize 7D state vector: [q0, q1, q2, q3, bx, by, bz]
    state = Eigen::VectorXd::Zero(7);

    // Set initial quaternion (normalized)
    Eigen::Vector4d q_norm = normalizeQuaternion(initialQuaternion);
    state.segment<4>(0) = q_norm; // First 4 elements are quaternion

    // Initialize gyro bias to zero
    state.segment<3>(4) = Eigen::Vector3d::Zero(); // Last 3 elements are bias

    // Initialize covariance matrix (7x7)
    covariance = Eigen::MatrixXd::Identity(7, 7);
    covariance.block<4, 4>(0, 0) *= 0.01; // Small quaternion uncertainty
    covariance.block<3, 3>(4, 4) *= 0.1;  // Larger bias uncertainty

    // Initialize process noise (7x7)
    processNoise = Eigen::MatrixXd::Zero(7, 7);
    processNoise.block<4, 4>(0, 0) = 0.001 * Eigen::Matrix4d::Identity();  // Quaternion noise
    processNoise.block<3, 3>(4, 4) = 0.0001 * Eigen::Matrix3d::Identity(); // Bias noise

    // Initialize measurement noise (3x3) - accelerometer noise
    measurementNoise = 0.1 * Eigen::Matrix3d::Identity();
}

Eigen::Vector4d ExtendedKalmanFilter::normalizeQuaternion(const Eigen::Vector4d &q)
{
    double norm = q.norm();
    if(norm < 1e-6)
    {
        // Return identity quaternion if input is too small
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    return q / norm; //
}

Eigen::Matrix3d ExtendedKalmanFilter::quaternionToRotationMatrix(const Eigen::Vector4d &q) const
{
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    Eigen::Matrix3d R;
    R(0, 0) = 1 - 2 * (q2 * q2 + q3 * q3);
    R(0, 1) = 2 * (q1 * q2 - q0 * q3);
    R(0, 2) = 2 * (q1 * q3 + q0 * q2);

    R(1, 0) = 2 * (q1 * q2 + q0 * q3);
    R(1, 1) = 1 - 2 * (q1 * q1 + q3 * q3);
    R(1, 2) = 2 * (q2 * q3 - q0 * q1);

    R(2, 0) = 2 * (q1 * q3 - q0 * q2);
    R(2, 1) = 2 * (q2 * q3 + q0 * q1);
    R(2, 2) = 1 - 2 * (q1 * q1 + q2 * q2);

    return R;
}

Eigen::Vector3d ExtendedKalmanFilter::rotationMatrixToEuler(const Eigen::Matrix3d &R) const
{
    double roll = atan2(R(2, 1), R(2, 2));
    double pitch = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    double yaw = atan2(R(1, 0), R(0, 0));

    return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Vector4d ExtendedKalmanFilter::getQuaternion() const
{
    return state.segment<4>(0);
}

Eigen::Vector3d ExtendedKalmanFilter::getEulerAngles() const
{
    Eigen::Vector4d q = getQuaternion();
    Eigen::Matrix3d R = quaternionToRotationMatrix(q);
    return rotationMatrixToEuler(R);
}

void ExtendedKalmanFilter::setGyroData(const Eigen::MatrixXd &data)
{
    gyroData = data;
}

void ExtendedKalmanFilter::setAccelData(const Eigen::MatrixXd &data)
{
    accelData = data;
}

void ExtendedKalmanFilter::predict(const Eigen::Vector3d &gyroReading)
{
    // Extract current quaternion and bias from state
    Eigen::Vector4d q = state.segment<4>(0);
    Eigen::Vector3d bias = state.segment<3>(4);

    // Bias-corrected angular velocity
    Eigen::Vector3d omega_corrected = gyroReading - bias;

    // Build omega skew-symmetric matrix Ω(ω)
    Eigen::Matrix4d Omega;
    Omega << 0, -omega_corrected(0), -omega_corrected(1), -omega_corrected(2),
        omega_corrected(0), 0, omega_corrected(2), -omega_corrected(1),
        omega_corrected(1), -omega_corrected(2), 0, omega_corrected(0),
        omega_corrected(2), omega_corrected(1), -omega_corrected(0), 0;

    // Quaternion derivative: q̇ = 0.5 * Ω(ω) * q
    Eigen::Vector4d q_dot = 0.5 * Omega * q;

    // Euler integration: q_new = q_old + dt * q̇
    Eigen::Vector4d q_predicted = q + dt * q_dot;

    // Normalize quaternion to maintain unit norm
    q_predicted = normalizeQuaternion(q_predicted);

    // Update state (quaternion changes, bias remains constant)
    state.segment<4>(0) = q_predicted;
    // state.segment<3>(4) unchanged (bias constant model)

    // Compute state transition Jacobian F
    Eigen::MatrixXd F = getStateTransitionJacobian(omega_corrected);

    // Predict covariance: P⁻ = F*P*Fᵀ + Q
    covariance = F * covariance * F.transpose() + processNoise;
}

Eigen::MatrixXd ExtendedKalmanFilter::getStateTransitionJacobian(const Eigen::Vector3d &omega)
{
    // State transition Jacobian F (7x7)
    // F = [ ∂f_q/∂q   ∂f_q/∂b ]
    //     [ ∂f_b/∂q   ∂f_b/∂b ]

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(7, 7);

    // ∂f_q/∂q: Linearization of quaternion kinematics (4x4)
    // From q_new = q + dt * 0.5 * Ω(ω) * q
    // ∂f_q/∂q = I + dt * 0.5 * Ω(ω)
    Eigen::Matrix4d Omega;
    Omega << 0, -omega(0), -omega(1), -omega(2),
        omega(0), 0, omega(2), -omega(1),
        omega(1), -omega(2), 0, omega(0),
        omega(2), omega(1), -omega(0), 0;

    F.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity() + dt * 0.5 * Omega;

    // ∂f_q/∂b: How bias affects quaternion propagation (4x3)
    // Since ω_corrected = ω_raw - b, ∂ω/∂b = -I
    // ∂(Ω*q)/∂ω needs to be computed
    Eigen::Vector4d q = state.segment<4>(0);

    // Derivative of Ω(ω)*q with respect to ω components
    Eigen::Matrix<double, 4, 3> dOmega_q_domega;
    dOmega_q_domega << -q(1), -q(2), -q(3),
                        q(0),  q(3), -q(2),
                       -q(3),  q(0),  q(1),
                        q(2), -q(1),  q(0);

    // Chain rule: ∂f_q/∂b = ∂f_q/∂ω * ∂ω/∂b = -dt * 0.5 * dOmega_q_domega
    F.block<4, 3>(0, 4) = -dt * 0.5 * dOmega_q_domega;

    // ∂f_b/∂q: Zero (bias independent of quaternion) (3x4)
    // Already zero from initialization

    // ∂f_b/∂b: Identity (bias constant model) (3x3)
    F.block<3, 3>(4, 4) = Eigen::Matrix3d::Identity();

    return F;
}

void ExtendedKalmanFilter::update(const Eigen::Vector3d &accelReading)
{
    // Step 1: Correct accelerometer sign (sensor X-axis is inverted)
    // Same issue as in ComplementaryFilter - ax should be negated
    Eigen::Vector3d accelCorrected = accelReading;
    accelCorrected(0) = -accelCorrected(0);  // Negate ax

    // Normalize accelerometer reading (get gravity direction)
    double accelMagnitude = accelCorrected.norm();
    Eigen::Vector3d a_norm = accelCorrected / accelMagnitude;

    // Step 2: Predicted measurement - gravity direction in body frame
    // Get current quaternion
    Eigen::Vector4d q = state.segment<4>(0);

    // Convert quaternion to rotation matrix
    Eigen::Matrix3d R = quaternionToRotationMatrix(q);

    // Gravity in world frame: [0, 0, -1] pointing down (what accelerometer measures at rest)
    // When body is upright, accelerometer measures [0, 0, -g] (pointing down)
    // So we predict: R^T * [0, 0, -1] and compare with normalized accel
    Eigen::Vector3d g_world(0.0, 0.0, -1.0);

    // Rotate gravity from world to body frame: z_predicted = R^T * g_world
    Eigen::Vector3d z_predicted = -R.transpose() * g_world;

    // Step 3: Innovation (measurement residual)
    Eigen::Vector3d innovation = a_norm - z_predicted;

    // Step 4: Compute Measurement Jacobian H (3x7)
    Eigen::MatrixXd H = getMeasurementJacobian();

    // Step 5: Innovation covariance S = H*P*H^T + R
    Eigen::Matrix3d S = H * covariance * H.transpose() + measurementNoise;

    // Step 6: Kalman gain K = P*H^T*S^(-1)
    Eigen::MatrixXd K = covariance * H.transpose() * S.inverse();

    // Step 7: Update state: x = x + K*innovation
    state = state + K * innovation;

    // Step 8: Normalize quaternion to maintain unit norm
    Eigen::Vector4d q_updated = state.segment<4>(0);
    q_updated = normalizeQuaternion(q_updated);
    state.segment<4>(0) = q_updated;

    // Step 9: Update covariance: P = (I - K*H)*P
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
    covariance = (I - K * H) * covariance;
}

Eigen::MatrixXd ExtendedKalmanFilter::getMeasurementJacobian()
{
    // Measurement Jacobian H (3x7)
    // H = [ ∂h/∂q   ∂h/∂b ]
    //     (3x4)    (3x3)

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 7);

    // Get current quaternion
    Eigen::Vector4d q = state.segment<4>(0);
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    // Gravity in world frame (pointing down)
    Eigen::Vector3d g_world(0.0, 0.0, -1.0);

    // ∂h/∂q: Derivative of (R^T * g_world) with respect to quaternion (3x4)
    // This comes from differentiating the rotation matrix w.r.t quaternion components

    // ∂h/∂q0
    H(0, 0) = 2.0 * (q2 * g_world(0) - q1 * g_world(1));
    H(1, 0) = 2.0 * (q3 * g_world(0) - q0 * g_world(1) - 2.0 * q1 * g_world(2));
    H(2, 0) = 2.0 * (-q0 * g_world(0) - q3 * g_world(1));

    // ∂h/∂q1
    H(0, 1) = 2.0 * (q3 * g_world(0) + 2.0 * q1 * g_world(1) + q0 * g_world(2));
    H(1, 1) = 2.0 * (q2 * g_world(0) - q1 * g_world(1));
    H(2, 1) = 2.0 * (q1 * g_world(0) - q2 * g_world(1));

    // ∂h/∂q2
    H(0, 2) = 2.0 * (q0 * g_world(0) + 2.0 * q2 * g_world(1) - q3 * g_world(2));
    H(1, 2) = 2.0 * (q1 * g_world(0) + q2 * g_world(1));
    H(2, 2) = 2.0 * (-q2 * g_world(0) + q1 * g_world(1));

    // ∂h/∂q3
    H(0, 3) = 2.0 * (q1 * g_world(0) - q2 * g_world(2));
    H(1, 3) = 2.0 * (q0 * g_world(0) + 2.0 * q3 * g_world(1) + q1 * g_world(2));
    H(2, 3) = 2.0 * (q3 * g_world(0) - q0 * g_world(1));

    // ∂h/∂b: Zero (3x3) - measurement independent of gyro bias
    // Already zero from initialization

    return -H;
}