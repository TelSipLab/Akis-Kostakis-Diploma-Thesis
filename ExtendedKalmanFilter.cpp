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