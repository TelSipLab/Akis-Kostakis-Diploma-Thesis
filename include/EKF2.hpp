#ifndef EKF2_HPP
#define EKF2_HPP

#include "pch.h"

class EKF2 {
  public:
    // Constructor
    EKF2(double dt, const Eigen::Vector3d& initial_accel);

    // Main processing function
    void processAllData(const Eigen::MatrixXd& gyro_data, const Eigen::MatrixXd& accel_data);

    // Get current state
    Eigen::VectorXd getState() const {
        return x;
    }
    Eigen::Vector4d getQuaternion() const {
        return x.head<4>();
    }
    Eigen::Vector3d getBias() const {
        return x.tail<3>();
    }

    // Get roll and pitch from quaternion
    double getRoll() const;
    double getPitch() const;

    // Prediction step
    void predict(const Eigen::Vector3d& gyro);

    // Update step
    void update(const Eigen::Vector3d& accel);

  private:
    // State: [q0, q1, q2, q3, bx, by, bz]^T (7x1)
    Eigen::VectorXd x;

    // Covariance matrix (7x7)
    Eigen::MatrixXd P;

    // Process noise covariance (7x7)
    Eigen::MatrixXd Q;

    // Measurement noise covariance (3x3)
    Eigen::MatrixXd R;

    // Time step
    double dt;

    // Quaternion utilities
    void normalizeQuaternion();
    Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Vector4d& q) const;

    // Jacobians
    Eigen::MatrixXd computeF(const Eigen::Vector3d& gyro_corrected) const;
    Eigen::MatrixXd computeH() const;
};

#endif // EKF2_HPP
