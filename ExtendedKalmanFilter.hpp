#ifndef EXTENTENDED_KALMAN_FILTER
#define EXTENTENDED_KALMAN_FILTER

#include <Eigen/Dense>

class ExtendedKalmanFilter
{
  public:
    ExtendedKalmanFilter(double dt, const Eigen::Vector4d &initialQuaternion);

    void setGyroData(const Eigen::MatrixXd &data);
    void setAccelData(const Eigen::MatrixXd &data);
    void setMagData(const Eigen::MatrixXd &data); // Optional for full 3D orientation

    void predict(const Eigen::Vector3d &gyroReading);
    void update(const Eigen::Vector3d &accelReading);

    Eigen::Vector4d getQuaternion() const;
    Eigen::Vector3d getEulerAngles() const; // Roll, Pitch, Yaw

    void processAllData(); // Process entire dataset like ComplementaryFilter

  private:
    // EKF matrices - 7D state: [q0, q1, q2, q3, bx, by, bz]
    Eigen::VectorXd state;            // State vector: quaternion + gyro bias
    Eigen::MatrixXd covariance;       // State covariance P (7x7)
    Eigen::MatrixXd processNoise;     // Q matrix (7x7)
    Eigen::Matrix3d measurementNoise; // R matrix (3x3)

    // Data
    Eigen::MatrixXd gyroData;
    Eigen::MatrixXd accelData;

    double dt;

    // Internal functions
    Eigen::MatrixXd getStateTransitionJacobian(const Eigen::Vector3d &gyro);
    Eigen::MatrixXd getMeasurementJacobian();

    // Quaternion utility functions
    Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d &q);
    Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Vector4d &q) const;
    Eigen::Vector3d rotationMatrixToEuler(const Eigen::Matrix3d &R) const;
};

#endif // EXTENTENDED_KALMAN_FILTER