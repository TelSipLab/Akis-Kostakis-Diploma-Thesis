#ifndef EXTENTENDED_KALMAN_FILTER
#define EXTENTENDED_KALMAN_FILTER

#include <Eigen/Dense>

class ExtendedKalmanFilter
{
  public:
    ExtendedKalmanFilter(double dt, const Eigen::Vector4d &initialState);

    void setGyroData(const Eigen::MatrixXd &data);
    void setAccelData(const Eigen::MatrixXd &data);
    void setMagData(const Eigen::MatrixXd &data); // Optional for full 3D orientation

    void predict(const Eigen::Vector3d &gyroReading);
    void update(const Eigen::Vector3d &accelReading);

    Eigen::Vector4d getQuaternion() const;
    Eigen::Vector3d getEulerAngles() const; // Roll, Pitch, Yaw

    void processAllData(); // Process entire dataset like ComplementaryFilter

  private:
    // EKF matrices
    Eigen::Vector4d state;            // Quaternion [q0, q1, q2, q3]
    Eigen::Matrix4d covariance;       // State covariance P
    Eigen::Matrix4d processNoise;     // Q matrix
    Eigen::Matrix3d measurementNoise; // R matrix

    // Data
    Eigen::MatrixXd gyroData;
    Eigen::MatrixXd accelData;

    double dt;

    // Internal functions
    Eigen::Matrix4d getStateTransitionJacobian(const Eigen::Vector3d &gyro);
    Eigen::MatrixXd getMeasurementJacobian();
};

#endif // EXTENTENDED_KALMAN_FILTER