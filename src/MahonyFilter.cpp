#include "MahonyFilter.hpp"

void MahonyFilter::setData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData) {
    this->gyroData = gyroData;
    this->accelometerData = accelData;

    if (gyroData.rows() != accelData.rows()) {
        std::cout << "Gyro data and accel data are different\n";
        exit(-1);
    }
}

void MahonyFilter::calculate() {
    int size = gyroData.rows();

    for (int i = 0; i < size; i++) {
        // Step 1: Construct measured rotation from accelerometer
        // Assumes accel ≈ -R^T * gravity_inertial
        // gravity_inertial = [0, 0, -9.81] in inertial frame        
        Eigen::Vector3d accel = accelometerData.row(i).transpose();
        Eigen::Vector3d accel_normalized = accel.normalized();

        // Estimated gravity direction in body frame using current estimate
        Eigen::Vector3d gravity_inertial(0, 0, -1);  // Normalized (points down)
        Eigen::Vector3d gravity_estimated = q_hat.conjugate() * gravity_inertial;
    }
}