#include "MahonyFilter.hpp"

MahonyFilter::MahonyFilter(double dt, double kp) :
    rHat(Eigen::Matrix3d::Identity()),
    dt(dt),
    kp(kp) {

}

void MahonyFilter::setData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData) {
    this->gyroData = gyroData;
    this->accelometerData = accelData;

    if (gyroData.rows() != accelData.rows()) {
        std::cout << "Gyro data and accel data are different\n";
        exit(-1);
    }
}

void MahonyFilter::update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y) {

    // Equation 1
    Eigen::Matrix3d rTilda = rHat.transpose() * R_y;

    // Equation 7
    Eigen::Matrix3d Pa_R_tilde = 0.5 * (rTilda - rTilda.transpose());
    Eigen::Vector3d omega_mes = vex(Pa_R_tilde);

    // Equation 10
    Eigen::Vector3d omega_total = omega_y + kp * omega_mes;
    Eigen::Matrix3d Omega_skew = skew(omega_total);

    rHat = rHat + rHat * Omega_skew * dt;

    orthonormalize();
}

// Does Equation 10 from the papper
void MahonyFilter::calculate() {
    // int size = gyroData.rows();

    // for (int i = 0; i < size; i++) {

    //     rTilda = rHat.transpose() * 
    // }
}