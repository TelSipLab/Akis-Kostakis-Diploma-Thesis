#include "MahonyFilter.hpp"
#include "Utils.hpp"

MahonyFilter::MahonyFilter(double dt, double kp) :
    rHat(Eigen::Matrix3d::Identity()),
    dt(dt),
    kp(kp) {
}

void MahonyFilter::setIMUData(const Eigen::MatrixXd &gyroData, const Eigen::MatrixXd &accelData) {
    this->gyroData = gyroData;
    this->accelometerData = accelData;

    if(gyroData.rows() != accelData.rows()) {
        std::cout << "Wrong data\n";
        exit(-1);
    }

    int numSamples = gyroData.rows();
    rollEstimation.resize(numSamples);
    pitchEstimation.resize(numSamples);
    rollEstimation.setZero();
    pitchEstimation.setZero();
}

void MahonyFilter::predictForAllData() {
    for(int i = 0; i < gyroData.rows(); i++) {
        Eigen::Vector3d gyroReading = gyroData.row(i).transpose();
        Eigen::Vector3d accelReading = accelometerData.row(i).transpose();
        accelReading(0) = -accelReading(0); // Sign correction

        // Normalize accelerometer reading
        Eigen::Vector3d accelNormalized= accelReading.normalized();

        double rollFromSenor = Utils::calculateRollFromAccelInput(accelNormalized.y(), accelNormalized.z());
        double pitchFromSensor = Utils::calculatePitchFromAccelInput(accelNormalized.x(), accelNormalized.y(), accelNormalized.z());

        // Rotation Matrix
        // TODO Check what Eigen does behind the scenes 
        Eigen::Matrix3d R_y = (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(pitchFromSensor, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(rollFromSenor, Eigen::Vector3d::UnitX()))
                            .toRotationMatrix();

        update(gyroReading, R_y);
        
        // Extract roll and pitch from rotation matrix R̂
        // For ZYX Euler: roll = atan2(R32, R33), pitch = atan2(-R31, sqrt(R32² + R33²))
        
        double rollEst = Utils::calculateRollFromAccelInput(rHat(2, 1), rHat(2, 2)) * 180.0 / M_PI;
        rollEstimation(i) = rollEst;

        double pitchEst = Utils::calculatePitchFromAccelInput(rHat(2, 0), rHat(2, 1), rHat(2, 2)) * 180.0 / M_PI;
        pitchEstimation(i) = pitchEst;
    }
}

const Eigen::VectorXd &MahonyFilter::getRollEstimation() {
    return rollEstimation;
}

const Eigen::VectorXd &MahonyFilter::getPitchEstimation() {
    return pitchEstimation;
}

void MahonyFilter::update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y) {
    // omega_y is the gyro reading (Apparently this is somewhat standard naming)
    // R_y is the rotation matrix calculated based on sensor input

    // Equation 1
    Eigen::Matrix3d rTilda = rHat.transpose() * R_y; // Rotation error matrix

    // Equation 7
    Eigen::Matrix3d Pa_R_tilde = 0.5 * (rTilda - rTilda.transpose()); // Find the skew-symmetric matrix and extract the vector via vex
    Eigen::Vector3d omega_mes = Utils::vexFromSkewMatrix(Pa_R_tilde); // This vector tell us how to much to correct at each axis


    // Equation 10
    // Some weird mathematics how we get these equations
    Eigen::Vector3d omega_total = omega_y + kp * omega_mes; // Complenatary filtering 
    Eigen::Matrix3d Omega_skew = Utils::skewMatrixFromVector(omega_total); // We need the result as Matrix since we multiply by R_Hat which is matrix

    rHat = rHat + rHat * Omega_skew * dt;

    orthonormalize();
}
