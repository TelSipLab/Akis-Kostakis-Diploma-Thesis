#include "ExtendedKalmanFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const int DISPLAY_SAMPLES = 10;

    // Load data
    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();

    CsvReader accelData("Data/accel.csv");
    accelData.read();
    Eigen::MatrixXd accelMeasurements = accelData.getEigenData();

    CsvReader anglesData("Data/angles.csv");
    anglesData.read();
    Eigen::MatrixXd groundTruthAngles = anglesData.getEigenData();

    // Ground truth (convert from radians to degrees)
    Eigen::VectorXd rollGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 0);
    Eigen::VectorXd pitchGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 1);
    Utils::convertToDeg(rollGroundTruth);
    Utils::convertToDeg(pitchGroundTruth);

    // Initialize EKF with initial accelerometer reading
    Eigen::Vector3d initialAccel = accelMeasurements.row(0).transpose();
    initialAccel(0) = -initialAccel(0);

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    ExtendedKalmanFilter ekf(dt, initialAccel);
    ekf.setIMUData(gyroMeasurements, accelMeasurements);

    std::cout << "Extended Kalman Filter Test\n\n";

    std::cout << "Dataset: " << numSamples << " samples\n";
    std::cout << "Time step (dt): " << dt << " seconds\n";
    std::cout << "Initial Quaternion: " << ekf.getQuaternion().transpose() << "\n\n";

    ekf.predictForAllData();

    // Convert results to degrees
    Utils::convertToDeg(ekf.getRollEstimationNonConst());
    Utils::convertToDeg(ekf.getPitchEstimationNonConst());

    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) in degrees ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, ekf.getRollEstimation()) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, ekf.getRollEstimation()) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, ekf.getPitchEstimation()) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, ekf.getPitchEstimation()) << " degrees\n";

    Utils::printVecToFile(ekf.getRollEstimation(), "Results/Results/EkfRoll.txt");
    Utils::printVecToFile(ekf.getPitchEstimation(), "Results/Results/EkfPitch.txt");

    return 0;
}
