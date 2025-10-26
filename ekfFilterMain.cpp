#include "EKF2.hpp"
#include "csvreader.hpp"
#include "Utils.hpp"

#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const int DISPLAY_SAMPLES = 10;

    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();

    CsvReader accelData("Data/accel.csv");
    accelData.read();
    Eigen::MatrixXd accelMeasurements = accelData.getEigenData();

    CsvReader anglesData("Data/angles.csv");
    anglesData.read();
    Eigen::MatrixXd groundTruthAngles = anglesData.getEigenData();

    Eigen::VectorXd rollGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 0);
    Eigen::VectorXd pitchGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 1);
    Utils::convertToDeg(rollGroundTruth);
    Utils::convertToDeg(pitchGroundTruth);

    Eigen::Vector3d initialAccel = accelMeasurements.row(0).transpose();
    initialAccel(0) = -initialAccel(0);

    EKF2 ekf(dt, initialAccel);

    std::cout << "EKF2 Initialized with Real IMU Data\n\n";

    const int numSamples = static_cast<int>(gyroMeasurements.rows());

    std::cout << "Dataset: " << numSamples << " samples\n";
    std::cout << "Initial Quaternion: " << ekf.getQuaternion().transpose() << "\n";
    std::cout << "Initial Roll (deg): " << ekf.getRoll() * 180.0 / M_PI << "\n";
    std::cout << "Initial Pitch (deg): " << ekf.getPitch() * 180.0 / M_PI << "\n\n";

    Eigen::VectorXd rollEstimated(numSamples);
    Eigen::VectorXd pitchEstimated(numSamples);

    std::cout << "Processing all samples...\n";
    std::cout << "Step | Roll Truth | Roll Est | Pitch Truth | Pitch Est\n";
    std::cout << "-----+------------+----------+-------------+-----------\n";

    for(int i = 0; i < numSamples; i++)
    {
        Eigen::Vector3d gyroReading = gyroMeasurements.row(i).transpose();
        Eigen::Vector3d accelReading = accelMeasurements.row(i).transpose();
        accelReading(0) = -accelReading(0);

        ekf.predict(gyroReading);
        ekf.update(accelReading);

        rollEstimated(i) = ekf.getRoll() * 180.0 / M_PI;
        pitchEstimated(i) = ekf.getPitch() * 180.0 / M_PI;

        if (i < DISPLAY_SAMPLES || i >= numSamples - DISPLAY_SAMPLES) {
            std::cout << std::setw(4) << i+1 << " | ";
            std::cout << std::setw(10) << rollGroundTruth(i) << " | ";
            std::cout << std::setw(8) << rollEstimated(i) << " | ";
            std::cout << std::setw(11) << pitchGroundTruth(i) << " | ";
            std::cout << std::setw(9) << pitchEstimated(i) << "\n";
        }
        else if (i == DISPLAY_SAMPLES) {
            std::cout << "...\n";
        }
    }

    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, pitchEstimated) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, pitchEstimated) << " degrees\n";


    Utils::printVecToFile(rollEstimated, "Results/predicted_roll_ekf.txt");
    // Utils::printVecToFile(rollTruthVector, "Results/expected_roll.txt");

    Utils::printVecToFile(pitchEstimated, "Results/predicted_pitch_ekf.txt");
    // Utils::printVecToFile(pitchTruthVector, "Results/expected_pitch.txt");

    return 0;
}
