#include "MahonyFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const double kp = 50.0; // Increased gain for faster correction
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

    // Initialize Mahony filter
    MahonyFilter mahony(dt, kp);

    std::cout << "Mahony Passive Complementary Filter Test\n\n";

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    std::cout << "Dataset: " << numSamples << " samples\n";
    std::cout << "Time step (dt): " << dt << " seconds\n";
    std::cout << "Proportional gain (kP): " << kp << "\n\n";

    mahony.setIMUData(gyroMeasurements, accelMeasurements);
    mahony.predictForAllData();

    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";

    Utils::printVecToFile(mahony.getRollEstimation(), "Results/predicted_roll_mahony_50.txt");
    Utils::printVecToFile(mahony.getPitchEstimation(), "Results/predicted_pitch_mahony_50.txt");


    return 0;
}