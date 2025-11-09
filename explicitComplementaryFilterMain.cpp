#include "ExplicitComplementaryFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Filter parameters (recommended values from Mahony et al. 2008, Section VI)
    const double dt = 0.02;      // Sample time (50 Hz)
    const double kp = 5.0;       // Proportional gain (testing higher value)
    const double ki = 0.5;       // Integral gain

    std::cout << "Explicit Complementary Filter Test\n";
    std::cout << "===================================\n\n";

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

    const int numSamples = static_cast<int>(gyroMeasurements.rows());

    // Display configuration
    std::cout << "Dataset: " << numSamples << " samples\n";
    std::cout << "Time step (dt): " << dt << " seconds\n";
    std::cout << "Proportional gain (kp): " << kp << " rad/s\n";
    std::cout << "Integral gain (ki): " << ki << " rad/s\n\n";

    // Create and run filter
    ExplicitComplementaryFilter explicitCF(dt, kp, ki);
    explicitCF.setIMUData(gyroMeasurements, accelMeasurements);

    std::cout << "Running filter...\n";
    explicitCF.predictForAllData();
    std::cout << "Filter complete!\n\n";

    // Convert results to degrees
    Utils::convertToDeg(explicitCF.getRollEstimationNonConst());
    Utils::convertToDeg(explicitCF.getPitchEstimationNonConst());

    // Calculate and display error metrics
    double rollRMSE = Utils::rmse(rollGroundTruth, explicitCF.getRollEstimation());
    double rollMEA = Utils::mea(rollGroundTruth, explicitCF.getRollEstimation());
    double pitchRMSE = Utils::rmse(pitchGroundTruth, explicitCF.getPitchEstimation());
    double pitchMEA = Utils::mea(pitchGroundTruth, explicitCF.getPitchEstimation());

    std::cout << "=== Error Metrics (all " << numSamples << " samples) ===\n";
    std::cout << "Roll RMSE:  " << rollRMSE << " degrees\n";
    std::cout << "Roll MEA:   " << rollMEA << " degrees\n";
    std::cout << "Pitch RMSE: " << pitchRMSE << " degrees\n";
    std::cout << "Pitch MEA:  " << pitchMEA << " degrees\n";
    std::cout << "Combined RMSE: " << (rollRMSE + pitchRMSE) / 2.0 << " degrees\n\n";

    // Display estimated gyro bias
    Eigen::Vector3d bias = explicitCF.getBiasEstimation();
    std::cout << "=== Estimated Gyro Bias ===\n";
    std::cout << "Bias X: " << bias(0) << " rad/s\n";
    std::cout << "Bias Y: " << bias(1) << " rad/s\n";
    std::cout << "Bias Z: " << bias(2) << " rad/s\n";

    return 0;
}
