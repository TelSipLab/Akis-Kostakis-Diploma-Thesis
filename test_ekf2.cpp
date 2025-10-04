#include "EKF2.hpp"
#include "csvreader.hpp"
#include "Utils.hpp"
#include <iostream>
#include <iomanip>

int main()
{
    std::cout << std::fixed << std::setprecision(6);

    // Read gyro data
    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    std::cout << "=== Gyro Data Loaded ===\n";
    gyroData.printStats();

    // Read accel data
    CsvReader accelData("Data/accel.csv");
    accelData.read();
    std::cout << "=== Accel Data Loaded ===\n";
    accelData.printStats();

    // Read ground truth angles
    CsvReader anglesData("Data/angles.csv");
    anglesData.read();
    std::cout << "=== Ground Truth Angles Loaded ===\n";
    anglesData.printStats();
    std::cout << "\n";

    // Get matrices
    Eigen::MatrixXd gyroMatrix = gyroData.getEigenData();
    Eigen::MatrixXd accelMatrix = accelData.getEigenData();
    Eigen::MatrixXd anglesMatrix = anglesData.getEigenData();

    // Initialize EKF2 with first accelerometer reading
    double dt = 0.02; // 50 Hz sampling rate
    Eigen::Vector3d firstAccel = accelMatrix.row(0).transpose();
    firstAccel(0) = -firstAccel(0); // Correct sensor X-axis inversion

    EKF2 ekf(dt, firstAccel);

    std::cout << "=== EKF2 Test with Real Data ===\n\n";

    int numSamples = std::min(500, static_cast<int>(gyroMatrix.rows()));

    // Extract ground truth
    Eigen::VectorXd rollTruth = Utils::getVectorFromMatrix(anglesMatrix, 0);
    Eigen::VectorXd pitchTruth = Utils::getVectorFromMatrix(anglesMatrix, 1);
    Utils::convertToDeg(rollTruth);
    Utils::convertToDeg(pitchTruth);

    std::cout << "Number of samples: " << numSamples << "\n";
    std::cout << "Initial Quaternion: " << ekf.getQuaternion().transpose() << "\n";
    std::cout << "Initial Roll (deg): " << ekf.getRoll() * 180.0 / M_PI << "\n";
    std::cout << "Initial Pitch (deg): " << ekf.getPitch() * 180.0 / M_PI << "\n\n";

    // Vectors to store results
    Eigen::VectorXd rollPredicted(numSamples);
    Eigen::VectorXd pitchPredicted(numSamples);

    // Process samples
    std::cout << "Processing " << numSamples << " samples...\n";
    std::cout << "Step | Roll Truth | Roll Est | Pitch Truth | Pitch Est\n";
    std::cout << "-----+------------+----------+-------------+-----------\n";

    for(int i = 0; i < numSamples; i++)
    {
        // Get sensor readings
        Eigen::Vector3d gyroReading = gyroMatrix.row(i).transpose();
        Eigen::Vector3d accelReading = accelMatrix.row(i).transpose();
        accelReading(0) = -accelReading(0); // Correct sensor X-axis inversion

        // Run PREDICT step
        ekf.predict(gyroReading);

        // Run UPDATE step
        ekf.update(accelReading);

        // Get results in degrees
        rollPredicted(i) = ekf.getRoll() * 180.0 / M_PI;
        pitchPredicted(i) = ekf.getPitch() * 180.0 / M_PI;

        // Display first 20 samples
        if (i < 20) {
            std::cout << std::setw(4) << i+1 << " | ";
            std::cout << std::setw(10) << rollTruth(i) << " | ";
            std::cout << std::setw(8) << rollPredicted(i) << " | ";
            std::cout << std::setw(11) << pitchTruth(i) << " | ";
            std::cout << std::setw(9) << pitchPredicted(i) << "\n";
        }
    }

    std::cout << "\n=== RMSE and MEA for " << numSamples << " samples ===\n";
    Eigen::VectorXd rollTruthSubset = rollTruth.head(numSamples);
    Eigen::VectorXd pitchTruthSubset = pitchTruth.head(numSamples);

    std::cout << "RMSE between estimated and truth roll: " << Utils::rmse(rollTruthSubset, rollPredicted) << std::endl;
    std::cout << "MEA between estimated and truth roll: " << Utils::mea(rollTruthSubset, rollPredicted) << std::endl;
    std::cout << "RMSE between estimated and truth pitch: " << Utils::rmse(pitchTruthSubset, pitchPredicted) << std::endl;
    std::cout << "MEA between estimated and truth pitch: " << Utils::mea(pitchTruthSubset, pitchPredicted) << std::endl;

    return 0;
}
