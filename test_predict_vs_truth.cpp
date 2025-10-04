#include "ExtendedKalmanFilter.hpp"
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

    // Read ground truth angles
    CsvReader anglesData("Data/angles.csv");
    anglesData.read();

    std::cout << "=== Data Loaded ===\n";
    gyroData.printStats();
    anglesData.printStats();
    std::cout << "\n";

    // Initialize EKF
    double dt = 0.02; // 50 Hz sampling rate
    Eigen::Vector4d initialQuaternion(1.0, 0.0, 0.0, 0.0); // Identity quaternion

    ExtendedKalmanFilter ekf(dt, initialQuaternion);

    // Get matrices
    Eigen::MatrixXd gyroMatrix = gyroData.getEigenData();
    Eigen::MatrixXd anglesMatrix = anglesData.getEigenData();
    int numSamples = gyroMatrix.rows();

    // Extract ground truth roll and pitch
    Eigen::VectorXd rollTruth = Utils::getVectorFromMatrix(anglesMatrix, 0);
    Eigen::VectorXd pitchTruth = Utils::getVectorFromMatrix(anglesMatrix, 1);

    // Convert to degrees
    Utils::convertToDeg(rollTruth);
    Utils::convertToDeg(pitchTruth);

    std::cout << "=== EKF Predict vs Ground Truth ===\n\n";
    std::cout << "Processing " << numSamples << " samples...\n\n";

    // Vectors to store results
    Eigen::VectorXd rollPredicted(numSamples);
    Eigen::VectorXd pitchPredicted(numSamples);

    // Process all samples
    for(int i = 0; i < numSamples; i++)
    {
        // Get gyro reading
        Eigen::Vector3d gyroReading = gyroMatrix.row(i).transpose();

        // Run predict
        ekf.predict(gyroReading);

        // Get Euler angles in degrees
        Eigen::Vector3d euler = ekf.getEulerAngles() * 180.0 / M_PI;
        rollPredicted(i) = euler(0);
        pitchPredicted(i) = euler(1);
    }

    // Display first 20 samples
    std::cout << "First 20 samples comparison:\n";
    std::cout << "Step | Roll Truth | Roll Predicted | Error | Pitch Truth | Pitch Predicted | Error\n";
    std::cout << "-----+------------+----------------+-------+-------------+-----------------+-------\n";

    int displaySteps = std::min(20, numSamples);
    for(int i = 0; i < displaySteps; i++)
    {
        double rollError = rollPredicted(i) - rollTruth(i);
        double pitchError = pitchPredicted(i) - pitchTruth(i);

        std::cout << std::setw(4) << i+1 << " | ";
        std::cout << std::setw(10) << rollTruth(i) << " | ";
        std::cout << std::setw(14) << rollPredicted(i) << " | ";
        std::cout << std::setw(5) << rollError << " | ";
        std::cout << std::setw(11) << pitchTruth(i) << " | ";
        std::cout << std::setw(15) << pitchPredicted(i) << " | ";
        std::cout << std::setw(5) << pitchError << "\n";
    }

    // Calculate error metrics
    std::cout << "\n=== Error Metrics ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollTruth, rollPredicted) << " deg\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollTruth, rollPredicted) << " deg\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchTruth, pitchPredicted) << " deg\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchTruth, pitchPredicted) << " deg\n";

    // Final values
    std::cout << "\n=== Final Values (sample " << numSamples << ") ===\n";
    std::cout << "Roll  - Truth: " << rollTruth(numSamples-1) << " deg, Predicted: " << rollPredicted(numSamples-1) << " deg\n";
    std::cout << "Pitch - Truth: " << pitchTruth(numSamples-1) << " deg, Predicted: " << pitchPredicted(numSamples-1) << " deg\n";

    // Save results for plotting
    Utils::printVecToFile(rollPredicted, "Results/ekf_predict_roll.txt");
    Utils::printVecToFile(pitchPredicted, "Results/ekf_predict_pitch.txt");
    Utils::printVecToFile(rollTruth, "Results/truth_roll.txt");
    Utils::printVecToFile(pitchTruth, "Results/truth_pitch.txt");

    std::cout << "\nResults saved to Results/ directory\n";
    std::cout << "Note: This is PREDICT only (no accelerometer correction yet)\n";
    std::cout << "Gyro drift will accumulate over time without the UPDATE step!\n";

    return 0;
}
