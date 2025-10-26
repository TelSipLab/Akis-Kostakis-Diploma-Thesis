#include "ExtendedKalmanFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"
#include <iomanip>
#include <iostream>

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

    // Get matrices first
    Eigen::MatrixXd gyroMatrix = gyroData.getEigenData();
    Eigen::MatrixXd accelMatrix = accelData.getEigenData();
    Eigen::MatrixXd anglesMatrix = anglesData.getEigenData();

    // Initialize EKF with quaternion from first accelerometer reading
    double dt = 0.02; // 50 Hz sampling rate

    // Get first accel reading and compute initial roll/pitch
    Eigen::Vector3d firstAccel = accelMatrix.row(0).transpose();
    firstAccel(0) = -firstAccel(0); // Correct sensor X-axis inversion

    // Compute roll and pitch from accelerometer
    double roll0 = atan2(firstAccel(1), firstAccel(2));
    double pitch0 = atan2(-firstAccel(0), sqrt(firstAccel(1) * firstAccel(1) + firstAccel(2) * firstAccel(2)));

    // Convert to quaternion (Z-Y-X Euler angles, yaw=0)
    double cy = cos(0.0 * 0.5);
    double sy = sin(0.0 * 0.5);
    double cp = cos(pitch0 * 0.5);
    double sp = sin(pitch0 * 0.5);
    double cr = cos(roll0 * 0.5);
    double sr = sin(roll0 * 0.5);

    Eigen::Vector4d initialQuaternion;
    initialQuaternion(0) = cr * cp * cy + sr * sp * sy; // q0
    initialQuaternion(1) = sr * cp * cy - cr * sp * sy; // q1
    initialQuaternion(2) = cr * sp * cy + sr * cp * sy; // q2
    initialQuaternion(3) = cr * cp * sy - sr * sp * cy; // q3

    ExtendedKalmanFilter ekf(dt, initialQuaternion);

    std::cout << "=== EKF Predict + Update Test with Real Data ===\n\n";

    int numSamples = gyroMatrix.rows();

    // Extract ground truth
    Eigen::VectorXd rollTruth = Utils::getVectorFromMatrix(anglesMatrix, 0);
    Eigen::VectorXd pitchTruth = Utils::getVectorFromMatrix(anglesMatrix, 1);
    Utils::convertToDeg(rollTruth);
    Utils::convertToDeg(pitchTruth);

    std::cout << "Number of samples: " << numSamples << "\n";
    std::cout << "Initial Quaternion: " << ekf.getQuaternion().transpose() << "\n";
    std::cout << "Initial Euler Angles (deg): " << ekf.getEulerAngles().transpose() * 180.0 / M_PI << "\n\n";

    // Vectors to store results
    Eigen::VectorXd rollPredicted(numSamples);
    Eigen::VectorXd pitchPredicted(numSamples);

    // Process first 20 samples and display results
    std::cout << "First 20 samples (Predict + Update):\n";
    std::cout << "Step | Roll Truth | Roll Est | Pitch Truth | Pitch Est\n";
    std::cout << "-----+------------+----------+-------------+-----------\n";

    int displaySteps = std::min(100, numSamples);
    for(int i = 0; i < displaySteps; i++)
    {
        // Get sensor readings
        Eigen::Vector3d gyroReading = gyroMatrix.row(i).transpose();
        Eigen::Vector3d accelReading = accelMatrix.row(i).transpose();

        // Run PREDICT step
        ekf.predict(gyroReading);

        // Run UPDATE step
        ekf.update(accelReading);

        // Get results in degrees
        Eigen::Vector3d euler = ekf.getEulerAngles() * 180.0 / M_PI;
        rollPredicted(i) = euler(0);
        pitchPredicted(i) = euler(1);

        std::cout << std::setw(4) << i + 1 << " | ";
        std::cout << std::setw(10) << rollTruth(i) << " | ";
        std::cout << std::setw(8) << rollPredicted(i) << " | ";
        std::cout << std::setw(11) << pitchTruth(i) << " | ";
        std::cout << std::setw(9) << pitchPredicted(i) << "\n";
    }

    std::cout << "\n=== RMSE and MEA for first 20 samples ===\n";
    Eigen::VectorXd rollTruth20 = rollTruth.head(displaySteps);
    Eigen::VectorXd pitchTruth20 = pitchTruth.head(displaySteps);
    Eigen::VectorXd rollPredicted20 = rollPredicted.head(displaySteps);
    Eigen::VectorXd pitchPredicted20 = pitchPredicted.head(displaySteps);

    std::cout << "RMSE between estimated and truth roll: " << Utils::rmse(rollTruth20, rollPredicted20) << std::endl;
    std::cout << "MEA between estimated and truth roll: " << Utils::mea(rollTruth20, rollPredicted20) << std::endl;
    std::cout << "RMSE between estimated and truth pitch: " << Utils::rmse(pitchTruth20, pitchPredicted20) << std::endl;
    std::cout << "MEA between estimated and truth pitch: " << Utils::mea(pitchTruth20, pitchPredicted20) << std::endl;
}
