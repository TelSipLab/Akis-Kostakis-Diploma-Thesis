#include "MahonyFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const double kp = 50.0;  // Increased gain for faster correction
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

    std::cout << "Mahony Passive Complementary Filter Test\n";
    std::cout << "=========================================\n\n";

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    std::cout << "Dataset: " << numSamples << " samples\n";
    std::cout << "Time step (dt): " << dt << " seconds\n";
    std::cout << "Proportional gain (kP): " << kp << "\n\n";

    Eigen::VectorXd rollEstimated(numSamples);
    Eigen::VectorXd pitchEstimated(numSamples);

    std::cout << "Processing all samples...\n";
    std::cout << "Step | Roll Truth | Roll Est | Pitch Truth | Pitch Est\n";
    std::cout << "-----+------------+----------+-------------+-----------\n";

    for (int i = 0; i < numSamples; i++) {
        Eigen::Vector3d gyroReading = gyroMeasurements.row(i).transpose();
        Eigen::Vector3d accelReading = accelMeasurements.row(i).transpose();
        accelReading(0) = -accelReading(0);  // Sign correction

        // Normalize accelerometer reading
        Eigen::Vector3d accel_norm = accelReading.normalized();

        // Compute roll and pitch from accelerometer
        double roll_meas = atan2(accel_norm.y(), accel_norm.z());
        double pitch_meas = atan2(-accel_norm.x(),
            sqrt(accel_norm.y() * accel_norm.y() +
                accel_norm.z() * accel_norm.z()));

        // Construct rotation matrix R_y from accelerometer measurements
        Eigen::Matrix3d R_y =
            (Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *           // yaw = 0
                Eigen::AngleAxisd(pitch_meas, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_meas, Eigen::Vector3d::UnitX())).toRotationMatrix();

        // Update Mahony filter (Equation 10)
        mahony.update(gyroReading, R_y);

        // Extract roll and pitch from rotation matrix R̂
        // For ZYX Euler: roll = atan2(R32, R33), pitch = atan2(-R31, sqrt(R32² + R33²))
        Eigen::Matrix3d R_hat = mahony.rHat;

        double rollEst = atan2(R_hat(2, 1), R_hat(2, 2)) * 180.0 / M_PI;
        double pitchEst = atan2(-R_hat(2, 0),
            sqrt(R_hat(2, 1) * R_hat(2, 1) +
                R_hat(2, 2) * R_hat(2, 2))) * 180.0 / M_PI;

        rollEstimated(i) = rollEst;
        pitchEstimated(i) = pitchEst;

        // Display first and last samples
        if (i < DISPLAY_SAMPLES || i >= numSamples - DISPLAY_SAMPLES) {
            std::cout << std::setw(4) << i + 1 << " | ";
            std::cout << std::setw(10) << rollGroundTruth(i) << " | ";
            std::cout << std::setw(8) << rollEstimated(i) << " | ";
            std::cout << std::setw(11) << pitchGroundTruth(i) << " | ";
            std::cout << std::setw(9) << pitchEstimated(i) << "\n";
        }
        else if (i == DISPLAY_SAMPLES) {
            std::cout << "...\n";
        }
    }

    // Calculate error metrics
    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, pitchEstimated) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, pitchEstimated) << " degrees\n";

    return 0;
}