#include "MahonyFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const double kp = 1.0;  // Proportional gain from paper
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

    // Ground truth (convert to degrees)
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

    // Initialize filter with first accelerometer reading
    Eigen::Vector3d accel0 = accelMeasurements.row(0).transpose();
    accel0(0) = -accel0(0);
    Eigen::Vector3d accel0_norm = accel0.normalized();

    // double roll0 = atan2(accel0_norm.y(), accel0_norm.z());
    // double pitch0 = atan2(accel0_norm.x(), sqrt(accel0_norm.y() * accel0_norm.y() + accel0_norm.z() * accel0_norm.z()));

    // double cr0 = cos(roll0);
    // double sr0 = sin(roll0);
    // double cp0 = cos(pitch0);
    // double sp0 = sin(pitch0);

    // Eigen::Matrix3d R_init;
    // R_init << cp0, sp0*sr0, sp0*cr0,
    //           0,   cr0,     -sr0,
    //           -sp0, cp0*sr0, cp0*cr0;

    // mahony.initialize(R_init);

    // std::cout << "Initialized with first accelerometer reading:\n";
    // std::cout << "  Initial Roll:  " << roll0 * 180.0 / M_PI << " degrees\n";
    // std::cout << "  Initial Pitch: " << pitch0 * 180.0 / M_PI << " degrees\n\n";

    std::cout << "Processing all samples...\n";
    std::cout << "Step | Roll Truth | Roll Est | Pitch Truth | Pitch Est\n";
    std::cout << "-----+------------+----------+-------------+-----------\n";

    for (int i = 0; i < numSamples; i++) {
        Eigen::Vector3d gyroReading = gyroMeasurements.row(i).transpose();
        Eigen::Vector3d accelReading = accelMeasurements.row(i).transpose();
        accelReading(0) = -accelReading(0);  // Sign correction (same as EKF)

        // Construct R_y from accelerometer (simplified - assumes gravity is dominant)
        // This gives us roll and pitch, but not yaw
        Eigen::Vector3d accel_norm = accelReading.normalized();

        // Compute roll and pitch from accelerometer (same as ComplementaryFilter)
        double roll_meas = atan2(accel_norm.y(), accel_norm.z());
        double pitch_meas = atan2(accel_norm.x(),  // No negative sign!
                                   sqrt(accel_norm.y() * accel_norm.y() +
                                        accel_norm.z() * accel_norm.z()));

        // Construct rotation matrix R_y (ZYX Euler angles with yaw=0)
        Eigen::Matrix3d R_y;
        double cr = cos(roll_meas);
        double sr = sin(roll_meas);
        double cp = cos(pitch_meas);
        double sp = sin(pitch_meas);

        R_y << cp, sp*sr, sp*cr,
               0,  cr,    -sr,
               -sp, cp*sr, cp*cr;

        // Update Mahony filter
        mahony.update(gyroReading, R_y);

        // Extract roll and pitch directly from rotation matrix R̂
        // For ZYX Euler angles: roll = atan2(R32, R33), pitch = atan2(-R31, sqrt(R32² + R33²))
        Eigen::Vector3d euler = mahony.getEulerAngles();
        Eigen::Matrix3d R_hat = mahony.rHat;  // Access rotation matrix

        double rollEst = atan2(R_hat(2, 1), R_hat(2, 2));
        double pitchEst = atan2(-R_hat(2, 0), sqrt(R_hat(2, 1) * R_hat(2, 1) + R_hat(2, 2) * R_hat(2, 2)));

        rollEstimated(i) = rollEst * 180.0 / M_PI;
        pitchEstimated(i) = pitchEst * 180.0 / M_PI;

        // Display first and last samples
        if (i < DISPLAY_SAMPLES || i >= numSamples - DISPLAY_SAMPLES) {
            std::cout << std::setw(4) << i+1 << " | ";
            std::cout << std::setw(10) << rollGroundTruth(i) << " | ";
            std::cout << std::setw(8) << rollEstimated(i) << " | ";
            std::cout << std::setw(11) << pitchGroundTruth(i) << " | ";
            std::cout << std::setw(9) << pitchEstimated(i) << "\n";
        } else if (i == DISPLAY_SAMPLES) {
            std::cout << "...\n";
        }
    }

    // Calculate error metrics
    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, rollEstimated) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, pitchEstimated) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, pitchEstimated) << " degrees\n";

    // Save results to files
    // Utils::printVecToFile(rollEstimated, "Results/predicted_roll_mahony.txt");
    // Utils::printVecToFile(pitchEstimated, "Results/predicted_pitch_mahony.txt");

    // std::cout << "\nResults saved to Results/predicted_roll_mahony.txt and predicted_pitch_mahony.txt\n";

    return 0;
}
