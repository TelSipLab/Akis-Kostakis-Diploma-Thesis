#include "ComplementaryFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

int main() {
    std::cout.precision(10);

    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    gyroData.printStats();

    CsvReader accelData("Data/accel.csv");
    accelData.read();
    accelData.printStats();

    CsvReader anglesResults("Data/angles.csv");
    anglesResults.read();
    anglesResults.printStats();

    double alpha = 0.8;
    double dt = 0.02;
    ComplementaryFilter filter(alpha, dt);
    filter.setGyroData(gyroData.getEigenData());
    filter.setAccelData(accelData.getEigenData());
    filter.calculateRoll();
    filter.calculatePitch();

    auto& roll = filter.getRoll(); // Results are in RAD
    Utils::convertToDeg(roll);     // Now results are in Degree
    Utils::printVec(roll);

    auto& pitch = filter.getPitch(); // Results are in RAD
    Utils::convertToDeg(pitch);      // Now results are in Degree

    // File contains RAD
    Eigen::VectorXd rollTruthVector = Utils::getVectorFromMatrix(anglesResults.getEigenData(), 0);
    Eigen::VectorXd pitchTruthVector = Utils::getVectorFromMatrix(anglesResults.getEigenData(), 1);

    Utils::convertToDeg(rollTruthVector);  // Now in Degrees
    Utils::convertToDeg(pitchTruthVector); // Now in Degrees

    std::cout << "\n=== Error Metrics (all samples) ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollTruthVector, roll) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollTruthVector, roll) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchTruthVector, pitch) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchTruthVector, pitch) << " degrees\n";

    Utils::printVecToFile(roll, "Results/predicted_roll_complementary.txt");
    // Utils::printVecToFile(rollTruthVector, "Results/expected_roll.txt");

    Utils::printVecToFile(pitch, "Results/predicted_pitch_complementary.txt");
    // Utils::printVecToFile(pitchTruthVector, "Results/expected_pitch.txt");

    return 0;
}
