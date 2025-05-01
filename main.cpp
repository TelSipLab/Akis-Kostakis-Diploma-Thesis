#include "csvreader.hpp"
#include "Utils.hpp"
#include "ComplementaryFilter.hpp"

int main()
{
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

    ComplementaryFilter filter;
    filter.setGyroData(gyroData.getEigenData());
    filter.setAccelData(accelData.getEigenData());
    filter.calculateRoll();

    auto& roll = filter.getRoll(); // Results are in RAD
    Utils::convertToDeg(roll); // Now results are in Degree

    // File contains RAD
    Eigen::VectorXd rollTruthVector = Utils::getVectorFromMatrix(anglesResults.getEigenData(), 0);
    Utils::convertToDeg(rollTruthVector); // Now in Degrees


    std::cout << "RMSE between estimated and truth roll: " << Utils::rmse(rollTruthVector, roll) << std::endl;
    std::cout << "MEA between estimated and truth roll: " << Utils::mea(rollTruthVector, roll) << std::endl;

    Utils::printVecToFile(roll, "Results/predicted_roll.txt");
    Utils::printVecToFile(rollTruthVector, "Results/expected_roll.txt");

    return 0;
}
