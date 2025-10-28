#include "MahonyFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    bool calculateBestK = false;

    const double dt = 0.02;
    const double kp = 9; // Was calcualted via searching loop
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


    Utils::printVecToFile(rollGroundTruth, "Results/ExpectedResults/expected_roll.txt");
    Utils::printVecToFile(pitchGroundTruth, "Results/ExpectedResults/expected_pitch.txt");
    exit(-1);

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    MahonyFilter mahony(dt, kp);
    mahony.setIMUData(gyroMeasurements, accelMeasurements);

    if(calculateBestK) {
        double bestK=1;
        double bestRMSE = 100.00;

        for(int i=1; i < 101; i++) {
            MahonyFilter mahonyTmp(dt, i);
            mahonyTmp.setIMUData(gyroMeasurements, accelMeasurements);

            mahonyTmp.predictForAllData();
            double tmpBetRmse = Utils::rmse(rollGroundTruth, mahonyTmp.getRollEstimation());

            if (tmpBetRmse < bestRMSE) {
                bestRMSE = tmpBetRmse;
                bestK = i;
            }
        }

        std::cout << "Best K " << bestK << " and best RMSE " << bestRMSE << std::endl;

    } else {
        std::cout << "Mahony Passive Complementary Filter Test\n\n";

        std::cout << "Dataset: " << numSamples << " samples\n";
        std::cout << "Time step (dt): " << dt << " seconds\n";
        std::cout << "Proportional gain (kP): " << kp << "\n\n";

        mahony.predictForAllData();
    }

    // Conver results to degrees
    Utils::convertToDeg(mahony.getRollEstimationNonConst());
    Utils::convertToDeg(mahony.getPitchEstimationNonConst());


    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) in degrees ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";

    Utils::printVecToFile(mahony.getRollEstimation(), "Results/Results/MahonyRoll_kp_9.txt");
    Utils::printVecToFile(mahony.getPitchEstimation(), "Results/Results/MahonyPitch_kp_9.txt");

    return 0;
}
