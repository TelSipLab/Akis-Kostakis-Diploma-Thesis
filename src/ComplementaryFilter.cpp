#include "ComplementaryFilter.hpp"

#include <iostream>

ComplementaryFilter::ComplementaryFilter(double alphaC, double dtarg) {
    alphaCoeff = alphaC;
    dt = dtarg;
}

void ComplementaryFilter::setAccelData(const Eigen::MatrixXd& data) {
    accelometerData = data;

    // TODO move these out of here
    phiA.resize(data.rows());
    phiA.setZero();
}

void ComplementaryFilter::setGyroData(const Eigen::MatrixXd& data) {
    gyroData = data;

    // TODO move these out of here
    phiG.resize(gyroData.rows());
    roll.resize(data.rows());
    pitch.resize(data.rows());
    thetaG.resize(data.rows());
    thetaA.resize(data.rows());

    phiG.setZero();
    roll.setZero();
    pitch.setZero();
}

void ComplementaryFilter::calculateRoll() {
    if(gyroData.rows() != accelometerData.rows()) {
        // TODO throw exception ??
        std::cout << "Wrong number of rows wont calculate roll \n";
        return;
    }

    // All data are init to zero no need to initialize
    for(int i = 1; i < gyroData.rows(); i++) { // We start from 1
        double ay = accelometerData(i, 1);
        double az = accelometerData(i, 2);

        phiG(i) = roll(i - 1) + gyroData(i, 0) * dt;
        phiA(i) = std::atan2(ay, az);

        double rollResult = alphaCoeff * phiG(i) + (1 - alphaCoeff) * phiA(i);
        roll(i) = rollResult;
    }
}

void ComplementaryFilter::calculatePitch() {
    if(gyroData.rows() != accelometerData.rows()) {
        // TODO throw exception ??
        std::cout << "Wrong data wont calculate pitch...\n";
        return;
    }

    thetaG.setZero();
    thetaA.setZero();
    const double diffConst = 1 - alphaCoeff;

    for(int i = 1; i < gyroData.rows(); i++) {
        double ax = accelometerData(i, 0);
        double ay = accelometerData(i, 1);
        double az = accelometerData(i, 2);
        double squareData = (ay * ay) + (az * az);

        thetaG(i) = pitch(i - 1) + gyroData(i, 1) * dt;
        thetaA(i) = std::atan2(ax, std::sqrt(squareData)); // TODO NOTICE

        double pitchResult = alphaCoeff * thetaG(i) + diffConst * thetaA(i);
        pitch(i) = pitchResult;
    }
}
