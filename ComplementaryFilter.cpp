#include "ComplementaryFilter.hpp"

#include <iostream>

ComplementaryFilter::ComplementaryFilter(double alphaC, double dtarg) {
    alphaCoeff = alphaC;
    dt = dtarg;
}

void ComplementaryFilter::setAccelData(const Eigen::MatrixXd& data) {
    accelometerData = data;
    phiA.resize(data.rows());
    phiA.setZero();
}

void ComplementaryFilter::setGyroData(const Eigen::MatrixXd& data) {
    gyroData = data;
    phiG.resize(gyroData.rows());
    roll.resize(data.rows());

    phiG.setZero();
    roll.setZero();
}

void ComplementaryFilter::calculateRoll() {
    if(gyroData.rows() != accelometerData.rows()) {
        std::cout << "Wrong number of rows wont calculate roll \n";
        return;
    }

    // All data are init to zero no need to initialize

    for (int i = 1; i < gyroData.rows(); i++) { // At 0 roll is zero so we start from 1
        phiG(i) = roll(i - 1) + gyroData(i, 0) * dt;
        phiA(i) = atan2(accelometerData(i, 1), accelometerData(i, 2));

        double result = alphaCoeff * phiG(i) + (1 - alphaCoeff) * phiA(i);
        roll(i) = result;
    }
}
