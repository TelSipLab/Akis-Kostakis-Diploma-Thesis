#include "ComplementaryFilter.hpp"

#include "pch.h"

void ComplementaryFilter::calculateRoll() {
    phiG(0) = gyroData(0, 0) * dt;

    for (int i = 1; i < gyroData.rows(); i++) {
        phiG(i) = phiG(i - 1) + gyroData(i, 0) * dt;
    }

    for (int i = 0; i < accelometerData.rows(); i++) {
        phiA(i) = atan2(accelometerData(i, 1), accelometerData(i, 2));
    }

    for (int i = 0; i < gyroData.rows(); i++) {
        double result = alphaCoeff * phiG(i) + (1 - alphaCoeff) * phiA(i);
        roll(i) = result;
    }
}

void ComplementaryFilter::setAccelData(const Eigen::MatrixXd& data) {
    accelometerData = data;
    phiA.resize(data.rows());
}

void ComplementaryFilter::setGyroData(const Eigen::MatrixXd& data) {
    gyroData = data;
    phiG.resize(gyroData.rows());
    roll.resize(data.rows());
}