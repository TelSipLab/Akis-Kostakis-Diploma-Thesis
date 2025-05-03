#include "ComplementaryFilter.hpp"

#include <cmath>
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
    
    thetaG.setZero();
    thetaA.setZero();
    phiG.setZero();
    roll.setZero();
    pitch.setZero();
}

void ComplementaryFilter::calculate() {
    if(gyroData.rows() != accelometerData.rows()) {
        // TODO throw exception ??
        std::cout << "Wrong number of rows wont calculate roll \n";
        return;
    }

    // All data are init to zero no need to initialize
    for (int i = 1; i < gyroData.rows(); i++) { // We start from 1
        double ax = accelometerData(i,0);
        double ay = accelometerData(i,1);
        double az = accelometerData(i, 2);

        phiG(i) = roll(i - 1) + gyroData(i, 0) * dt;
        thetaG(i) = pitch(i - 1) + gyroData(i,1) * dt;

        phiA(i) = std::atan2(ay, az);
        thetaA(i) = std::atan2(ax, std::sqrt((ay * ay) + (az * az)));
        
        double rollResult = alphaCoeff * phiG(i) + (1 - alphaCoeff) * phiA(i);
        double pitchResult = alphaCoeff * thetaG(i) + (1 - alphaCoeff) * thetaA(i);

        roll(i) = rollResult;
        pitch(i) = pitchResult;
    }
}
