#ifndef MAHONYFILTER_HPP
#define MAHONYFILTER_HPP

#include "pch.h"

class MahonyFilter {
public:
    MahonyFilter(double dt);

    void setData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);
    
    // Just do a full run on the data
    void calculate();

private:
    double dt;

    Eigen::Quaterniond q_hat;

    // Data gathered from the sensons
    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;
};

#endif // MAHONYFILTER_HPP
