#ifndef COMPLEMENTARY_FILTER_HPP
#define COMPLEMENTARY_FILTER_HPP

#include "pch.h"

class ComplementaryFilter {
public:
    ComplementaryFilter() = default;
    ~ComplementaryFilter() = default;
    

    void calculateRoll();
    void setAccelData(const Eigen::MatrixXd& data);
    void setGyroData(const Eigen::MatrixXd& data);

    inline const Eigen::VectorXd& getRoll() const {
        return roll;
    }

    inline Eigen::VectorXd& getRoll() {
        return roll;
    }

private:
    const double dt{ 0.02 };
    double alphaCoeff{ 0.1 };

    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;

    Eigen::VectorXd phiG;
    Eigen::VectorXd phiA;
    Eigen::VectorXd roll;
};

#endif // COMPLEMENTARY_FILTER_HPP
