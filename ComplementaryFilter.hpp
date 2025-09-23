#ifndef COMPLEMENTARY_FILTER_HPP
#define COMPLEMENTARY_FILTER_HPP

#include "pch.h"

class ComplementaryFilter
{
  public:
    ComplementaryFilter() = default;
    ~ComplementaryFilter() = default;

    ComplementaryFilter(double alphaC, double dt);

    void setAccelData(const Eigen::MatrixXd &data);
    void setGyroData(const Eigen::MatrixXd &data);

    inline const Eigen::VectorXd &getRoll() const
    {
        return roll;
    }

    inline Eigen::VectorXd &getRoll()
    {
        return roll;
    }

    inline Eigen::VectorXd &getPitch()
    {
        return pitch;
    }

    void calculateRoll();
    void calculatePitch();

  private:
    // Some default values
    double dt{0.02};
    double alphaCoeff{0.9};

    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;

    Eigen::VectorXd phiG;
    Eigen::VectorXd phiA;
    Eigen::VectorXd thetaG;
    Eigen::VectorXd thetaA;

    // Predictions
    Eigen::VectorXd roll;
    Eigen::VectorXd pitch;
};

#endif // COMPLEMENTARY_FILTER_HPP
