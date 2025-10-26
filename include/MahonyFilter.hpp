#ifndef MAHONYFILTER_HPP
#define MAHONYFILTER_HPP

#include "pch.h"

class MahonyFilter {
public:
    MahonyFilter(double dt, double kp);

    void setData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);

    void update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y);

    // Initialize R̂ with a given rotation matrix
    void initialize(const Eigen::Matrix3d& R_init) {
        rHat = R_init;
    }

    // Just do a full run on the data
    void calculate();

    Eigen::Vector3d getEulerAngles() const {
        return rHat.eulerAngles(0, 1, 2);
    }

    Eigen::Matrix3d rHat;  // Make public for direct access

private:
    double dt;
    double kp;

    // Data gathered from the sensons
    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;


    // UTILS TO BE MOVED

    // Skew-symmetric matrix from vector (Section II-A)
    static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d s;
        s << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        return s;
    }


    // Vector from skew-symmetric matrix (Section II-A)
    static Eigen::Vector3d vex(const Eigen::Matrix3d& M) {
        return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
    }

    // TODO Check why this is needed
    void orthonormalize() {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rHat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        rHat = svd.matrixU() * svd.matrixV().transpose();

        // Ensure det(R̂) = 1
        if (rHat.determinant() < 0) {  // ← Fixed
            Eigen::Matrix3d U = svd.matrixU();
            U.col(2) *= -1;
            rHat = U * svd.matrixV().transpose();  // ← Fixed
        }
    }
};

#endif // MAHONYFILTER_HPP
