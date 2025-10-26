#ifndef UTILS_HPP
#define UTILS_HPP

#include "pch.h"

#include <fstream>
#include <numeric>

class Utils
{
  public:
    // Degrees -> Radians
    static void convertToRad(Eigen::VectorXd &vector)
    {
        vector = (vector * M_PI) / 180.0;
    }

    // Radians -> Degrees
    static void convertToDeg(Eigen::VectorXd &vector)
    {
        vector = (vector * 180.0) / M_PI;
    }

    static void printVec(const Eigen::VectorXd &vector, int n = 10)
    {
        for(int i = 0; i < 10; i++)
        {
            std::cout << vector(i) << "\n";
        }

        std::cout << std::endl;
    }

    static double rmse(const Eigen::VectorXd &A, const Eigen::VectorXd &B)
    {
        double sum = std::inner_product(A.data(), A.data() + A.size(), B.data(), 0.0, std::plus<double>(),
                                        [](double a, double b)
                                        {
                                            double e = a - b;
                                            return e * e;
                                        });

        return std::sqrt(sum / A.size());
    }

    static double mea(const Eigen::VectorXd &truth, const Eigen::VectorXd &predicted)
    {
        return (truth - predicted).cwiseAbs().mean();
    }

    static Eigen::VectorXd getVectorFromMatrix(Eigen::MatrixXd &matrix, int colNumber)
    {
        return matrix.col(colNumber);
    }

    static void printVecToFile(const Eigen::VectorXd &vector, const std::string &filePath)
    {
        std::ofstream outFile(filePath, std::ios::trunc);
        if(outFile.is_open())
        {
            for(int i = 0; i < vector.size(); i++)
            {
                outFile << vector(i) << "\n";
            }
            outFile.close();
        }
        else
        {
            std::cerr << "Unable to open file: " << filePath << std::endl;
        }
    }

    // Skew-symmetric matrix from vector (Section II-A)
    static Eigen::Matrix3d skewMatrixFromVector(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d s;
        s << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return s;
    }

    // Vector from skew-symmetric matrix (Section II-A)
    static Eigen::Vector3d vexFromSkewMatrix(const Eigen::Matrix3d &skewMatrix)
    {
        return Eigen::Vector3d(skewMatrix(2, 1), skewMatrix(0, 2), skewMatrix(1, 0));
    }

    // Equation from (Complementary_Filter_Introduction.pdf)
    static double calculateRollFromAccelInput(double ay, double az) {
        return std::atan2(ay, az);
    }

    // Equation from (Complementary_Filter_Introduction.pdf)
    static double calculatePitchFromAccelInput(double ax, double ay, double az) {
        double tmp = std::sqrt(ay * ay + az * az);
        return std::atan2(-ax, tmp);
    }
};

#endif // UTILS_HPP
