#include <Eigen/Dense>

#include <iostream>

int main () {
    Eigen::Matrix<int, 2, 2> matrix;

    std::cout << matrix(0,0) << std::endl;

    return 0;
}