#include <Eigen/Dense>

#include <iostream>

int main()
{
    Eigen::Matrix<int, 2, 2> matrix;

    std::cout << matrix(0, 0) << std::endl;

    std::cout << "Number of cols: " << matrix.cols() << std::endl;

    std::cout << matrix.col(2);
    return 0;
}