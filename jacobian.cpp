#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

// Simple function: f(x) = [x1^2 + x2, sin(x1), x1*x2]
Vector3d simpleFunction(const Vector2d& x) {
    Vector3d result;
    result(0) = x(0)*x(0) + x(1);      // x1^2 + x2
    result(1) = sin(x(0));             // sin(x1)
    result(2) = x(0) * x(1);           // x1 * x2
    return result;
}

// Analytical Jacobian of f(x) with respect to x
// J = [∂f/∂x1  ∂f/∂x2]
MatrixXd analyticalJacobian(const Vector2d& x) {
    MatrixXd J(3, 2);

    // Row 1: ∂(x1^2 + x2)/∂x = [2*x1, 1]
    J(0, 0) = 2 * x(0);
    J(0, 1) = 1;

    // Row 2: ∂(sin(x1))/∂x = [cos(x1), 0]
    J(1, 0) = cos(x(0));
    J(1, 1) = 0;

    // Row 3: ∂(x1*x2)/∂x = [x2, x1]
    J(2, 0) = x(1);
    J(2, 1) = x(0);

    return J;
}

// Numerical Jacobian using finite differences
MatrixXd numericalJacobian(const Vector2d& x, double epsilon = 1e-6) {
    MatrixXd J(3, 2);
    Vector3d f_x = simpleFunction(x);

    for (int j = 0; j < 2; ++j) {
        Vector2d x_plus = x;
        x_plus(j) += epsilon;
        Vector3d f_plus = simpleFunction(x_plus);

        // Central difference: (f(x+h) - f(x-h)) / (2*h)
        Vector2d x_minus = x;
        x_minus(j) -= epsilon;
        Vector3d f_minus = simpleFunction(x_minus);

        J.col(j) = (f_plus - f_minus) / (2 * epsilon);
    }

    return J;
}

int main() {
    // Test point
    Vector2d x(1.5, 2.0);

    cout << "=== Simple Jacobian Example with Eigen ===" << endl;
    cout << "Function: f(x) = [x1^2 + x2, sin(x1), x1*x2]" << endl;
    cout << "Test point: x = [" << x.transpose() << "]" << endl << endl;

    // Evaluate function
    Vector3d f_val = simpleFunction(x);
    cout << "f(x) = [" << f_val.transpose() << "]" << endl << endl;

    // Analytical Jacobian
    MatrixXd J_analytical = analyticalJacobian(x);
    cout << "Analytical Jacobian:" << endl;
    cout << J_analytical << endl << endl;

    // Numerical Jacobian
    MatrixXd J_numerical = numericalJacobian(x);
    cout << "Numerical Jacobian:" << endl;
    cout << J_numerical << endl << endl;

    // Compare accuracy
    MatrixXd diff = J_analytical - J_numerical;
    double max_error = diff.array().abs().maxCoeff();
    cout << "Maximum difference: " << max_error << endl;
    cout << "Jacobians match: " << (max_error < 1e-5 ? "YES" : "NO") << endl;

    return 0;
}