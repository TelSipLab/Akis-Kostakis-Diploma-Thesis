#include <string>
#include <iostream>
#include <torch/torch.h>

#include "csvreader.hpp"

using torch::Tensor;

class LSTMNetwork: public torch::nn::Module{
public:
    static Tensor eigenToTensor(Eigen::MatrixXd& matrixToConvert) {
        int rows = matrixToConvert.rows();
        int cols = matrixToConvert.cols();
        
        auto options = torch::TensorOptions().dtype(torch::kDouble);
        // Create tensor from raw data pointer
        auto tensor = torch::zeros({rows, cols}, options);

        for (int i = 0; i < rows; i++) {
            std::vector<double> rowData(cols);
            for (int j = 0; j < cols; j++) {
                rowData[j] = matrixToConvert(i, j);
            }

            auto rowTensor = torch::from_blob(rowData.data(), {cols}, options).clone();
            tensor[i] = rowTensor;  // Direct indexing - much cleaner!
        }

        return tensor;
    }
};

int main() {
    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    gyroData.printStats();
    
    CsvReader filterRollData("Results/Results/EkfRollRad.txt");
    filterRollData.read();
    filterRollData.printStats();

    CsvReader filterPitchData("Results/Results/EkfPitchRad.txt");
    filterPitchData.read();
    filterPitchData.printStats();

    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();
    Eigen::MatrixXd filterResultsRollRad = filterRollData.getEigenData();
    Eigen::MatrixXd filterResultsPitchRad = filterPitchData.getEigenData();

    if(gyroMeasurements.rows() != filterResultsRollRad.rows()) {
        std::cout << "Size mis-match \n";
        exit(-1);
    }

    Eigen::MatrixXd matrixInputArguments;
    int numberOfFeatures = 7; // TODO
    matrixInputArguments.resize(gyroMeasurements.rows(), 7);

    for(int i = 0; i < gyroMeasurements.rows(); i++) {
        matrixInputArguments.row(i) <<  gyroMeasurements(i, 0),
                                        gyroMeasurements(i, 1),
                                        gyroMeasurements(i, 2),
                                        4.0, 5.0, 6.0, 7.0;
    }

    std::cout << "Populated eigen array: (" << matrixInputArguments.rows() << "X"
                << matrixInputArguments.cols() << "). Will be used as tensor input \n";

    Tensor createdTensor = LSTMNetwork::eigenToTensor(matrixInputArguments);

    std::cout << "X shape: " << createdTensor.sizes() << std::endl;
    std::cout << createdTensor[-1] << std::endl;  // Output: 5
    std::cout << "Last row of Eigen matrix:\n" << matrixInputArguments.row(matrixInputArguments.rows()-1) << std::endl;
    
    return 0;
}