#include <string>
#include <iostream>
#include "csvreader.hpp"

#include <torch/torch.h>

using torch::Tensor;

// TODO Move out of here if network get's too complex
class LSTMNetwork: public torch::nn::Module {
public:
    LSTMNetwork(int inputSize, int hiddenStateSize, int outputSize) {
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(inputSize, hiddenStateSize).batch_first(true)
        ));

        fc = register_module("fc", torch::nn::Linear(hiddenStateSize, outputSize));

        this->to(torch::kDouble);
    }

    Tensor forward(Tensor X) {
        auto lstmOutput = lstm->forward(X);
        auto out = std::get<0>(lstmOutput); // TODO Check what is the second value of the tuple
        out = fc->forward(out);
        return out;
    }

    static Tensor eigenToTensor(const Eigen::MatrixXd& matrixToConvert) {
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
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};

int main() {
    int numberOfFeatures = 5; // CHECK MEEEEEE
    // [omegax], [omegay], [omegaz], [ekfPitchRad], [efkRollRad] 

    CsvReader gyroData("Data/gyro.csv");
    CsvReader filterPitchData("Results/Results/EkfPitchRad.txt");
    CsvReader filterRollData("Results/Results/EkfRollRad.txt");

    gyroData.read();
    gyroData.printStats();

    filterPitchData.read();
    filterPitchData.printStats();

    filterRollData.read();
    filterRollData.printStats();

    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();
    Eigen::MatrixXd filterResultsPitchRad = filterPitchData.getEigenData();
    Eigen::MatrixXd filterResultsRollRad = filterRollData.getEigenData();

    if(gyroMeasurements.rows() != filterResultsRollRad.rows()) {
        std::cout << "Size mis-match \n";
        exit(-1);
    }

    Eigen::MatrixXd matrixInputArguments;    
    matrixInputArguments.resize(gyroMeasurements.rows(), numberOfFeatures);

    for(int i = 0; i < gyroMeasurements.rows(); i++) {
        matrixInputArguments.row(i) <<  gyroMeasurements(i, 0),
                                        gyroMeasurements(i, 1),
                                        gyroMeasurements(i, 2),
                                        filterResultsPitchRad(i), filterResultsRollRad(i);
    }

    std::cout << "Populated eigen array: (" << matrixInputArguments.rows() << "X"
                << matrixInputArguments.cols() << "). Will be used as tensor input \n";

    Tensor allInputDataTensor = LSTMNetwork::eigenToTensor(matrixInputArguments);

    std::cout << "X shape: " << allInputDataTensor.sizes() << std::endl;
    std::cout << allInputDataTensor[-1] << std::endl;  // Output: 5
    std::cout << "Last row of Eigen matrix:\n" << matrixInputArguments.row(matrixInputArguments.rows()-1) << std::endl;
    
    auto model = std::make_shared<LSTMNetwork>(numberOfFeatures,10,2);

    // TEST A SAMPLE
    // auto tmpTensorInput = createdTensor.index({0}).unsqueeze(0).unsqueeze(0);
    // auto output = model->forward(tmpTensorInput);
    
    // std::cout << "Test output shape: " << output.sizes() << std::endl;
    // std::cout << "Test output:\n" << output << std::endl;


    Tensor X = allInputDataTensor.unsqueeze(0);  // Add batch dimension
    std::cout << "Input shape: " << X.sizes() << std::endl;  // [1, 1409, 5]

    Tensor output = model->forward(X);
    std::cout << "Output shape: " << output.sizes() << std::endl;  // [1, 1409, 2]
    
    // Remove batch dimension to get predictions
    Tensor predictions = output.squeeze(0);  // [1409, 2]
    std::cout << "Predictions shape: " << predictions.sizes() << std::endl;

    // Show first and last predictions
    std::cout << "\nFirst prediction: " << predictions[0] << std::endl;
    std::cout << "Last prediction:  " << predictions[-1] << std::endl;

    return 0;
}