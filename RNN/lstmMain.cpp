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
        // LSTM expects 3D input: [batch, seq_len, features]
        // If input is 2D [batch, features], add seq_len=1 dimension
        if (X.dim() == 2) {
            X = X.unsqueeze(1);  // [batch, features] → [batch, 1, features]
        }

        auto lstmOutput = lstm->forward(X);
        auto out = std::get<0>(lstmOutput); // Get output, shape: [batch, seq_len, hidden]

        // Take last timestep output
        out = out.select(1, -1);  // [batch, seq_len, hidden] → [batch, hidden]

        // Pass through FC layer
        out = fc->forward(out);  // [batch, hidden] → [batch, output_size]
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
    const int windowSize = 10;  // Predict next 10 timesteps
    const int NUM_INPUT_FEATURES = 9;   // 9 columns in dataset_1.csv
    const int NUM_OUTPUT_FEATURES = 3;  // Predict 3 angles (roll, pitch, yaw)

    std::cout << "=== LSTM Multi-Step Ahead Prediction ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Prediction horizon (N): " << windowSize << " timesteps" << std::endl;
    std::cout << "  Input features: " << NUM_INPUT_FEATURES << std::endl;
    std::cout << "  Output features: " << NUM_OUTPUT_FEATURES << std::endl;
    std::cout << std::endl;

    CsvReader datasetReader("Data/dataset_1.csv");
    datasetReader.read();
    datasetReader.printStats();

    Eigen::MatrixXd dataset = datasetReader.getEigenData();
    int totalSamples = dataset.rows();

    std::cout << "Dataset loaded: " << totalSamples << " samples" << std::endl;
    std::cout << "Columns: [roll_gt, pitch_gt, yaw_gt, gyro_r, gyro_p, gyro_y, torque_r, torque_p, torque_y]" << std::endl;
    std::cout << std::endl;

    int trainingSamples = totalSamples - windowSize;  // Can't use last N rows as input
    std::cout << "Number of training samples: " << trainingSamples << std::endl;

    // Convert dataset to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);
    std::cout << "Dataset tensor shape: " << datasetTensor.sizes() << std::endl;

    // Pre-allocate X and y tensors
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor X = torch::zeros({trainingSamples, NUM_INPUT_FEATURES}, options);
    Tensor y = torch::zeros({trainingSamples, windowSize, NUM_OUTPUT_FEATURES}, options);

    // Pre-extract angle columns (first 3 columns) for efficiency
    Tensor anglesTensor = datasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);
    // Shape: [3397, 3] - all timesteps, angles only

    // Create samples (single loop - much cleaner!)
    for (int i = 0; i < trainingSamples; i++) {
        // Input: all 9 features at timestep i
        X[i] = datasetTensor[i];

        // Target: next N timesteps, angles only
        y[i] = anglesTensor.slice(0, i+1, i+windowSize+1);  // Rows [i+1 to i+N]
    }

    std::cout << std::endl;
    std::cout << "=== Data Shapes ===" << std::endl;
    std::cout << "X (inputs):  " << X.sizes() << std::endl;
    std::cout << "y (targets): " << y.sizes() << std::endl;
    std::cout << std::endl;

    // ==========================================
    // VERIFY DATA
    // ==========================================
    std::cout << "=== Data Verification ===" << std::endl;
    std::cout << "First input sample (timestep 0):" << std::endl;
    std::cout << X[0] << std::endl;
    std::cout << std::endl;

    std::cout << "First target (predict timesteps 1-10, angles only):" << std::endl;
    std::cout << "Shape: " << y[0].sizes() << std::endl;
    std::cout << "First 3 predictions:" << std::endl;
    std::cout << y[0].slice(0, 0, 3) << std::endl;
    std::cout << std::endl;

    // // ==========================================
    // // TEST MODEL (Optional - just forward pass)
    // // ==========================================
    // std::cout << "=== Testing Model Architecture ===" << std::endl;

    // // Model output should be [batch, N*3] or we reshape to [batch, N, 3]
    int outputSize = windowSize * NUM_OUTPUT_FEATURES;  // 10 * 3 = 30
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, 64, outputSize);

    // // Test with first sample
    Tensor testInput = X[0].unsqueeze(0);  // Add batch dimension: [1, 9]
    std::cout << "Test input shape: " << testInput.sizes() << std::endl;

    Tensor testOutput = model->forward(testInput);
    std::cout << "Test output shape: " << testOutput.sizes() << std::endl;

    // Reshape output to [batch, N, 3]
    Tensor reshapedOutput = testOutput.view({-1, windowSize, NUM_OUTPUT_FEATURES});
    std::cout << "Reshaped output: " << reshapedOutput.sizes() << std::endl;
    std::cout << std::endl;
    std::cout << "Prediction output: " << reshapedOutput << std::endl;

    // std::cout << "=== Data preparation complete! ===" << std::endl;
    // std::cout << "Ready for training implementation." << std::endl;

    return 0;
}
