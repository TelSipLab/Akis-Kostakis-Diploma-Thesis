#include <string>
#include <iostream>
#include <iomanip>
#include "csvreader.hpp"

#include <torch/torch.h>

using torch::Tensor;

// Copy of LSTMNetwork class (must match training!)
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
        if (X.dim() == 2) {
            X = X.unsqueeze(1);  // [batch, features] → [batch, 1, features]
        }
        auto lstmOutput = lstm->forward(X);
        auto out = std::get<0>(lstmOutput);
        out = out.select(1, -1);  // Take last timestep
        out = fc->forward(out);
        return out;
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./lstmEval.out <model_path> [sample_index]" << std::endl;
        std::cout << "Example: ./lstmEval.out lstm_model_epoch_300.pt 0" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int sampleIndex = (argc >= 3) ? std::atoi(argv[2]) : 0;

    const int lookbackWindow = 10;  // Must match training!
    const int windowSize = 5;
    const int NUM_INPUT_FEATURES = 9;
    const int NUM_OUTPUT_FEATURES = 3;
    const int outputSize = windowSize * NUM_OUTPUT_FEATURES;

    std::cout << "=== LSTM Model Evaluation (Single Sample) ===" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Sample index: " << sampleIndex << std::endl;
    std::cout << "Lookback window: " << lookbackWindow << " timesteps" << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model..." << std::endl;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, 128, outputSize);

    try {
        torch::load(model, modelPath);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // Load CSV to get the sample
    CsvReader datasetReader("Data/dataset_1.csv");
    datasetReader.read();
    Eigen::MatrixXd dataset = datasetReader.getEigenData();

    if (sampleIndex >= dataset.rows() - windowSize - lookbackWindow + 1) {
        std::cerr << "Sample index too large! Max: " << dataset.rows() - windowSize - lookbackWindow << std::endl;
        return 1;
    }

    std::cout << std::endl;

    // Prepare sequence input (lookback window)
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor input = torch::zeros({1, lookbackWindow, NUM_INPUT_FEATURES}, options);

    // Fill input with lookback window data
    for (int t = 0; t < lookbackWindow; t++) {
        for (int j = 0; j < NUM_INPUT_FEATURES; j++) {
            input[0][t][j] = dataset(sampleIndex + t, j);
        }
    }

    // Ground truth for next 5 timesteps (after lookback window)
    Tensor groundTruth = torch::zeros({windowSize, NUM_OUTPUT_FEATURES}, options);
    int predStart = sampleIndex + lookbackWindow;
    for (int t = 0; t < windowSize; t++) {
        for (int j = 0; j < NUM_OUTPUT_FEATURES; j++) {
            groundTruth[t][j] = dataset(predStart + t, j);  // Next 5 timesteps after lookback, first 3 columns
        }
    }

    // Run prediction
    model->eval();
    torch::NoGradGuard no_grad;

    Tensor output = model->forward(input);  // [1, 15]
    Tensor prediction = output.view({windowSize, NUM_OUTPUT_FEATURES});  // [5, 3]

    // Display results
    std::cout << "=== Input (timesteps " << sampleIndex << " to " << sampleIndex + lookbackWindow - 1 << ") ===" << std::endl;
    std::cout << "Shape: [1, " << lookbackWindow << ", " << NUM_INPUT_FEATURES << "]" << std::endl;
    std::cout << input << std::endl;
    std::cout << std::endl;

    std::cout << "=== Prediction (next 5 timesteps) ===" << std::endl;
    std::cout << "Shape: [5, 3] (roll, pitch, yaw)" << std::endl;
    std::cout << prediction << std::endl;
    std::cout << std::endl;

    std::cout << "=== Ground Truth (next 5 timesteps) ===" << std::endl;
    std::cout << groundTruth << std::endl;
    std::cout << std::endl;

    // Calculate error
    Tensor error = prediction - groundTruth;
    Tensor rmse = error.pow(2).mean().sqrt();
    std::cout << "=== Error Metrics ===" << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(6)
              << rmse.item<double>() << " rad = "
              << std::setprecision(3) << rmse.item<double>() * 180.0 / M_PI << " degrees" << std::endl;

    return 0;
}
