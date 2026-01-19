#include "csvreader.hpp"

#include <exception>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

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
        // Input: [batch, seq_len=10, features=9]
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

bool isNumber(const std::string& str) {
    if (str.empty()) return false;
    
    try {
        std::size_t pos;
        std::stoi(str, &pos);
        return pos == str.length(); // ensure entire string was converted
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    bool saveAll = false;
    std::string modelPath;
    int sampleIndex = 0;

    if(argc == 1) {
        std::cout << "Wrong input use -h to see help" << std::endl;
        return -1;
    }

    modelPath = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--save-all" || arg == "-a" ) {
            saveAll = true;
        } else if(i == 2 && isNumber(arg)) {
            sampleIndex = std::stoi(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./lstmEval.out <model_path> [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  [sample_index]       Evaluate single sample (e.g., 0, 1000)" << std::endl;
            std::cout << "  --save-all, -a       Generate predictions for ALL samples and save to CSV" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  ./lstmEval.out lstm_model_epoch_300.pt 0" << std::endl;
            std::cout << "  ./lstmEval.out lstm_model_epoch_300.pt --save-all" << std::endl;
            return 0;
        } else {
            std::cout << "Wrong argument use -h to view help" << std::endl;
        }
    }

    const int lookbackWindow = 10;  // Must match training!
    const int windowSize = 10;
    const int NUM_INPUT_FEATURES = 9;
    const int NUM_OUTPUT_FEATURES = 3;
    const int outputSize = windowSize * NUM_OUTPUT_FEATURES;

    std::cout << "LSTM Model Evaluation" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    if (saveAll) {
        std::cout << "Mode: Generate predictions for ALL samples" << std::endl;
    } else {
        std::cout << "Mode: Single sample evaluation (index: " << sampleIndex << ")" << std::endl;
    }
    std::cout << "Lookback window: " << lookbackWindow << " timesteps" << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model..." << std::endl;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, 128, outputSize);

    try {
        torch::load(model, modelPath);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // Load CSV
    CsvReader datasetReader("Data/dataset_1.csv");
    datasetReader.read();
    Eigen::MatrixXd dataset = datasetReader.getEigenData();

    int totalSamples = dataset.rows();
    int trainingSamples = totalSamples - windowSize - lookbackWindow + 1;

    std::cout << std::endl;

    // Handle --save-all mode
    if (saveAll) {
        std::cout << "Generating predictions for " << trainingSamples << " samples..." << std::endl;

        model->eval();
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kDouble);

        // Open output file
        std::ofstream outFile("Results/lstm_predictions.csv");
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open Results/lstm_predictions.csv for writing" << std::endl;
            return 1;
        }

        // Write header
        outFile << "timestep,step_ahead,roll_pred,pitch_pred,yaw_pred,roll_gt,pitch_gt,yaw_gt" << std::endl;

        // Process all samples
        for (int sample = 0; sample < trainingSamples; sample++) {
            // Prepare input
            Tensor input = torch::zeros({1, lookbackWindow, NUM_INPUT_FEATURES}, options);
            for (int t = 0; t < lookbackWindow; t++) {
                for (int j = 0; j < NUM_INPUT_FEATURES; j++) {
                    input[0][t][j] = dataset(sample + t, j);
                }
            }

            // Get prediction
            Tensor output = model->forward(input);
            Tensor prediction = output.view({windowSize, NUM_OUTPUT_FEATURES});

            // Get ground truth
            int predStart = sample + lookbackWindow;

            // Write predictions for each of the 5 timesteps
            for (int step = 0; step < windowSize; step++) {
                int absoluteTimestep = predStart + step;
                int stepAhead = step + 1;
                
                // Get predicted angles
                auto pred = prediction[step];
                double roll_pred = pred[0].item<double>();
                double pitch_pred = pred[1].item<double>();
                double yaw_pred = pred[2].item<double>();

                // Get ground truth angles
                double roll_gt = dataset(absoluteTimestep, 0);
                double pitch_gt = dataset(absoluteTimestep, 1);
                double yaw_gt = dataset(absoluteTimestep, 2);

                // Write to CSV
                outFile << absoluteTimestep << "," << stepAhead << ","
                       << roll_pred << "," << pitch_pred << "," << yaw_pred << ","
                       << roll_gt << "," << pitch_gt << "," << yaw_gt << std::endl;
            }

            // Progress indicator
            if (sample % 500 == 0) {
                std::cout << "  Processed " << sample << "/" << trainingSamples << " samples" << std::endl;
            }
        }

        outFile.close();
        std::cout << "Done! Saved predictions to Results/lstm_predictions.csv" << std::endl;
        std::cout << "Total prediction rows: " << trainingSamples * windowSize << std::endl;
        return 0;
    }

    // Single sample evaluation mode
    if (sampleIndex >= trainingSamples) {
        std::cerr << "Sample index too large! Max: " << trainingSamples - 1 << std::endl;
        return 1;
    }

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
