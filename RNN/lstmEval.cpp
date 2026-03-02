#include "Utils.hpp"
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
    LSTMNetwork(int inputSize, int hiddenStateSize, int outputSize, double dropoutRate = 0.2) {
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(inputSize, hiddenStateSize).batch_first(true)
        ));

        // Attention layers
        attn_linear = register_module("attn_linear", torch::nn::Linear(hiddenStateSize, hiddenStateSize));
        attn_vector = register_module("attn_vector", torch::nn::Linear(hiddenStateSize, 1));

        // Dropout (inactive during eval)
        dropout = register_module("dropout", torch::nn::Dropout(dropoutRate));

        fc = register_module("fc", torch::nn::Linear(hiddenStateSize, outputSize));
        this->to(torch::kDouble);
    }

    Tensor forward(Tensor X) {
        auto lstmOutput = lstm->forward(X);
        auto hiddenStates = std::get<0>(lstmOutput);

        hiddenStates = dropout->forward(hiddenStates);

        auto energy = torch::tanh(attn_linear->forward(hiddenStates));
        auto scores = attn_vector->forward(energy).squeeze(-1);
        auto weights = torch::softmax(scores, /*dim=*/1);

        auto context = torch::bmm(weights.unsqueeze(1), hiddenStates).squeeze(1);
        context = dropout->forward(context);

        auto out = fc->forward(context);
        return out;
    }

    static Tensor eigenToTensor(const Eigen::MatrixXd& matrixToConvert) {
        int rows = matrixToConvert.rows();
        int cols = matrixToConvert.cols();

        auto options = torch::TensorOptions().dtype(torch::kDouble);
        auto tensor = torch::zeros({rows, cols}, options);

        for (int i = 0; i < rows; i++) {
            std::vector<double> rowData(cols);
            for (int j = 0; j < cols; j++) {
                rowData[j] = matrixToConvert(i, j);
            }

            auto rowTensor = torch::from_blob(rowData.data(), {cols}, options).clone();
            tensor[i] = rowTensor;
        }
        return tensor;
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear attn_linear{nullptr};
    torch::nn::Linear attn_vector{nullptr};
    torch::nn::Dropout dropout{nullptr};
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

    // Check if first argument is a help flag
    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        std::cout << "Usage: ./lstmEval.out <model_path> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  [sample_index]       Evaluate single sample (e.g., 0, 1000)" << std::endl;
        std::cout << "  --save-all, -a       Generate predictions for ALL samples and save to CSV" << std::endl;
        std::cout << "  --help, -h           Show this help message" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt 0" << std::endl;
        std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt --save-all" << std::endl;
        return 0;
    }

    modelPath = firstArg;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--save-all" || arg == "-a" ) {
            saveAll = true;
        } else if(i == 2 && isNumber(arg)) {
            sampleIndex = std::stoi(arg);
        } else {
            std::cout << "Wrong argument use -h to view help" << std::endl;
        }
    }

    const int lookbackWindow = 10;  // Must match training!
    const int windowSize = 10;
    const int NUM_INPUT_FEATURES = 9;
    const int NUM_OUTPUT_FEATURES = 3;
    const int outputSize = windowSize * NUM_OUTPUT_FEATURES;
    const int hiddenStateSize = 32;

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
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

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

    // Convert to tensor and normalize (must match training!)
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);

    Tensor featureMean = datasetTensor.mean(/*dim=*/0, /*keepdim=*/true);
    Tensor featureStd = datasetTensor.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/true);
    featureStd = featureStd + 1e-8;

    Tensor scaledDatasetTensor = (datasetTensor - featureMean) / featureStd;

    Tensor targetMean = featureMean.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);
    Tensor targetStd = featureStd.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);

    // Train/Val split (must match training!)
    int numTrain = static_cast<int>(trainingSamples * 0.8);

    std::cout << "Data normalized (z-score standardization)" << std::endl;
    std::cout << "Train/Val split: " << numTrain << " train | " << (trainingSamples - numTrain) << " val" << std::endl;
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
        outFile << "timestep,step_ahead,roll_pred,pitch_pred,yaw_pred,roll_gt,pitch_gt,yaw_gt,set" << std::endl;

        // Process all samples
        for (int sample = 0; sample < trainingSamples; sample++) {
            // Prepare normalized input
            Tensor input = scaledDatasetTensor.slice(0, sample, sample + lookbackWindow).unsqueeze(0);

            // Get prediction (in normalized space)
            Tensor output = model->forward(input);
            Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

            // Denormalize predictions back to radians
            Tensor unscaledPred = (prediction * targetStd) + targetMean;
            unscaledPred = unscaledPred.squeeze(0); // [windowSize, 3]

            // Get ground truth
            int predStart = sample + lookbackWindow;

            // Write predictions for each step
            for (int step = 0; step < windowSize; step++) {
                int absoluteTimestep = predStart + step;
                int stepAhead = step + 1;

                // Get predicted angles (denormalized)
                auto pred = unscaledPred[step];
                double roll_pred = pred[0].item<double>();
                double pitch_pred = pred[1].item<double>();
                double yaw_pred = pred[2].item<double>();

                // Get ground truth angles (original dataset)
                double roll_gt = dataset(absoluteTimestep, 0);
                double pitch_gt = dataset(absoluteTimestep, 1);
                double yaw_gt = dataset(absoluteTimestep, 2);

                // Write to CSV
                std::string setLabel = (sample < numTrain) ? "train" : "val";
                outFile << absoluteTimestep << "," << stepAhead << ","
                       << roll_pred << "," << pitch_pred << "," << yaw_pred << ","
                       << roll_gt << "," << pitch_gt << "," << yaw_gt << ","
                       << setLabel << std::endl;
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

    // Prepare normalized input (lookback window)
    Tensor input = scaledDatasetTensor.slice(0, sampleIndex, sampleIndex + lookbackWindow).unsqueeze(0);

    // Ground truth for next N timesteps (after lookback window, original scale)
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor groundTruth = torch::zeros({windowSize, NUM_OUTPUT_FEATURES}, options);
    int predStart = sampleIndex + lookbackWindow;
    for (int t = 0; t < windowSize; t++) {
        for (int j = 0; j < NUM_OUTPUT_FEATURES; j++) {
            groundTruth[t][j] = dataset(predStart + t, j);
        }
    }

    // Run prediction
    model->eval();
    torch::NoGradGuard no_grad;

    Tensor output = model->forward(input);  // [1, N*3]
    Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

    // Denormalize predictions back to radians
    Tensor unscaledPred = ((prediction * targetStd) + targetMean).squeeze(0); // [N, 3]

    // Display results
    std::cout << "=== Input (timesteps " << sampleIndex << " to " << sampleIndex + lookbackWindow - 1 << ") ===" << std::endl;
    std::cout << "Shape: [1, " << lookbackWindow << ", " << NUM_INPUT_FEATURES << "]" << std::endl;
    std::cout << "(normalized)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Prediction (next " << windowSize << " timesteps) ===" << std::endl;
    std::cout << "Shape: [" << windowSize << ", 3] (roll, pitch, yaw) in radians" << std::endl;
    std::cout << unscaledPred << std::endl;
    std::cout << std::endl;

    std::cout << "=== Ground Truth (next " << windowSize << " timesteps) ===" << std::endl;
    std::cout << groundTruth << std::endl;
    std::cout << std::endl;

    // Calculate error on original scale
    Tensor error = unscaledPred - groundTruth;
    Tensor rmse = error.pow(2).mean().sqrt();
    std::cout << "=== Error Metrics ===" << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(6)
              << rmse.item<double>() << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmse.item<double>()) << " degrees" << std::endl;

    return 0;
}
