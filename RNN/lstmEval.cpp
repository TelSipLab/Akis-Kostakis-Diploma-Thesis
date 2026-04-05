#include "Utils.hpp"
#include "csvreader.hpp"
#ifdef USE_NO_ATTENTION
#include "LSTMNetworkNoAttention.h"
#else
#include "LSTMNetwork.h"
#endif

#include <exception>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <torch/torch.h>

using torch::Tensor;

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

int printHelpMessageAndExit() {
    std::cout << "Usage: ./lstmEval.out <model_path> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  [sample_index]       Evaluate single sample (e.g., 0, 1000)" << std::endl;
    std::cout << "  --save-all, -a       Generate predictions for ALL samples and save to CSV" << std::endl;
    std::cout << "  --window N, -w N     Prediction horizon (default: 10)" << std::endl;
    std::cout << "  --help, -h           Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt 0" << std::endl;
    std::cout << "  ./lstmEval.out lstm_model_epoch_1000.pt --save-all" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    bool saveAll = false;
    std::string modelPath;
    int sampleIndex = 0;
    int windowSize = 10;  // Default, override with --window N

    if(argc == 1) {
        std::cout << "Wrong input use -h to see help" << std::endl;
        return -1;
    }

    // Check if first argument is a help flag
    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        return printHelpMessageAndExit();
    }

    modelPath = firstArg;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--save-all" || arg == "-a" ) {
            saveAll = true;
        } else if((arg == "--window" || arg == "-w") && i + 1 < argc) {
            windowSize = std::stoi(argv[i + 1]);
            i++;
        } else if(i == 2 && isNumber(arg)) {
            sampleIndex = std::stoi(arg);
        } else {
            std::cout << "Wrong argument use -h to view help" << std::endl;
            return -1;
        }
    }

    // Model Configuration
    const int lookbackWindow = 10;  // Must match training
    const int NUM_INPUT_FEATURES = 9;
    const int NUM_OUTPUT_FEATURES = 3;
    const int outputSize = windowSize * NUM_OUTPUT_FEATURES;
    const int hiddenStateSize = 128;
    // randomSeed no longer needed — flight-based split is deterministic

    std::cout << "LSTM Model Evaluation" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    if (saveAll) {
        std::cout << "Mode: Generate predictions for ALL samples" << std::endl;
    } else {
        std::cout << "Mode: Single sample evaluation (index: " << sampleIndex << ")" << std::endl;
    }
    std::cout << "Lookback window: " << lookbackWindow << " timesteps" << std::endl;
    std::cout << "Prediction horizon: " << windowSize << " timesteps" << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model..." << std::endl;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

    try {
        torch::load(model, modelPath);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Load CSV
    CsvReader datasetReader("Data/Quadcopter_Datasets/all_combined_reordered.csv");
    datasetReader.read();
    Eigen::MatrixXd dataset = datasetReader.getEigenData();

    // Convert to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);

    // ========================
    // Flight-based data split (must match training!)
    // ========================
    std::vector<int> flightSizes = {2590, 2599, 5591, 2594, 2600, 2597, 2593};
    std::vector<int> flightStartIdx(flightSizes.size());
    flightStartIdx[0] = 0;
    for (size_t i = 1; i < flightSizes.size(); i++) {
        flightStartIdx[i] = flightStartIdx[i - 1] + flightSizes[i - 1];
    }

    int trainFlightsEnd = flightStartIdx[5];

    // Normalize using ONLY training data statistics (flights 1-5) - must match training!
    Tensor trainDataTensor = datasetTensor.slice(0, 0, trainFlightsEnd);
    Tensor featureMean = trainDataTensor.mean(/*dim=*/0, /*keepdim=*/true);
    Tensor featureStd = trainDataTensor.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/true);
    featureStd = featureStd + 1e-8;

    Tensor scaledDatasetTensor = (datasetTensor - featureMean) / featureStd;

    Tensor targetMean = featureMean.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);
    Tensor targetStd = featureStd.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);

    // Create sliding windows per flight (windows never cross flight boundaries)
    auto options_w = torch::TensorOptions().dtype(torch::kDouble);
    Tensor anglesTensor = scaledDatasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);

    auto createFlightWindows = [&](int startRow, int flightLen) {
        int numWindows = flightLen - lookbackWindow - windowSize + 1;
        Tensor Xf = torch::zeros({numWindows, lookbackWindow, NUM_INPUT_FEATURES}, options_w);
        Tensor yf = torch::zeros({numWindows, windowSize, NUM_OUTPUT_FEATURES}, options_w);
        for (int i = 0; i < numWindows; i++) {
            int rowIdx = startRow + i;
            Xf[i] = scaledDatasetTensor.slice(0, rowIdx, rowIdx + lookbackWindow);
            int predStart = rowIdx + lookbackWindow;
            yf[i] = anglesTensor.slice(0, predStart, predStart + windowSize);
        }
        return std::make_pair(Xf, yf);
    };

    // Build training windows from flights 1-5
    std::vector<Tensor> trainXList, trainYList;
    for (int f = 0; f < 5; f++) {
        auto [Xf, yf] = createFlightWindows(flightStartIdx[f], flightSizes[f]);
        trainXList.push_back(Xf);
        trainYList.push_back(yf);
    }
    Tensor X_train = torch::cat(trainXList, 0);
    Tensor y_train = torch::cat(trainYList, 0);

    // Validation windows from flight 6
    auto [X_val, y_val] = createFlightWindows(flightStartIdx[5], flightSizes[5]);

    // Test windows from flight 7
    auto [X_test, y_test] = createFlightWindows(flightStartIdx[6], flightSizes[6]);

    int numTrain = X_train.size(0);
    int numVal = X_val.size(0);
    int numTest = X_test.size(0);
    int trainingSamples = numTrain + numVal + numTest;

    std::cout << "Data normalized (z-score, training flights only)" << std::endl;
    std::cout << "Train: " << numTrain << " | Val: " << numVal << " | Test: " << numTest
              << " (flight-based split, no data leakage)" << std::endl;
    std::cout << std::endl;

    // Evaluate all splits
    model->eval();
    {
        torch::NoGradGuard no_grad;
        evaluateSet(model, X_train, y_train, numTrain, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Training");
        evaluateSet(model, X_val, y_val, numVal, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Validation");
        evaluateSet(model, X_test, y_test, numTest, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Test");
    }

    // Concatenate all splits for indexed access (train, then val, then test - chronological)
    Tensor X_all = torch::cat({X_train, X_val, X_test}, 0);
    Tensor y_all = torch::cat({y_train, y_val, y_test}, 0);

    // Handle --save-all mode
    if (saveAll) {
        std::cout << "Generating predictions for " << trainingSamples << " samples..." << std::endl;

        model->eval();
        torch::NoGradGuard no_grad;

        // Open output file
        std::ofstream outFile("Results/lstm_predictions.csv");
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open Results/lstm_predictions.csv for writing" << std::endl;
            return 1;
        }

        // Write header
        outFile << "sample,step_ahead,roll_pred,pitch_pred,yaw_pred,roll_gt,pitch_gt,yaw_gt,set" << std::endl;

        // Process all samples (ordered: train, val, test)
        for (int sample = 0; sample < trainingSamples; sample++) {
            Tensor input = X_all[sample].unsqueeze(0);
            Tensor output = model->forward(input);
            Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

            // Denormalize predictions and targets back to radians
            Tensor unscaledPred = ((prediction * targetStd) + targetMean).squeeze(0);
            Tensor unscaledTarget = (y_all[sample] * targetStd) + targetMean;

            // Determine set label
            std::string setLabel;
            if (sample < numTrain) {
                setLabel = "train";
            } else if (sample < numTrain + numVal) {
                setLabel = "val";
            } else {
                setLabel = "test";
            }

            // Write predictions for each step
            for (int step = 0; step < windowSize; step++) {
                int stepAhead = step + 1;

                auto pred = unscaledPred[step];
                double roll_pred = pred[0].item<double>();
                double pitch_pred = pred[1].item<double>();
                double yaw_pred = pred[2].item<double>();

                auto gt = unscaledTarget[step];
                double roll_gt = gt[0].item<double>();
                double pitch_gt = gt[1].item<double>();
                double yaw_gt = gt[2].item<double>();

                outFile << sample << "," << stepAhead << ","
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

    // Determine which set this sample belongs to
    std::string setLabel;
    if (sampleIndex < numTrain) {
        setLabel = "train";
    } else if (sampleIndex < numTrain + numVal) {
        setLabel = "val";
    } else {
        setLabel = "test";
    }

    std::cout << "Sample " << sampleIndex << " belongs to: " << setLabel << " set" << std::endl;
    std::cout << std::endl;

    // Run prediction
    model->eval();
    torch::NoGradGuard no_grad;

    Tensor input = X_all[sampleIndex].unsqueeze(0);
    Tensor output = model->forward(input);
    Tensor prediction = output.view({1, windowSize, NUM_OUTPUT_FEATURES});

    // Denormalize predictions and targets back to radians
    Tensor unscaledPred = ((prediction * targetStd) + targetMean).squeeze(0);
    Tensor groundTruth = (y_all[sampleIndex] * targetStd) + targetMean;

    // Display results
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
