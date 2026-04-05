#include "LSTMNetwork.h"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

#include <torch/torch.h>

using torch::Tensor;

void setRandomSeeds(unsigned int seed) {
    // C++
    std::srand(seed);

    // PyTorch/LibTorch
    torch::manual_seed(seed);

    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
    }
}

int main(int argc, char* argv[]) {
    int numEpochs = 300;  // Default value
    unsigned int randomSeed = 42;  // Default seed for reproducibility

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-epochs" || arg == "--epochs") && i + 1 < argc) {
            numEpochs = std::atoi(argv[i + 1]);
            i++;  // Skip next argument
        } else if ((arg == "-seed" || arg == "--seed") && i + 1 < argc) {
            randomSeed = std::atoi(argv[i + 1]);
            i++;  // Skip next argument
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -epochs N, --epochs N   Number of training epochs (default: 300)" << std::endl;
            std::cout << "  -seed N, --seed N       Random seed for reproducibility (default: 42)" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            return 0;
        } else {
            std::cout << "Wrong argument exiting \n";
            return 0;
        }
    }

    const int lookbackWindow = 10;
    const int windowSize = 10;
    const int NUM_INPUT_FEATURES = 9;   // 9 columns in data: all_combined_reordered.csv
    const int NUM_OUTPUT_FEATURES = 3;  // Predict 3 angles (roll, pitch, yaw)
    const int hiddenStateSize = 128;

    // Set random seeds for reproducibility
    setRandomSeeds(randomSeed);

    std::cout << "LSTM Multi-Step Ahead Prediction" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Lookback window (K): " <<  lookbackWindow << " timesteps" << std::endl;
    std::cout << "  Prediction horizon (N): " << windowSize << " timesteps" << std::endl;
    std::cout << "  Input features: " << NUM_INPUT_FEATURES << std::endl;
    std::cout << "  Output features: " << NUM_OUTPUT_FEATURES << std::endl;
    std::cout << "  Epochs: " << numEpochs << std::endl;
    std::cout << "  Random seed: " << randomSeed << std::endl;
    std::cout << std::endl;

    CsvReader datasetReader("Data/Quadcopter_Datasets/all_combined_reordered.csv");
    datasetReader.read();
    datasetReader.printStats();

    Eigen::MatrixXd dataset = datasetReader.getEigenData();
    int totalSamples = dataset.rows();

    // Convert dataset to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);
    std::cout << "Converted eigen matrix to Tensor" << std::endl;
    std::cout << "Tensor shape: " << datasetTensor.sizes() << std::endl;

    // ========================
    // Flight-based data split (no data leakage)
    // ========================
    // Flight sample counts (from individual dataset_*.csv files)
    // Train: Flights 1-5 | Validation: Flight 6 | Test: Flight 7
    std::vector<int> flightSizes = {2590, 2599, 5591, 2594, 2600, 2597, 2593};
    std::vector<int> flightStartIdx(flightSizes.size());
    flightStartIdx[0] = 0;
    for (size_t i = 1; i < flightSizes.size(); i++) {
        flightStartIdx[i] = flightStartIdx[i - 1] + flightSizes[i - 1];
    }

    int trainFlightsEnd = flightStartIdx[5];  // End of flight 5 = start of flight 6
    int valFlightEnd = flightStartIdx[6];      // End of flight 6 = start of flight 7

    std::cout << "Flight-based split:" << std::endl;
    std::cout << "  Train (flights 1-5): samples 0-" << trainFlightsEnd - 1
              << " (" << trainFlightsEnd << " samples)" << std::endl;
    std::cout << "  Val   (flight 6):    samples " << trainFlightsEnd << "-" << valFlightEnd - 1
              << " (" << flightSizes[5] << " samples)" << std::endl;
    std::cout << "  Test  (flight 7):    samples " << valFlightEnd << "-" << totalSamples - 1
              << " (" << flightSizes[6] << " samples)" << std::endl;

    // ========================
    // Normalize using ONLY training data statistics (flights 1-5)
    // ========================
    Tensor trainDataTensor = datasetTensor.slice(0, 0, trainFlightsEnd);
    Tensor featureMean = trainDataTensor.mean(/*dim=*/0, /*keepdim=*/true);
    Tensor featureStd = trainDataTensor.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/true);

    // Add a tiny epsilon to std to prevent division by zero in case a column is perfectly constant
    featureStd = featureStd + 1e-8;

    // Scale the entire dataset using training statistics
    Tensor scaledDatasetTensor = (datasetTensor - featureMean) / featureStd;
    std::cout << "Data standardized using training flights statistics only." << std::endl;

    // Saved to be used later for evaluation
    Tensor targetMean = featureMean.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);
    Tensor targetStd = featureStd.slice(/*dim=*/1, /*start=*/0, /*end=*/NUM_OUTPUT_FEATURES);

    // ========================
    // Create sliding windows per flight (windows never cross flight boundaries)
    // ========================
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor anglesTensor = scaledDatasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);

    // Helper: create windows from a flight's data range [startRow, startRow + flightLen)
    auto createFlightWindows = [&](int startRow, int flightLen) {
        int numWindows = flightLen - lookbackWindow - windowSize + 1;
        Tensor Xf = torch::zeros({numWindows, lookbackWindow, NUM_INPUT_FEATURES}, options);
        Tensor yf = torch::zeros({numWindows, windowSize, NUM_OUTPUT_FEATURES}, options);
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

    std::cout << std::endl;
    std::cout << "Data Shapes (windows never cross flight boundaries)" << std::endl;
    std::cout << "X_train: " << X_train.sizes() << "  y_train: " << y_train.sizes() << std::endl;
    std::cout << "X_val:   " << X_val.sizes() << "  y_val:   " << y_val.sizes() << std::endl;
    std::cout << "X_test:  " << X_test.sizes() << "  y_test:  " << y_test.sizes() << std::endl;
    std::cout << "Train: " << numTrain << " | Val: " << numVal << " | Test: " << numTest
              << " (flight-based split, no data leakage)" << std::endl;
    std::cout << std::endl;

    // Model Creation
    int outputSize = windowSize * NUM_OUTPUT_FEATURES; // Output = how many timesteps to predict * features
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

    std::cout << "Model Created" << std::endl;
    std::cout << "Architecture: " << model->describe() << std::endl;
    std::cout << std::endl;

    // Training parameters
    const int batchSize = 64;
    const double learningRate = 0.001; // Classic value .. TODO Should we tune this ? 

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(model->parameters(), learningRate);

    std::cout << "Training Configuration" << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    std::cout << "Optimizer: Adam" << std::endl;
    std::cout << "Loss function: " << criterion->name() << std::endl;
    std::cout << std::endl;

    std::cout << "Starting Training" << std::endl;
    model->train();  // Set model to training mode

    // Vector holding one index for each training sample
    std::vector<int> indices(numTrain);
    for (int i = 0; i < numTrain; i++) {
        indices[i] = i;
    }

    std::mt19937 rng(randomSeed);

    auto start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double epochLoss = 0.0;
        int numBatches = 0;

        // Shuffle indices at the beginning of each epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        // Training per batch
        for (int batchStart = 0; batchStart < numTrain; batchStart += batchSize) {
            int currentBatchSize = std::min(batchSize, numTrain - batchStart); // Just for the last iteration no left overs ...

            // Extract batch using shuffled indices - Gather data
            std::vector<Tensor> batchXList, batchYList;
            for (int i = 0; i < currentBatchSize; i++) {
                int idx = indices[batchStart + i];
                batchXList.push_back(X_train[idx]);
                batchYList.push_back(y_train[idx]);
            }

            Tensor batchX = torch::stack(batchXList);
            Tensor batchY = torch::stack(batchYList);

            // Flatten target: [batch, N, 3] -> [batch, N*3]
            Tensor batchYFlat = batchY.reshape({currentBatchSize, -1});

            // Forward pass
            Tensor predictions = model->forward(batchX);
            Tensor loss = criterion(predictions, batchYFlat);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epochLoss += loss.item<double>();
            numBatches++;
        }

        // Print progress with validation loss
        if (epoch % 5 == 0 || epoch == numEpochs - 1) {
            double avgTrainLoss = epochLoss / numBatches;

            // Compute validation loss
            model->eval();
            torch::NoGradGuard val_no_grad;
            Tensor valPred = model->forward(X_val);
            Tensor valYFlat = y_val.reshape({numVal, -1});
            double valLoss = criterion(valPred, valYFlat).item<double>();
            model->train();

            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Train Loss: " << std::fixed << std::setprecision(6)
                      << avgTrainLoss
                      << " | Val Loss: " << valLoss << std::endl;
        }

        // Save model every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == numEpochs - 1) { // make sure to save at last epoch
            std::string modelPath = "lstm_model_epoch_" + std::to_string(epoch + 1) + ".pt";
            torch::save(model, modelPath);
            std::cout << "  -- Saved model: " << modelPath << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Training Complete" << std::endl;
    std::cout << "Elapsed(s) = " << since<std::chrono::seconds>(start).count() << std::endl;

    // EVALUATION
    model->eval();
    torch::NoGradGuard no_grad;

    evaluateSet(model, X_train, y_train, numTrain, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Training");
    evaluateSet(model, X_val, y_val, numVal, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Validation");
    evaluateSet(model, X_test, y_test, numTest, windowSize, NUM_OUTPUT_FEATURES, targetStd, targetMean, "Test");
    
    return 0;
}
