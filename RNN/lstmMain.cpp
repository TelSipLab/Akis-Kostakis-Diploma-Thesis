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

// Print evaluation metrics in a formatted way
void printMetrics(double rmseValue, double maeValue,
                  const std::vector<double>& rmsePerAngle,
                  const std::vector<std::vector<double>>& rmsePerStep,
                  int windowSize) {
    std::cout << "Overall Metrics:" << std::endl;
    std::cout << "  RMSE (all): " << std::fixed << std::setprecision(6)
              << rmseValue << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmseValue) << " deg" << std::endl;
    std::cout << "  MAE  (all): " << std::fixed << std::setprecision(6)
              << maeValue << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(maeValue) << " deg" << std::endl;
    std::cout << std::endl;

    std::cout << "RMSE per angle (all samples, all steps):" << std::endl;
    std::cout << "  Roll  RMSE: " << std::setprecision(6) << rmsePerAngle[0] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[0])<< " deg " << std::endl;
    std::cout << "  Pitch RMSE: " << std::setprecision(6) << rmsePerAngle[1] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[1]) << " deg " << std::endl;
    std::cout << "  Yaw   RMSE: " << std::setprecision(6) << rmsePerAngle[2] << " rad = "
              << std::setprecision(3) << Utils::convertToDeg(rmsePerAngle[2]) << " deg" << std::endl;
    std::cout << std::endl;

    // RMSE per step
    std::cout << "RMSE per step" << std::endl;
    std::cout << "Step | Roll (deg) | Pitch (deg) | Yaw (deg)" << std::endl;
    std::cout << "-----+------------+-------------+-----------" << std::endl;
    for (int step = 0; step < windowSize; step++) {
        std::cout << std::setw(4) << (step + 1) << " | "
                  << std::fixed << std::setprecision(6)
                  << std::setw(10) << Utils::convertToDeg(rmsePerStep[step][0]) << " | "
                  << std::setw(11) << Utils::convertToDeg(rmsePerStep[step][1]) << " | "
                  << std::setw(9) << Utils::convertToDeg(rmsePerStep[step][2]) << std::endl;
    }
    std::cout << std::endl;
}

// TODO Move out of here if network get's too complex
class LSTMNetwork: public torch::nn::Module {
public:
    LSTMNetwork(int inputSize, int hiddenStateSize, int outputSize) {
        // 1. Standard LSTM layer
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(inputSize, hiddenStateSize).batch_first(true)
        ));
    

        //2. Attention Projection: Learns a non-linear representation of the higgen states
        attn_linear = register_module("attn_linear", torch::nn::Linear(hiddenStateSize, hiddenStateSize));
        
        // 3. Attention Scoring: Reduces the project representation to a scalar score per timestemp
        attn_vector = register_module("attn_vector", torch::nn::Linear(hiddenStateSize, 1));

        //4. Fully Connected Output Layer
        fc = register_module("fc", torch::nn::Linear(hiddenStateSize, outputSize));

        this->to(torch::kDouble);
    }

    Tensor forward(Tensor X) {
        // Step 1: Pass through LSTM
        // hiddenStates shape: [batch, seq_len, hidden_size]
        auto lstmOutput = lstm->forward(X);
        auto hiddenStates = std::get<0>(lstmOutput);

        // Step 2: Calculate "Energy"
        // Let the model learn complex temporal allignments
        auto energy = torch::tanh(attn_linear->forward(hiddenStates)); 

        // Step 3: Compute raw attention scores
        // attn_vector reduces hidden dimension to 1 -> [batch, seq_len, 1]
        // squeeze(-1) removes the trailing dimension -> [batch, seq_len]
        auto scores = attn_vector->forward(energy).squeeze(-1);

        // Step 4: Normalize scores into probabilities via softmax
        // Weights will sum to 1.0
        auto weights = torch::softmax(scores, /*dim=*/1);

        // Step 5: Construct the Context Vector
        // weights.unsqueeze(1) -> [batch, 1, seq_len]
        // Batch Matrix Multiplication (bmm) computes the weighted sum: context = sum(weights[t] * hiddenStates[t])
        // Result shape: [batch, hidden_size]
        auto context = torch::bmm(weights.unsqueeze(1), hiddenStates).squeeze(1); // [batch, hidden]

        // Step 6: Pass through FC layer
        auto out = fc->forward(context);  // [batch, hidden] → [batch, output_size]
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
            tensor[i] = rowTensor;
        }
        return tensor;
    }
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear attn_linear{nullptr};
    torch::nn::Linear attn_vector{nullptr};
    torch::nn::Linear fc{nullptr};
};

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    int numEpochs = 300;  // Default value
    unsigned int randomSeed = 42;  // Default seed for reproducibility

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
        }
    }

    // Set random seeds for reproducibility
    setRandomSeeds(randomSeed);

    const int lookbackWindow = 10;
    const int windowSize = 10;           // Predict next 10 timesteps
    const int NUM_INPUT_FEATURES = 9;   // 9 columns in dataset_1.csv
    const int NUM_OUTPUT_FEATURES = 3;  // Predict 3 angles (roll, pitch, yaw)

    std::cout << "LSTM Multi-Step Ahead Prediction" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Lookback window (K): " <<  lookbackWindow << " timesteps" << std::endl;
    std::cout << "  Prediction horizon (N): " << windowSize << " timesteps" << std::endl;
    std::cout << "  Input features: " << NUM_INPUT_FEATURES << std::endl;
    std::cout << "  Output features: " << NUM_OUTPUT_FEATURES << std::endl;
    std::cout << "  Epochs: " << numEpochs << std::endl;
    std::cout << "  Random seed: " << randomSeed << std::endl;
    std::cout << std::endl;

    CsvReader datasetReader("Data/dataset_1.csv");
    datasetReader.read();
    datasetReader.printStats();

    Eigen::MatrixXd dataset = datasetReader.getEigenData();
    int totalSamples = dataset.rows();

    int trainingSamples = totalSamples - windowSize - lookbackWindow +1;  // Remove windowSize from the training sample
    std::cout << "Number of training samples: " << trainingSamples  << " after removing windowSize and loopback" << std::endl;

    // Convert dataset to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);
    std::cout << "Converted eigen matrix to Tensor" << std::endl;
    std::cout << "Tensor shape: " << datasetTensor.sizes() << std::endl;

    // Pre-allocate X and y tensors
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor X = torch::zeros({trainingSamples, lookbackWindow, NUM_INPUT_FEATURES}, options);
    Tensor y = torch::zeros({trainingSamples, windowSize, NUM_OUTPUT_FEATURES}, options);

    // Angles tensor containg the ground truth values
    Tensor anglesTensor = datasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);
    std::cout << "Angles tensor shape: " << anglesTensor.sizes() << std::endl;
    // [3397, 3]

    // Create sequences
    for (int i = 0; i < trainingSamples; i++) {
        X[i] = datasetTensor.slice(0, i, i + lookbackWindow);
        
        int predStart = i + lookbackWindow;
        y[i] = anglesTensor.slice(0, predStart, predStart + windowSize);
    }

    std::cout << std::endl;
    std::cout << "Data Shapes" << std::endl;
    std::cout << "X (inputs):  " << X.sizes() << std::endl;
    std::cout << "y (targets): " << y.sizes() << std::endl;
    std::cout << std::endl;

    // Model Creation
    int outputSize = windowSize * NUM_OUTPUT_FEATURES; // Output = how many timesteps to predict * features
    int hiddenStateSize = 128;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, hiddenStateSize, outputSize);

    std::cout << "Model Created" << std::endl;
    std::cout << "Architecture: Input(" << NUM_INPUT_FEATURES << ") -> LSTM(" << hiddenStateSize << ") -> Attention -> FC(" << outputSize << ")" << std::endl;
    std::cout << std::endl;

    // Training parameters
    const int batchSize = 32;
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

    // Vector holding one index for each sample
    std::vector<int> indices(trainingSamples);
    for (int i = 0; i < trainingSamples; i++) {
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
        for (int batchStart = 0; batchStart < trainingSamples; batchStart += batchSize) {
            int currentBatchSize = std::min(batchSize, trainingSamples - batchStart);

            // Extract batch using shuffled indices
            std::vector<Tensor> batchXList, batchYList;
            for (int i = 0; i < currentBatchSize; i++) {
                int idx = indices[batchStart + i];
                batchXList.push_back(X[idx]);
                batchYList.push_back(y[idx]);
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

        // Print progress
        if (epoch % 5 == 0 || epoch == numEpochs - 1) {
            double avgLoss = epochLoss / numBatches;
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << avgLoss << std::endl;
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

    // EVALUATION - RMSE PER STEP
    std::cout << "Evaluating RMSE for each prediction step" << std::endl;
    model->eval();  // Set model to evaluation mode
    torch::NoGradGuard no_grad;

    // Get predictions for all samples
    Tensor allPredictions = model->forward(X);
    allPredictions = allPredictions.view({trainingSamples, windowSize, NUM_OUTPUT_FEATURES});
    std::cout << "Calculated all predictions size: " <<  allPredictions.sizes() << std::endl;

    // Calculate overall metrics
    Tensor allDiff = allPredictions - y;

    Tensor overallRMSE = allDiff.pow(2).mean().sqrt();
    const double rmseValue = overallRMSE.item<double>();
    
    Tensor overallMAE = allDiff.abs().mean();
    const double maeValue = overallMAE.item<double>();

    // RMSE per angle (across ALL samples and ALL steps) - for EKF comparison
    std::vector<int64_t> dims = {0, 1};  // Average over samples and steps
    Tensor rmsePerAngleTensor = allDiff.pow(2).mean(dims).sqrt();
    auto rmse_acc = rmsePerAngleTensor.accessor<double, 1>();

    // Extract RMSE per angle into vector
    std::vector<double> rmsePerAngle = {rmse_acc[0], rmse_acc[1], rmse_acc[2]};

    // Calculate RMSE per step
    std::vector<std::vector<double>> rmsePerStep;
    for (int step = 0; step < windowSize; step++) {
        Tensor stepPred = allPredictions.select(1, step);
        Tensor stepTarget = y.select(1, step);
        Tensor rmse = (stepPred - stepTarget).pow(2).mean(0).sqrt();

        auto acc = rmse.accessor<double, 1>();
        rmsePerStep.push_back({acc[0], acc[1], acc[2]});
    }

    // Print all metrics
    printMetrics(rmseValue, maeValue, rmsePerAngle, rmsePerStep, windowSize);
    
    return 0;
}
