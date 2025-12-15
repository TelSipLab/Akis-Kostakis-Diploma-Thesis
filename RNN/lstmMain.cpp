#include <string>
#include <iostream>
#include <iomanip>
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
    const int windowSize = 5;  // Predict next 10 timesteps
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

    // ==========================================
    // CREATE MODEL
    // ==========================================
    int outputSize = windowSize * NUM_OUTPUT_FEATURES;  // 10 * 3 = 30
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, 128, outputSize);

    std::cout << "=== Model Created ===" << std::endl;
    std::cout << "Architecture: Input(" << NUM_INPUT_FEATURES << ") -> LSTM(64) -> FC(" << outputSize << ")" << std::endl;
    std::cout << std::endl;

    // ==========================================
    // TRAINING SETUP
    // ==========================================
    const int batchSize = 32;
    const int numEpochs = 300;
    const double learningRate = 0.001;

    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::Adam(model->parameters(), learningRate);

    std::cout << "=== Training Configuration ===" << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    std::cout << "Optimizer: Adam" << std::endl;
    std::cout << "Loss function: MSE" << std::endl;
    std::cout << std::endl;

    // ==========================================
    // TRAINING LOOP
    // ==========================================
    std::cout << "=== Starting Training ===" << std::endl;
    model->train();  // Set model to training mode

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double epochLoss = 0.0;
        int numBatches = 0;

        // Batch processing
        for (int i = 0; i < trainingSamples; i += batchSize) {
            int currentBatchSize = std::min(batchSize, trainingSamples - i);

            // Extract batch
            Tensor batchX = X.slice(0, i, i + currentBatchSize);
            Tensor batchY = y.slice(0, i, i + currentBatchSize);

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

        // Print progress every 10 epochs
        if (epoch % 10 == 0 || epoch == numEpochs - 1) {
            double avgLoss = epochLoss / numBatches;
            std::cout << "Epoch " << std::setw(3) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << avgLoss << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Training Complete! ===" << std::endl;
    std::cout << std::endl;

    // ==========================================
    // EVALUATION - RMSE PER STEP
    // ==========================================
    std::cout << "=== Evaluating RMSE for each prediction step ===" << std::endl;
    model->eval();  // Set model to evaluation mode
    torch::NoGradGuard no_grad;

    // Get predictions for all samples
    Tensor allPredictions = model->forward(X);  // [3387, 30]
    allPredictions = allPredictions.view({trainingSamples, windowSize, NUM_OUTPUT_FEATURES});  // [3387, 10, 3]

    // Calculate overall RMSE first
    Tensor allDiff = allPredictions - y;
    Tensor allSquaredDiff = allDiff.pow(2);
    Tensor overallMSE = allSquaredDiff.mean();
    Tensor overallRMSE = overallMSE.sqrt();

    std::cout << "Overall RMSE (all steps, all angles): "
              << std::fixed << std::setprecision(6)
              << overallRMSE.item<double>() << " rad = "
              << std::setprecision(3)
              << overallRMSE.item<double>() * 180.0 / M_PI << " degrees" << std::endl;
    std::cout << std::endl;

    std::cout << "Step | Roll (rad) | Pitch (rad) | Yaw (rad)" << std::endl;
    std::cout << "-----+------------+-------------+-----------" << std::endl;

    for (int step = 0; step < windowSize; step++) {
        // Extract predictions and targets for this step
        Tensor stepPred = allPredictions.select(1, step);  // [3387, 3]
        Tensor stepTarget = y.select(1, step);             // [3387, 3]

        // Calculate RMSE for each angle
        Tensor diff = stepPred - stepTarget;
        Tensor squaredDiff = diff.pow(2);
        Tensor mse = squaredDiff.mean(0);  // Mean over samples, keep 3 angles
        Tensor rmse = mse.sqrt();

        auto rmse_accessor = rmse.accessor<double, 1>();

        std::cout << std::setw(4) << (step + 1) << " | "
                  << std::fixed << std::setprecision(6)
                  << std::setw(10) << rmse_accessor[0] << " | "
                  << std::setw(11) << rmse_accessor[1] << " | "
                  << std::setw(9) << rmse_accessor[2] << std::endl;
    }

    std::cout << std::endl;

    // ==========================================
    // TESTING ON FIRST SAMPLE
    // ==========================================
    std::cout << "=== Testing Model on First Sample ===" << std::endl;

    // Test on first sample
    Tensor testInput = X[0].unsqueeze(0);  // [1, 9]
    Tensor testOutput = model->forward(testInput);
    Tensor testPrediction = testOutput.view({windowSize, NUM_OUTPUT_FEATURES});

    std::cout << "Input (timestep 0):" << std::endl;
    std::cout << testInput << std::endl;
    std::cout << std::endl;

    std::cout << "Predicted angles for next 10 timesteps:" << std::endl;
    std::cout << testPrediction << std::endl;
    std::cout << std::endl;

    std::cout << "Target (ground truth):" << std::endl;
    std::cout << y[0] << std::endl;

    return 0;
}
