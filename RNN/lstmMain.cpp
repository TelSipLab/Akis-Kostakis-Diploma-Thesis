#include "csvreader.hpp"

#include <string>
#include <iostream>
#include <iomanip>

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
    const int lookbackWindow = 10;
    const int windowSize = 5;           // Predict next 5 timesteps
    const int NUM_INPUT_FEATURES = 9;   // 9 columns in dataset_1.csv
    const int NUM_OUTPUT_FEATURES = 3;  // Predict 3 angles (roll, pitch, yaw)

    std::cout << "=== LSTM Multi-Step Ahead Prediction ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "Lookback window (K): " <<  lookbackWindow << " timesteps" << std::endl;
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

    int trainingSamples = totalSamples - windowSize - lookbackWindow +1;  // Remove windowSize from the training sample
    std::cout << "Number of training samples: " << trainingSamples  << " after removing windowSize" << std::endl;

    // Convert dataset to tensor
    Tensor datasetTensor = LSTMNetwork::eigenToTensor(dataset);
    std::cout << "Dataset tensor shape: " << datasetTensor.sizes() << std::endl;

    // Pre-allocate X and y tensors
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    Tensor X = torch::zeros({trainingSamples, lookbackWindow, NUM_INPUT_FEATURES}, options);  // 3D for sequence input
    Tensor y = torch::zeros({trainingSamples, windowSize, NUM_OUTPUT_FEATURES}, options);

    // Angles tensor containg the ground truth values
    Tensor anglesTensor = datasetTensor.slice(1, 0, NUM_OUTPUT_FEATURES);
    std::cout << "Angles tensor shape: " << anglesTensor.sizes() << std::endl;
    // Shape: [3397, 3]

    // Create sequences
    for (int i = 0; i < trainingSamples; i++) {
        X[i] = datasetTensor.slice(0, i, i + lookbackWindow);
        
        int predStart = i + lookbackWindow;
        y[i] = anglesTensor.slice(0, predStart, predStart + windowSize);
    }

    std::cout << std::endl;
    std::cout << "=== Data Shapes ===" << std::endl;
    std::cout << "X (inputs):  " << X.sizes() << std::endl;
    std::cout << "y (targets): " << y.sizes() << std::endl;
    std::cout << std::endl;

    // VERIFY DATA
    std::cout << "=== Data Verification ===" << std::endl;
    std::cout << "First input sample (timestep 0):" << std::endl;
    std::cout << X[0] << std::endl;
    std::cout << std::endl;

    std::cout << "First target (predict timesteps 1-10, angles only):" << std::endl;
    std::cout << "Shape: " << y[0].sizes() << std::endl;
    std::cout << "First 3 predictions:" << std::endl;
    std::cout << y[0].slice(0, 0, 3) << std::endl;
    std::cout << std::endl;

    // CREATE MODEL
    int outputSize = windowSize * NUM_OUTPUT_FEATURES;
    auto model = std::make_shared<LSTMNetwork>(NUM_INPUT_FEATURES, 128, outputSize);

    std::cout << "=== Model Created ===" << std::endl;
    std::cout << "Architecture: Input(" << NUM_INPUT_FEATURES << ") -> LSTM(64) -> FC(" << outputSize << ")" << std::endl;
    std::cout << std::endl;

    // TRAINING SETUP
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

    // TRAINING LOOP
    std::cout << "=== Starting Training ===" << std::endl;
    model->train();  // Set model to training mode

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double epochLoss = 0.0;
        int numBatches = 0;

        // Traing per batch
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

        // Save model every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == numEpochs - 1) {
            std::string modelPath = "lstm_model_epoch_" + std::to_string(epoch + 1) + ".pt";
            torch::save(model, modelPath);
            std::cout << "  --> Saved model: " << modelPath << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Training Complete! ===" << std::endl;
    std::cout << std::endl;

    // EVALUATION - RMSE PER STEP
    std::cout << "=== Evaluating RMSE for each prediction step ===" << std::endl;
    model->eval();  // Set model to evaluation mode
    torch::NoGradGuard no_grad;

    // Get predictions for all samples
    Tensor allPredictions = model->forward(X);  // [3387, 30]
    allPredictions = allPredictions.view({trainingSamples, windowSize, NUM_OUTPUT_FEATURES});  // [3387, 10, 3]

    // Calculate overall metrics
    Tensor allDiff = allPredictions - y;
    Tensor overallRMSE = allDiff.pow(2).mean().sqrt();
    Tensor overallMAE = allDiff.abs().mean();

    // RMSE per angle (across ALL samples and ALL steps) - for EKF comparison
    std::vector<int64_t> dims = {0, 1};  // Average over samples and steps
    Tensor rmsePerAngle = allDiff.pow(2).mean(dims).sqrt();
    auto rmse_acc = rmsePerAngle.accessor<double, 1>();

    std::cout << "Overall Metrics:" << std::endl;
    std::cout << "  RMSE (all): " << std::fixed << std::setprecision(6)
              << overallRMSE.item<double>() << " rad = "
              << std::setprecision(3) << overallRMSE.item<double>() * 180.0 / M_PI << " deg" << std::endl;
    std::cout << "  MAE  (all): " << std::fixed << std::setprecision(6)
              << overallMAE.item<double>() << " rad = "
              << std::setprecision(3) << overallMAE.item<double>() * 180.0 / M_PI << " deg" << std::endl;
    std::cout << std::endl;

    std::cout << "RMSE per angle (all samples, all steps):" << std::endl;
    std::cout << "  Roll  RMSE: " << std::setprecision(6) << rmse_acc[0] << " rad = "
              << std::setprecision(3) << rmse_acc[0] * 180.0 / M_PI << " deg " << std::endl;
    std::cout << "  Pitch RMSE: " << std::setprecision(6) << rmse_acc[1] << " rad = "
              << std::setprecision(3) << rmse_acc[1] * 180.0 / M_PI << " deg " << std::endl;
    std::cout << "  Yaw   RMSE: " << std::setprecision(6) << rmse_acc[2] << " rad = "
              << std::setprecision(3) << rmse_acc[2] * 180.0 / M_PI << " deg" << std::endl;
    std::cout << std::endl;

    // RMSE per step
    std::cout << "=== RMSE per step ===" << std::endl;
    std::cout << "Step | Roll (rad) | Pitch (rad) | Yaw (rad)" << std::endl;
    std::cout << "-----+------------+-------------+-----------" << std::endl;
    for (int step = 0; step < windowSize; step++) {
        Tensor stepPred = allPredictions.select(1, step);
        Tensor stepTarget = y.select(1, step);
        Tensor rmse = (stepPred - stepTarget).pow(2).mean(0).sqrt();
        auto acc = rmse.accessor<double, 1>();
        std::cout << std::setw(4) << (step + 1) << " | "
                  << std::fixed << std::setprecision(6)
                  << std::setw(10) << acc[0] << " | "
                  << std::setw(11) << acc[1] << " | "
                  << std::setw(9) << acc[2] << std::endl;
    }
    std::cout << std::endl;

    // // MAE per step
    // std::cout << "=== MAE per step ===" << std::endl;
    // std::cout << "Step | Roll (rad) | Pitch (rad) | Yaw (rad)" << std::endl;
    // std::cout << "-----+------------+-------------+-----------" << std::endl;
    // for (int step = 0; step < windowSize; step++) {
    //     Tensor stepPred = allPredictions.select(1, step);
    //     Tensor stepTarget = y.select(1, step);
    //     Tensor mae = (stepPred - stepTarget).abs().mean(0);
    //     auto acc = mae.accessor<double, 1>();
    //     std::cout << std::setw(4) << (step + 1) << " | "
    //               << std::fixed << std::setprecision(6)
    //               << std::setw(10) << acc[0] << " | "
    //               << std::setw(11) << acc[1] << " | "
    //               << std::setw(9) << acc[2] << std::endl;
    // }
    // std::cout << std::endl;

    // // R² per step
    // std::cout << "=== R² per step ===" << std::endl;
    // std::cout << "Step | Roll       | Pitch      | Yaw" << std::endl;
    // std::cout << "-----+------------+------------+-----------" << std::endl;
    // for (int step = 0; step < windowSize; step++) {
    //     Tensor stepPred = allPredictions.select(1, step);
    //     Tensor stepTarget = y.select(1, step);
    //     Tensor ss_res = (stepPred - stepTarget).pow(2).sum(0);
    //     Tensor ss_tot = (stepTarget - stepTarget.mean(0)).pow(2).sum(0);
    //     Tensor r2 = 1.0 - (ss_res / ss_tot);
    //     auto acc = r2.accessor<double, 1>();
    //     std::cout << std::setw(4) << (step + 1) << " | "
    //               << std::fixed << std::setprecision(6)
    //               << std::setw(10) << acc[0] << " | "
    //               << std::setw(10) << acc[1] << " | "
    //               << std::setw(9) << acc[2] << std::endl;
    // }
    // std::cout << std::endl;

    // // TESTING ON FIRST SAMPLE
    // std::cout << "=== Testing Model on First Sample ===" << std::endl;

    // // Test on first sample
    // Tensor testInput = X[0].unsqueeze(0);  // [1, 9]
    // Tensor testOutput = model->forward(testInput);
    // Tensor testPrediction = testOutput.view({windowSize, NUM_OUTPUT_FEATURES});

    // std::cout << "Input (timestep 0):" << std::endl;
    // std::cout << testInput << std::endl;
    // std::cout << std::endl;

    // std::cout << "Predicted angles for next 10 timesteps:" << std::endl;
    // std::cout << testPrediction << std::endl;
    // std::cout << std::endl;

    // std::cout << "Target (ground truth):" << std::endl;
    // std::cout << y[0] << std::endl;

    return 0;
}
