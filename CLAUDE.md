# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++14 diploma thesis project implementing and comparing attitude estimation algorithms for UAV navigation using IMU sensor fusion. The project estimates roll and pitch angles from 6-DOF IMU data (gyroscope + accelerometer) and evaluates performance against ground truth measurements.

**Implemented Algorithms:**
- Complementary Filter: Basic weighted sensor fusion baseline
- Mahony Filter: Nonlinear complementary filter on SO(3) - best classical filter
- Explicit Complementary Filter: Vector-based filter with bias estimation
- Extended Kalman Filter (EKF): Quaternion-based probabilistic state estimation
- **LSTM Multi-Step Ahead Predictor** ✅: Deep learning approach using LibTorch (C++) - **best overall performance for pitch (0.463°)**

## Build Commands

**Dependencies:**
- g++ with C++14 support
- Eigen3 version 3.4.1 (must be in system include paths)
- Python 3 with matplotlib (for visualization)

**Build:**
```bash
# Build all filters
make all -j

# Build individual filters
make mahonyFilter        # bin/mahony.out
make complemntaryFilter  # bin/complmentary.out (note spelling)
make ekfFilter           # bin/ekf.out
make explicitComplementaryFilter  # bin/explicitCF.out

# Clean
make clean
```

**Run Filters:**
```bash
./bin/mahony.out          # Recommended - best current performance
./bin/complmentary.out
./bin/ekf.out
./bin/explicitCF.out
```

**Visualize Results:**
```bash
cd Results
python3 plotDiagrams.py           # General plotting
python3 plotDiagrams_zoomed.py    # Zoomed-in analysis
python3 analyze_dynamics.py       # Gyro/Accel magnitude vs Error
```

**LSTM Training and Evaluation:**
```bash
cd RNN

# Train LSTM model (requires LibTorch)
make lstm
./lstm.out                # Train with default 300 epochs
./lstm.out -epochs 1000   # Train for 1000 epochs (recommended)

# Evaluate and export predictions
make lstmEval
./lstmEval.out lstm_model_epoch_1000.pt --save-all  # Generate predictions CSV

# Python visualization (requires Results/lstm_predictions.csv)
python3 plot_multistep_predictions.py 0 50      # Multi-step comparison plot
python3 plot_predictions.py 0 100               # Single-step prediction plot
python3 plot_single_pred.py 50                  # Single sample analysis
python3 plot_rmse_errors.py                     # Training loss curve

# Learning examples (optional - for understanding LSTM concepts)
# Do not use them directly just here for reference
cd Python
python3 hello_rnn.py      # Basic RNN example
python3 hello_lstm.py     # Basic LSTM example
```

## Code Architecture

### Filter API Pattern

All filters follow a consistent interface pattern:

1. **Constructor**: Takes algorithm parameters (dt, gain coefficients, etc.)
2. **setIMUData()**: Load gyroscope and accelerometer matrices
3. **predictForAllData()**: Process all timesteps (or for complementary: `calculateRoll()`, `calculatePitch()`)
4. **getRollEstimation() / getPitchEstimation()**: Retrieve results as Eigen::VectorXd

**Example usage pattern** (see `mahonyFilterMain.cpp`):
```cpp
MahonyFilter mahony(dt, kp);
mahony.setIMUData(gyroMeasurements, accelMeasurements);
mahony.predictForAllData();
Eigen::VectorXd roll = mahony.getRollEstimation();
Eigen::VectorXd pitch = mahony.getPitchEstimation();
```

### Core Components

- **`include/pch.h`**: Precompiled header with Eigen and common includes
- **`include/CsvReader.hpp`**: Reads CSV files into Eigen matrices
- **`include/Utils.hpp`**: Utility functions for:
  - Angle conversions (rad/deg)
  - RMSE and MAE metrics
  - Skew-symmetric matrix operations (Mahony filter math)
  - Euler angle extraction from accelerometer (`calculateEulerRollFromSensor`, `calculateEulerPitchFromInput`)
  - Rotation matrix construction
- **Filter headers**: Define state variables, update equations, and prediction methods
- **`*FilterMain.cpp`**: Entry points that load data, run filters, compute metrics, and save results

### Data Flow

#### Classical Filters Data

1. **Input**: CSV files in `Data/` directory
   - `gyro.csv`: 3-axis angular velocity (rad/s)
   - `accel.csv`: 3-axis linear acceleration (m/s²)
   - `angles.csv`: Ground truth roll/pitch (radians)

2. **Processing**: Filters read all data upfront, then process timestep-by-timestep

3. **Output**: Results written to `Results/Results/` as text files
   - Format: One angle value per line (in degrees)
   - Naming: `{FilterName}Roll_*.txt`, `{FilterName}Pitch_*.txt`

4. **Evaluation**: Python scripts read results and ground truth from `Results/ExpectedResults/expected_{roll|pitch}.txt`

#### LSTM Training Data

**New Dataset:** `Data/dataset_1.csv` (received December 2025)

- **Size**: 3397 samples
- **Sampling rate**: Ts = 0.03 sec (33.3 Hz)
- **Columns** (9 total):
  - **Columns 0-2**: Ground truth angles (roll, pitch, yaw) - **radians** (confirmed by professor)
  - **Columns 3-5**: Gyroscope angular velocities (ω_roll, ω_pitch, ω_yaw) - **rad/s** (confirmed by professor)
    - These are rotation rates AROUND the roll/pitch/yaw axes, NOT angle measurements
    - ω_roll = d(roll)/dt, ω_pitch = d(pitch)/dt, ω_yaw = d(yaw)/dt
  - **Columns 6-8**: Control torques (τ_roll, τ_pitch, τ_yaw) - from PID controller, NOT accelerometer

**Data Flow:**
- Each row represents a complete state at one timestep
- Used for multi-step ahead prediction (predict next N timesteps)
- Ground truth angles serve as both input features AND prediction targets

### Mathematical Utilities

The `Utils` class provides critical operations for attitude estimation:

- **Skew-symmetric operations** (`skewMatrixFromVector`, `vexFromSkewMatrix`): Convert between SO(3) vectors and skew-symmetric matrices (essential for Mahony filter)
- **Accelerometer-based Euler angles**: Extract tilt angles from gravity vector when stationary
- **Rotation matrices**: Convert between Euler angles and rotation matrix representation

## Development Conventions

- **Linear Algebra**: Always use Eigen for vector/matrix operations. Avoid manual loops for mathematical computations.
- **Precision**: Use `double` for all floating-point calculations.
- **Memory Management**:
  - Pre-allocate all Eigen matrices before processing loops
  - Never use `new`/`malloc` inside tight loops
  - Filters store predictions as `Eigen::VectorXd` sized to number of samples
- **Verification**: All algorithm changes must be tested against ground truth in `Data/angles.csv`. RMSE/MAE metrics are printed to stdout.
- **Output Format**: Results saved in degrees (ground truth loaded as radians, then converted)

## Filter-Specific Notes

**Mahony Filter:**
- Best tuned gain: `kp = 11` (optimized via grid search, see `mahonyFilterMain.cpp:15`)
- Uses rotation matrix representation (SO(3)) with orthonormalization to prevent drift
- Implements passive complementary filter from Mahony et al. 2007 paper
- Correction term based on cross product between measured and predicted gravity

**Extended Kalman Filter:**
- State vector: [q0, q1, q2, q3, bx, by, bz]^T (quaternion + gyro bias)
- Quaternion normalization required after each predict step
- Measurement model: Compares measured acceleration with gravity rotated by quaternion

**Complementary Filter:**
- Simple weighted average: `alpha * gyro_integration + (1-alpha) * accel_measurement`
- Default alpha = 0.9 balances noise rejection vs. response time

## LSTM Multi-Step Ahead Prediction (Implemented ✅)

**Status:** Implementation complete - January 2026

**Approach:** Sequence-to-sequence prediction using LSTM with lookback window

**Implementation Language:** C++ using LibTorch (PyTorch C++ API)

### LSTM Architecture (Final Implementation)

**Network Structure:**
```
Input(9) → LSTM(hidden_size=128) → FC(30) → Reshape to [10, 3]
```

**Key Parameters:**
- **Lookback window (K)**: 10 timesteps (0.30s of history)
- **Prediction horizon (N)**: 10 timesteps (0.30s into future)
- **Hidden state size**: 128 (optimized from 64)
- **Batch size**: 32
- **Epochs**: 1000
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss function**: MSE

**Input at timestep k** (sequence of 10 past timesteps):
```
Tensor [10, 9] = [
  [roll_gt(k-9), pitch_gt(k-9), yaw_gt(k-9), ω_x(k-9), ω_y(k-9), ω_z(k-9), τ_x(k-9), τ_y(k-9), τ_z(k-9)],
  [roll_gt(k-8), pitch_gt(k-8), yaw_gt(k-8), ω_x(k-8), ω_y(k-8), ω_z(k-8), τ_x(k-8), τ_y(k-8), τ_z(k-8)],
  ...
  [roll_gt(k), pitch_gt(k), yaw_gt(k), ω_x(k), ω_y(k), ω_z(k), τ_x(k), τ_y(k), τ_z(k)]  # Current state
]
```

**Output for timesteps k+1 to k+10** (multi-step ahead):
```
Matrix [10, 3] = [
  [roll_gt(k+1), pitch_gt(k+1), yaw_gt(k+1)]    # 1-step ahead (0.03s)
  [roll_gt(k+2), pitch_gt(k+2), yaw_gt(k+2)]    # 2-steps ahead (0.06s)
  [roll_gt(k+3), pitch_gt(k+3), yaw_gt(k+3)]    # 3-steps ahead (0.09s)
  [roll_gt(k+4), pitch_gt(k+4), yaw_gt(k+4)]    # 4-steps ahead (0.12s)
  [roll_gt(k+5), pitch_gt(k+5), yaw_gt(k+5)]    # 5-steps ahead (0.15s)
  [roll_gt(k+6), pitch_gt(k+6), yaw_gt(k+6)]    # 6-steps ahead (0.18s)
  [roll_gt(k+7), pitch_gt(k+7), yaw_gt(k+7)]    # 7-steps ahead (0.21s)
  [roll_gt(k+8), pitch_gt(k+8), yaw_gt(k+8)]    # 8-steps ahead (0.24s)
  [roll_gt(k+9), pitch_gt(k+9), yaw_gt(k+9)]    # 9-steps ahead (0.27s)
  [roll_gt(k+10), pitch_gt(k+10), yaw_gt(k+10)] # 10-steps ahead (0.30s)
]
```

### Training Data Preparation

**Creating samples from dataset_1.csv:**
```cpp
// For each timestep k (from 0 to 3397-K-N):
X[k] = dataset[k:k+K, :]           // Input: 10 past timesteps, all 9 features
y[k] = dataset[k+K:k+K+N, 0:3]     // Target: next 10 timesteps, angles only (cols 0-2)

// Resulting shapes:
// X: [3378, 10, 9] (3378 samples, 10 timesteps lookback, 9 features)
// y: [3378, 10, 3] (3378 samples, 10 timesteps ahead, 3 angles)
```

**Training Strategy:**
- **No test set** - use all data for training (per professor's instructions)
- **Model checkpoints**: Saved every 100 epochs
- **Training time**: ~645 seconds for 1000 epochs (~10.75 minutes)
- Evaluate on training data to verify correctness

### Evaluation Metrics

**Implemented metrics:**
- **RMSE** (Root Mean Square Error) - primary metric
- **MAE** (Mean Absolute Error) - alternative metric
- **R² score** (coefficient of determination) - implemented but not primary focus

**Metrics computed at multiple levels:**
1. **Overall**: Single number across all samples, steps, and angles
2. **Per angle**: RMSE for each angle (roll, pitch, yaw) across all samples and steps
3. **Per step**: RMSE for each prediction horizon showing accuracy degradation

### Performance Results (1000 epochs, hidden_size=128)

**Overall Performance:**
- **RMSE (all)**: 0.559° (across all predictions, all angles, all 10 steps)
- **MAE (all)**: 0.431°

**RMSE per Angle (all samples, all steps):**
- **Roll**: 0.710° (hardest to predict)
- **Pitch**: 0.463° ✅ **Beats EKF baseline (0.720°)!**
- **Yaw**: 0.468° (slightly easier than pitch)

**RMSE per Prediction Step (shows accuracy degradation over 0.30s horizon):**

| Step | Time Ahead | Roll | Pitch | Yaw | Observation |
|------|------------|------|-------|-----|-------------|
| 1 | 0.03s | 0.672° | 0.460° | 0.363° | **Short-term: Excellent** |
| 2 | 0.06s | 0.634° | 0.457° | 0.376° | Very stable |
| 3 | 0.09s | 0.633° | 0.464° | 0.430° | Still good |
| 4 | 0.12s | 0.649° | 0.463° | 0.502° | Modest degradation |
| 5 | 0.15s | 0.677° | 0.462° | 0.528° | Acceptable |
| 6 | 0.18s | 0.707° | 0.465° | 0.525° | Roll degrading |
| 7 | 0.21s | 0.738° | 0.467° | 0.501° | Roll continues rising |
| 8 | 0.24s | 0.763° | 0.466° | 0.464° | Roll challenging |
| 9 | 0.27s | 0.784° | 0.459° | 0.460° | Pitch remarkably stable |
| 10 | 0.30s | 0.815° | 0.470° | 0.492° | **Long-term: Roll <0.82°** |

**Key Findings:**
- **Pitch stability**: Pitch RMSE stays nearly constant (0.460°-0.470°) across all 10 prediction steps - remarkably stable!
- **Yaw pattern**: Increases then decreases (0.363° → 0.528° → 0.460°), suggesting different dynamics
- **Roll degradation**: Roll error increases steadily from 0.634° (step 2) to 0.815° (step 10) - 28% increase
- **Overall**: Error increase is much more gradual than naive extrapolation would suggest
- **Comparison**: Even at 10-step ahead (0.30s), average error (0.559°) beats the old 5-step model (0.691°)
- Demonstrates that longer training (1000 vs 300 epochs) significantly improves multi-step prediction accuracy

### Implementation Files

**Training Program:** `RNN/lstmMain.cpp`
- Loads `Data/dataset_1.csv`
- Creates LSTM network with lookback window
- Trains for specified epochs with batch processing
- Saves model checkpoints every 100 epochs
- Evaluates on training data with all metrics
- Outputs timing information

**Usage:**
```bash
cd RNN
make lstm
./lstm.out                # Default 300 epochs
./lstm.out -epochs 1000   # Custom number of epochs
./lstm.out --help         # Show usage
```

**Evaluation Program:** `RNN/lstmEval.cpp`
- Loads saved model checkpoint
- Two modes: single sample evaluation OR full dataset prediction export
- **Single sample mode**: Shows input sequence, predictions, and ground truth for debugging
- **Export mode**: Generates CSV with all predictions for visualization

**Usage:**
```bash
make lstmEval
# Single sample evaluation
./lstmEval.out lstm_model_epoch_1000.pt 0     # Test sample 0
./lstmEval.out lstm_model_epoch_1000.pt 1000  # Test sample 1000

# Generate full predictions CSV (for plotting)
./lstmEval.out lstm_model_epoch_1000.pt --save-all
# Creates Results/lstm_predictions.csv with columns:
# timestep, step_ahead, roll_pred, pitch_pred, yaw_pred, roll_gt, pitch_gt, yaw_gt
```

**Model Checkpoints:**
- `lstm_model_epoch_100.pt` - Model after 100 epochs
- `lstm_model_epoch_200.pt` - Model after 200 epochs
- `lstm_model_epoch_300.pt` - Intermediate checkpoint
- `lstm_model_epoch_1000.pt` - Final trained model

### Hyperparameter Optimization Notes

**Epoch experiments (with hidden_size=128, N=10 prediction horizon):**
- `300 epochs`: Overall RMSE = 0.691°, Roll = 0.881°, Pitch = 0.631°
- `1000 epochs`: Overall RMSE = 0.559°, Roll = 0.710°, Pitch = 0.463° ✅ **Better**
- Longer training (1000 epochs) provides **19.1% improvement** in overall RMSE
- Particularly effective for pitch estimation (26.6% improvement over 300 epochs)
- Training loss continues improving on log scale throughout all 1000 epochs

**Hidden size experiments (early experiments with N=5):**
- `hidden_size=64`: Overall RMSE = 0.770°, Roll = 0.990°, Pitch = 0.772°
- `hidden_size=128`: Overall RMSE = 0.691°, Roll = 0.881°, Pitch = 0.631° ✅ **Better**
- Larger hidden size (128) provides 11.6% improvement in overall RMSE

**Training time trade-offs:**
- Lookback window increases training time ~10x per epoch
- 10-timestep lookback: Processes 10x more data per sample than single-timestep
- 1000 epochs with hidden_size=128, N=10: ~645 seconds (~10.75 minutes)
- 300 epochs with hidden_size=128, N=5: ~195 seconds (~3.25 minutes)

**Recommendations:**
- hidden_size=128 is optimal for this dataset size (3378 samples)
- 1000 epochs provides significantly better results than 300 (worth the extra training time)
- Could experiment with 256 hidden size for larger datasets
- Lookback window significantly improves performance (worth the training time)

**Training Convergence Pattern:**
- **Rapid initial convergence**: 96% error reduction in first 50 epochs (0.614° → 0.022°)
- **Plateau phase**: Minimal improvement from epoch 50-1000 (0.022° → 0.003°)
- **Expected behavior**: Network learns main dynamics quickly, then fine-tunes
- **Early stopping viable**: Could stop at epoch 200-300 with minimal performance loss
- **Why fast convergence**: Ground truth angles in input + smooth temporal dynamics + well-sized architecture

### RNN Learning Examples

The `RNN/` directory contains educational examples in both Python and C++:

- **`RNN/Python/hello_rnn.py`**: Basic RNN in PyTorch - learns f(x) = x + 1
- **`RNN/Python/hello_lstm.py`**: Basic LSTM in PyTorch - same task, shows LSTM architecture
- **`RNN/mainExample.cpp`**: C++ LibTorch equivalent - demonstrates nn::Module, forward(), training loop

These examples use sequence input shape `(batch=1, seq_len=5, features=1)` and demonstrate:
- Creating custom nn::Module classes
- Forward pass implementation
- Training loop with MSELoss and Adam optimizer
- Difference between RNN (hidden state only) and LSTM (cell + hidden state + gates)

## Important File Locations

- **Mathematical Documentation**: `MDFiles/` contains detailed derivations and verification reports
- **Research Papers**: `Pappers/` includes foundational literature (Mahony, Kalman, quaternion-based EKF)
- **Parameter Tuning**: See filter main files for optimized constants (e.g., Mahony kp=11, Complementary alpha=0.9)
- **RNN Architecture Plan**: `MDFiles/RNN_Hybrid_Architecture.md` describes upcoming deep learning integration
- **Build Configuration**: Makefile uses `-std=c++14 -g -Iinclude` flags, outputs to `build/` and `bin/`
