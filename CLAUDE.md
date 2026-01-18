# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++14/17 diploma thesis project implementing and comparing attitude estimation algorithms for UAV navigation using IMU sensor fusion. The project estimates roll and pitch angles from 6-DOF IMU data (gyroscope + accelerometer) and evaluates performance against ground truth measurements. Also an LSTM model is later developed for altitude estimation

**Implemented Algorithms:**
- Complementary Filter: Basic weighted sensor fusion baseline
- Mahony Filter: Nonlinear complementary filter on SO(3) - best classical filter
- Explicit Complementary Filter: Vector-based filter with bias estimation
- Extended Kalman Filter (EKF): Quaternion-based probabilistic state estimation
- LSTM Multi-Step Ahead Predictor: Deep learning approach using LibTorch (C++)

## Build Commands

**Dependencies:**
- g++ with C++14 support
- Eigen3 version 3.4.1 (must be in system include paths)
- Python 3 with matplotlib (for visualization)
- LibTorch C++ denendencies

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

**LSTM Multi-Step Ahead Prediction:**

The LSTM model predicts future UAV attitude angles (roll, pitch, yaw) by learning temporal patterns from sensor data and control inputs.

```bash
cd RNN

# Build LSTM training and evaluation binaries (requires LibTorch C++)
make lstm          # Compiles lstmMain.cpp -> lstm.out
make lstmEval      # Compiles lstmEval.cpp -> lstmEval.out

# Train LSTM model
./lstm.out                      # Default: 300 epochs, random seed 42
./lstm.out -epochs 1000         # Train for 1000 epochs
./lstm.out -epochs 1000 -seed 123  # Custom random seed for reproducibility

# Evaluate trained model and export predictions
./lstmEval.out lstm_model_epoch_1000.pt --save-all   # Generates RNN/lstm_predictions.csv

# Visualize predictions (requires lstm_predictions.csv from lstmEval)
python3 plot_multistep_predictions.py 0 50    # Compare predictions at different horizons
python3 plot_predictions.py 0 100             # Single-step vs ground truth
python3 plot_single_pred.py 50                # Detailed analysis of one sample
python3 plot_rmse_errors.py                   # Training loss curve over epochs

# Learning examples (for understanding LSTM/RNN concepts)
cd Python
python3 hello_rnn.py       # Basic RNN implementation in PyTorch
python3 hello_lstm.py      # Basic LSTM implementation in PyTorch
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

#### LSTM Multi-Step Ahead Prediction

**Purpose:** Predict future UAV attitude angles (roll, pitch, yaw) over a prediction horizon by learning temporal patterns from sensor and control data.

**Model Architecture (lstmMain.cpp):**
- **Type**: Sequence-to-sequence LSTM
- **Layers**: Input(9) → LSTM(128 hidden units) → FC(N×3) → Reshape([batch, N, 3])
- **Configuration**:
  - Lookback window (K): 10 timesteps
  - Prediction horizon (N): 10 timesteps (configurable at line 119)
  - Input features: 9 (all dataset columns)
  - Output features: 3 (roll, pitch, yaw angles)

**Dataset:** `Data/dataset_1.csv`
- **Size**: 3397 samples at 33.3 Hz (Ts = 0.03 sec)
- **Training samples**: 3378 (after removing lookback and prediction windows)

**Input Features (9 columns):**
1. **Columns 0-2**: Ground truth angles (roll, pitch, yaw) in radians
   - Used as input to learn angle evolution patterns
2. **Columns 3-5**: Gyroscope angular velocities (ω_roll, ω_pitch, ω_yaw) in rad/s
   - Rotation rates: ω = d(angle)/dt
3. **Columns 6-8**: Control torques (τ_roll, τ_pitch, τ_yaw) from PID controller
   - Commanded control inputs to the UAV

**Output Targets (3 columns):**
- Next N timesteps of (roll, pitch, yaw) angles in radians
- Prediction horizon shifted forward from input window

**Training Configuration:**
- **Batch size**: 32
- **Learning rate**: 0.001 (Adam optimizer)
- **Loss function**: Mean Squared Error (MSE)
- **Data preprocessing**: Batch shuffling per epoch with reproducible random seed
- **Checkpoints**: Models saved every 100 epochs

**Data Flow:**
1. Load dataset as Eigen matrix, convert to LibTorch tensors
2. Create sliding windows: X[i] = rows[i:i+K], y[i] = angles[i+K:i+K+N]
3. Shuffle indices at start of each epoch
4. Train in mini-batches of 32 samples
5. Evaluate on training data (no train/test split currently)

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

**LSTM Multi-Step Predictor:**
- Uses LibTorch C++ API (PyTorch backend required)
- Prediction horizon N configurable at `lstmMain.cpp:119` (default: 10 timesteps)
- Random seed default: 42 (use `--seed` flag for different initializations)
- Model checkpoints saved every 100 epochs as `lstm_model_epoch_N.pt`
- Forward pass: LSTM processes K=10 timesteps, takes last hidden state, projects to N×3 outputs
- Important: Currently evaluates on training data only (no train/test split)
- To change prediction horizon: edit `windowSize` constant and recompile with `make lstm`

## Important File Locations

- **Thesis Document**: `reportMsc.pdf` - Main thesis document being written
- **Mathematical Documentation**: `MDFiles/` contains detailed derivations and verification reports
- **Research Papers**: `Pappers/` includes foundational literature (Mahony, Kalman, quaternion-based EKF)
- **Parameter Tuning**: See filter main files for optimized constants (e.g., Mahony kp=11, Complementary alpha=0.9)
- **RNN Architecture Plan**: `MDFiles/RNN_Hybrid_Architecture.md` describes deep learning integration design
- **LSTM Implementation**: `RNN/lstmMain.cpp` (training), `RNN/lstmEval.cpp` (evaluation), `RNN/Makefile` (LibTorch build)
- **LSTM Visualization**: `RNN/plot_*.py` scripts for multi-step predictions, single-step analysis, and training curves
- **Training Results**: `RNN/Results.txt` contains training logs and performance metrics from experiments
- **Build Configuration**: Makefile uses `-std=c++14 -g -Iinclude` flags, outputs to `build/` and `bin/`; RNN uses `-std=c++17` for LibTorch
