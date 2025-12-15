# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++14 diploma thesis project implementing and comparing attitude estimation algorithms for UAV navigation using IMU sensor fusion. The project estimates roll and pitch angles from 6-DOF IMU data (gyroscope + accelerometer) and evaluates performance against ground truth measurements.

**Implemented Algorithms:**
- Complementary Filter: Basic weighted sensor fusion baseline
- Mahony Filter: Nonlinear complementary filter on SO(3) (current best performer)
- Explicit Complementary Filter: Vector-based filter with bias estimation
- Extended Kalman Filter (EKF): Quaternion-based probabilistic state estimation

**Planned Work:**
- RNN Hybrid Architecture: Integrating LSTM/GRU with classical filters using LibTorch (see `MDFiles/RNN_Hybrid_Architecture.md`)

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
python plotDiagrams.py           # General plotting
python plotDiagrams_zoomed.py    # Zoomed-in analysis
python analyze_dynamics.py       # Gyro/Accel magnitude vs Error
```

**RNN Examples and Development:**
```bash
cd RNN

# Python learning examples (for understanding LSTM/RNN concepts)
cd Python
python hello_rnn.py      # Basic RNN: learns to add 1 to sequence
python hello_lstm.py     # Basic LSTM: same task, demonstrates architecture differences

# C++ LibTorch examples and production code
cd ..
# Edit Makefile line 7-8 to set LIBTORCH_PATH to your libtorch installation
make
./example.out            # Runs C++ RNN learning example

# Production RNN hybrid will be implemented in C++ using LibTorch
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
  - **Columns 3-5**: Gyroscope measurements (roll, pitch, yaw) - **rad/s** (confirmed by professor)
  - **Columns 6-8**: Control torques (roll, pitch, yaw) - from PID controller, NOT accelerometer

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

## RNN Hybrid Architecture (In Development)

**Status:** Implementation phase - specifications finalized December 2025

**Approach:** Multi-step ahead prediction using LSTM with full state information

**Implementation Language:** C++ using LibTorch (PyTorch C++ API)

### LSTM Architecture Specifications (Final)

**Input at timestep k** (single timestep, 9 features):
```
Vector [9] = [roll_gt(k), pitch_gt(k), yaw_gt(k),          # Ground truth angles
              gyro_roll(k), gyro_pitch(k), gyro_yaw(k),     # Gyro measurements
              torque_roll(k), torque_pitch(k), torque_yaw(k)] # Control torques
```

**Output for timesteps k+1 to k+N** (multi-step ahead):
```
Matrix [N, 3] where N=10 (configurable)
Predict ground truth angles for next 10 timesteps:
  [roll_gt(k+1), pitch_gt(k+1), yaw_gt(k+1)]
  [roll_gt(k+2), pitch_gt(k+2), yaw_gt(k+2)]
  ...
  [roll_gt(k+10), pitch_gt(k+10), yaw_gt(k+10)]
```

### Training Data Preparation

**Creating samples from dataset_1.csv:**
```cpp
// For each timestep k (from 0 to 3397-N-1):
X[k] = dataset[k, :]              // Input: all 9 columns at timestep k
y[k] = dataset[k+1:k+N+1, 0:3]    // Target: next N rows, angles only (cols 0-2)

// Resulting shapes:
// X: [num_samples, 9] where num_samples = 3397 - N
// y: [num_samples, N, 3]
```

**Training Strategy (per professor's instructions):**
- **No test set** - use all data for training (dataset size is small for deep learning)
- Train on all 3397 samples
- After training, evaluate on training data to verify correctness
- Should achieve good metrics on training data if implementation is correct

### Evaluation Metrics

**Compute separately for each prediction horizon:**
- **R² score** (coefficient of determination)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)

**For N=10, report 10 sets of metrics:**
```
Step 1 (k+1):  R²=?, MAE=?, RMSE=?  (1-step ahead)
Step 2 (k+2):  R²=?, MAE=?, RMSE=?  (2-steps ahead)
...
Step 10 (k+10): R²=?, MAE=?, RMSE=? (10-steps ahead)
```

**Expected behavior:** Metrics typically degrade with prediction horizon (short-term predictions more accurate than long-term)

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

## IDE Setup for clangd

The project uses `compile_commands.json` for LSP-based tooling. Generate it with:

```bash
sudo apt install bear  # if not installed
make clean
bear -- make all
```

VSCode settings recommended in README.md disable Microsoft C++ IntelliSense in favor of clangd for better performance.

## Important File Locations

- **Mathematical Documentation**: `MDFiles/` contains detailed derivations and verification reports
- **Research Papers**: `Pappers/` includes foundational literature (Mahony, Kalman, quaternion-based EKF)
- **Parameter Tuning**: See filter main files for optimized constants (e.g., Mahony kp=11, Complementary alpha=0.9)
- **RNN Architecture Plan**: `MDFiles/RNN_Hybrid_Architecture.md` describes upcoming deep learning integration
- **Build Configuration**: Makefile uses `-std=c++14 -g -Iinclude` flags, outputs to `build/` and `bin/`

## Common Pitfalls

- **Eigen version mismatch**: Code only compiles with Eigen 3.4.1. Check with `cat /usr/include/eigen3/Eigen/src/Core/util/Macros.h | grep EIGEN_WORLD_VERSION`
- **Misspellings in targets**: Note `complemntaryFilter` target and `complmentary.out` binary (missing 'e')
- **Data file paths**: All filters expect `Data/` directory in repository root
- **Units**: Gyro in rad/s, accel in m/s�, ground truth in radians (converted to degrees for display)
- **Result file overwrites**: Running filters overwrites previous results in `Results/Results/` directory
