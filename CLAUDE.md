# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ diploma thesis project for attitude estimation using IMU data. Implements and compares three sensor fusion algorithms for roll and pitch estimation from gyroscope and accelerometer measurements.

**Current Status**: ✅ **ALL FILTERS VERIFIED AND VALIDATED** - Four sensor fusion algorithms fully implemented, optimized, tested, and verified.

### Implemented Filters
1. **Complementary Filter** (α = 0.79) - Roll: 0.820° / Pitch: 0.771° - Simple weighted fusion baseline
2. **Mahony Passive Filter** (kp = 11) - Roll: 0.614° / Pitch: 0.756° - Passive CF with SO(3) representation
3. **Explicit Complementary Filter** (kp = 11, ki = 0.05) - Roll: 0.554° / Pitch: 0.752° - Direct vectorial CF with bias estimation (BEST complementary filter)
4. **Extended Kalman Filter (EKF)** - Roll: 0.298° / Pitch: 0.720° - Quaternion-based optimal estimator (BEST OVERALL)

**Project Status**: ✅ All algorithms complete, optimized, and verified. Comprehensive verification report available at `MDFiles/Filter_Verification_Report.md`. Ready for thesis writing.

## Data
Data/gyro.csv contains gyroscope measurements from the IMU. 3-axes 1 column per axe (x,y,z). Values are rad/sec
Data/accel.csv contains acceleration measuremnets from the IMU. 3-axes 1 column per axe (x,y,z). Values are m/sec^/2

Angles.csv contains the actual roll and pitch values during the time of the test, these are the truth data. First column is roll (phi) and
second column is pitch (theta). Values are rad/sec

## Build System

The project uses Makefile for building all filters:

```bash
make all           # Build all four filters
make clean         # Remove built files
```

## Architecture

### Core Components

#### Sensor Fusion Algorithms (All Complete ✅)

- **ComplementaryFilter**: Simple weighted fusion algorithm
  - `ComplementaryFilter.hpp/cpp`: Filter class with configurable alpha coefficient (0.79) and sampling rate (0.02s)
  - Implements: `setIMUData()`, `calculateRoll()`, `calculatePitch()`
  - Output: Roll/pitch estimations stored as Eigen vectors
  - Results: `Results/Results/ComplementaryRoll_a_0_79.txt`, `ComplementaryPitch_a_0_79.txt`

- **MahonyFilter**: Passive complementary filter with SO(3) representation
  - `MahonyFilter.hpp/cpp`: Rotation matrix-based filter with proportional gain (kp = 11)
  - Implements: `setIMUData()`, `predictForAllData()` with automatic orthonormalization
  - Features: Error correction in rotation space using skew-symmetric matrices
  - Results: `Results/Results/MahonyRoll_kp_11.txt`, `MahonyPitch_kp_11.txt`

- **ExplicitComplementaryFilter**: Direct vectorial complementary filter with bias estimation (✅ COMPLETE)
  - `ExplicitComplementaryFilter.hpp/cpp`: Works directly with vector measurements (no attitude reconstruction)
  - Implements: `setIMUData()`, `predictForAllData()`, `getBiasEstimation()` with automatic orthonormalization
  - Parameters: kp = 11.0 (proportional), ki = 0.05 (integral for bias)
  - Features: 3-axis gyro bias estimation, simple cross product correction (v × v̂), no trig functions
  - Results: `Results/Results/ExplicitComplementaryRoll.txt`, `ExplicitComplementaryPitch.txt`

- **ExtendedKalmanFilter**: Quaternion-based optimal estimator (✅ COMPLETE)
  - `ExtendedKalmanFilter.hpp/cpp`: 7D state vector (quaternion + 3D gyro bias)
  - Implements: Full predict-update cycle with Jacobian computation
    - `predict()`: Quaternion kinematics with bias-corrected gyro
    - `update()`: Accelerometer measurement model with Kalman gain
    - `computeF()`: State transition Jacobian (7×7)
    - `computeH()`: Measurement Jacobian (3×7)
  - Features: Automatic bias estimation, quaternion normalization, covariance tracking
  - Results: `Results/Results/EkfRoll.txt`, `EkfPitch.txt`

#### Support Components

- **CsvReader**: Data input handler for CSV files
  - `csvreader.hpp`: Template-based CSV parser that converts data to Eigen matrices
  - Handles gyro, accelerometer, and ground truth angle data from Data/ directory

- **Utils**: Mathematical utilities and file I/O operations
  - `Utils.hpp`: Static utility functions for angle conversion, RMSE/MEA calculations, vector operations
  - File output functions for results storage in Results/ directory

### Common Data Flow (All Filters)

1. **Load Data**: Read CSV files from `Data/` directory
   - `gyro.csv` - 3-axis gyroscope (rad/s)
   - `accel.csv` - 3-axis accelerometer (m/s²)
   - `angles.csv` - Ground truth roll/pitch (rad)

2. **Initialize Filter**: Create filter object with tuned parameters
   - Complementary: alpha coefficient (0.79), dt (0.02s)
   - Mahony Passive: kp gain (11), dt (0.02s)
   - Explicit CF: kp gain (11), ki gain (0.05), dt (0.02s)
   - EKF: dt (0.02s), initial quaternion from first accel sample

3. **Process Data**:
   - Call `setIMUData(gyro, accel)` to load sensor measurements
   - Call `predictForAllData()` to run filter on all samples
   - Filter handles predict-update cycle internally

4. **Extract Results**:
   - Get roll/pitch estimates: `getRollEstimation()`, `getPitchEstimation()`
   - Convert from radians to degrees
   - Calculate RMSE and MEA vs ground truth

5. **Save and Visualize**:
   - Save estimates to `Results/Results/*.txt`
   - Generate plots using Python scripts
   - Analyze performance metrics

## Dependencies

- **Eigen3**: Matrix operations library (must be installed in system paths)
- **Standard C++14**: Required compiler standard
- **Python with matplotlib**: For visualization (plotDiagrams.py)

## Build and Run

### Build Individual Filters
```bash
make complemntaryFilter          # Build complementary filter
make mahonyFilter                # Build Mahony passive filter
make explicitComplementaryFilter # Build explicit complementary filter
make ekfFilter                   # Build EKF
make all                         # Build all four filters
```

### Run Filters
```bash
./bin/complmentary.out     # Run complementary filter
./bin/mahony.out           # Run Mahony passive filter
./bin/explicitCF.out       # Run explicit complementary filter
./bin/ekf.out              # Run EKF
```

Each filter outputs:
- Console: RMSE and MEA error metrics
- Files: Roll/pitch estimates saved to `Results/Results/`

## Visualization and Analysis

### Individual Filter Plots
```bash
python Results/plotDiagrams.py         # Plot EKF results (currently configured)
python Results/plotDiagrams_zoomed.py  # Zoomed view of EKF (samples 300-700)
```
- Generates comparison plots: Estimated vs Ground Truth
- Displays RMSE on plot
- Saves to `Results/Figures/`

### Comprehensive Analysis
```bash
python Results/analyze_dynamics.py     # Full 3-filter comparison with sensor dynamics
```
Generates 4-panel analysis plot showing:
1. Roll angle comparison (all 3 filters vs ground truth)
2. Absolute errors with RMSE in legend
3. Gyro magnitude (angular rate dynamics)
4. Accelerometer magnitude (external acceleration detection)

**Key Features:**
- Automatic RMSE calculation and display
- High/low dynamics performance breakdown (75th percentile threshold)
- Best performer identification for different motion regimes
- Output: `Results/Figures/Roll_Dynamics_Analysis.png`

## Key Files

### Main Executables
- `complentaryFilterMain.cpp` - Complementary filter entry point
- `mahonyFilterMain.cpp` - Mahony passive filter entry point
- `explicitComplementaryFilterMain.cpp` - Explicit complementary filter entry point
- `ekfFilterMain.cpp` - EKF entry point

All four follow the same pattern:
1. Load CSV data (gyro, accel, ground truth angles)
2. Initialize filter with parameters
3. Call `setIMUData()` and `predictForAllData()` (or equivalent)
4. Convert results to degrees and calculate RMSE/MEA
5. Save results to `Results/Results/`

### Implementation Files
- `src/ComplementaryFilter.cpp` & `include/ComplementaryFilter.hpp`
- `src/MahonyFilter.cpp` & `include/MahonyFilter.hpp`
- `src/ExplicitComplementaryFilter.cpp` & `include/ExplicitComplementaryFilter.hpp`
- `src/ExtendedKalmanFilter.cpp` & `include/ExtendedKalmanFilter.hpp`
- `include/Utils.hpp` - Shared utilities (RMSE, angle conversion, file I/O)
- `include/csvreader.hpp` - CSV data loading

### Visualization Scripts (Python)
- `Results/plotDiagrams.py` - Individual filter plots (configurable)
- `Results/plotDiagrams_zoomed.py` - Zoomed view for detailed analysis
- `Results/analyze_dynamics.py` - **Comprehensive 3-filter comparison with dynamics analysis**

### Documentation
- **`MDFiles/Filter_Verification_Report.md`** - **COMPREHENSIVE VERIFICATION REPORT**
  - Complete verification of all four filters
  - Mathematical correctness validation
  - Hyperparameter optimization summaries
  - Performance comparison analysis
  - Bug fixes and resolutions documented
  - Thesis-ready checklist
- **`MDFiles/EKF_Complete_Mathematical_Reference.md`** - EKF mathematical reference
  - Complete mathematical documentation with all EKF equations
  - Quaternion mathematics, kinematics, and rotation representations
  - Detailed Jacobian derivations (F: 7×7 state transition, H: 3×7 measurement)
  - Tuning guide for process/measurement noise covariance matrices
- `MDFiles/ExplicitComplementaryFilter_Mathematical_Documentation.md` - Explicit CF theory and implementation
- `MDFiles/Explicit_vs_Passive_Mahony_Comparison.md` - Side-by-side comparison of Mahony filters
- `MDFiles/MahonyFilter_Mathematical_Documentation.md` - Mahony passive filter theory
- `pch.h` - Precompiled header with common includes (Eigen, iostream, etc.)
- `compile_commands.json` - Generated compilation database for IDE support

### Project Structure
```
├── Data/                    # Input sensor data
│   ├── gyro.csv            # Gyroscope measurements (rad/s)
│   ├── accel.csv           # Accelerometer measurements (m/s²)
│   └── angles.csv          # Ground truth angles (rad)
├── Results/
│   ├── Results/            # Filter output files (.txt)
│   ├── Figures/            # Generated plots (.png)
│   └── *.py                # Analysis scripts
├── src/                    # Implementation files (.cpp)
├── include/                # Header files (.hpp)
└── MDFiles/                # Documentation
```
## Performance Summary

| Filter | Roll RMSE | Pitch RMSE | Combined | Bias Estimation | Parameters |
|--------|-----------|------------|----------|-----------------|------------|
| **Complementary** | 0.820° | 0.771° | 0.795° | ❌ No | α = 0.79 |
| **Mahony Passive** | 0.614° | 0.756° | 0.685° | ❌ No | kp = 11 |
| **Explicit CF** | **0.554°** | 0.752° | 0.653° | ✅ Yes (3-axis) | kp = 11, ki = 0.05 |
| **EKF** | **0.298°** | **0.720°** | **0.509°** | ✅ Yes (3-axis) | Q, R matrices |

**Rankings:**
1. **🥇 EKF** - Best overall performance (0.298° roll, 0.720° pitch)
2. **🥈 Explicit CF** - Best complementary filter (0.554° roll) + bias estimation
3. **🥉 Mahony Passive** - Good performance (0.614° roll), simpler than Explicit CF
4. **Complementary** - Simple baseline (0.820° roll), single parameter

### High vs Low Dynamics Performance
The `analyze_dynamics.py` script automatically identifies high-dynamics periods (top 25% gyro magnitude) and calculates separate RMSE for each filter, revealing performance under different motion conditions.

## Research Materials

- **Papers Read**: All EKF research materials studied in order:
  1. `kalman_intro.pdf` - Basic Kalman filter theory and fundamentals
  2. `kalaman_intermediate.pdf` - Advanced Kalman filter concepts
  3. `Quaternion-based_extended_Kalman_filter_for_determining_orientation_by_inertial_and_magnetic_sensing.pdf` - Primary reference for quaternion-based EKF implementation
- **Mathematical References**: Comprehensive documentation in `MDFiles/EKF_Complete_Mathematical_Reference.md`
- All relevant papers located in `Pappers/` folder
- **Archive Note**: `ArchiveDoNotTouch/` contains old files - do not use

## Project Status Notes

✅ **FULLY COMPLETED AND VERIFIED (November 9, 2025):**
- **All four sensor fusion algorithms** implemented, optimized, and tested
  - Complementary Filter (α = 0.79)
  - Mahony Passive Filter (kp = 11)
  - Explicit Complementary Filter (kp = 11, ki = 0.05)
  - Extended Kalman Filter
- **Hyperparameter optimization** completed via grid search for all filters
  - Complementary: 99 tests for alpha
  - Mahony Passive: 100 tests for kp
  - Explicit CF: 580 tests for (kp, ki) combinations
  - EKF: Manual tuning of Q and R matrices
- **Complete verification** against ground truth (100% RMSE match with expected values)
- **Consistent API** across all filters (`setIMUData()`, `predictForAllData()` pattern)
- **Comprehensive documentation** (mathematical references, implementation guides, comparison documents)
- **All output files** generated and verified (8 files, 1409 samples each)
- **Visualization and analysis tools** complete
- **Build system** working perfectly (Makefile with 4 targets)

📊 **Verified Results:**
- All RMSE values match expected results from `notes.txt` (100% accuracy)
- Best overall: EKF (0.298° roll, 0.720° pitch)
- Best complementary filter: Explicit CF (0.554° roll, 0.752° pitch)
- See `MDFiles/Filter_Verification_Report.md` for complete verification details

✅ **Ready for thesis writing** - All classical filters complete

🚀 **NEW PHASE: RNN Hybrid Implementation (Starting)**
- **Approach:** Hybrid RNN combining filter output + IMU data
- **Technology:** C++ with libtorch (PyTorch C++ API)
- **Architecture:** Filter output (angles) + IMU → LSTM → Predicted state (k+1)
- **Input:** 8D vector [roll, pitch, ax, ay, az, ωx, ωy, ωz] at time k
- **Output:** 5D vector [roll, pitch, ωx, ωy, ωz] at time k+1
- **Goal:** Improve upon best classical filter (EKF: 0.298° roll, 0.720° pitch)
- **Documentation:** See `MDFiles/RNN_Hybrid_Architecture.md` for complete specification
- **Status:** Planning complete, ready to implement