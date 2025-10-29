# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ diploma thesis project for attitude estimation using IMU data. Implements and compares three sensor fusion algorithms for roll and pitch estimation from gyroscope and accelerometer measurements.

**Current Status**: ✅ **IMPLEMENTATION COMPLETE** - All three filters fully implemented, tested, and producing results.

### Implemented Filters
1. **Complementary Filter** (α = 0.79) - RMSE: ~0.6° - Simple weighted fusion of gyro integration and accel measurements
2. **Mahony Filter** (kp = 9) - Roll RMSE: ~0.588° / Pitch RMSE: ~1.43° - Passive complementary filter with rotation matrix representation (⚠️ pitch needs re-tuning)
3. **Extended Kalman Filter (EKF)** - RMSE: ~0.298° - Quaternion-based optimal estimator with gyro bias correction

**Project Status**: Core algorithms complete. Investigating RNN/deep learning approaches and optimizing Mahony filter performance.

## Data
Data/gyro.csv contains gyroscope measurements from the IMU. 3-axes 1 column per axe (x,y,z). Values are rad/sec
Data/accel.csv contains acceleration measuremnets from the IMU. 3-axes 1 column per axe (x,y,z). Values are m/sec^/2

Angles.csv contains the actual roll and pitch values during the time of the test, these are the truth data. First column is roll (phi) and
second column is pitch (theta). Values are rad/sec

## Build System

The project supports both Make and CMake build systems:

### Using Make (Primary)
```bash
make run          # Compile and run the application
make main         # Compile only
make clean        # Remove built files
```

### Using CMake (Alternative)
```bash
cmake -B build

# Build individual filters
cmake --build build --target complemntaryFilter
cmake --build build --target mahonyFilter
cmake --build build --target ekfFilter

# Build all
cmake --build build --target all_filters

# Run filters
cmake --build build --target run_complementary
cmake --build build --target run_mahony
cmake --build build --target run_ekf
```

Both build systems (Make and CMake) are fully supported and produce identical binaries in `bin/` directory.

## Architecture

### Core Components

#### Sensor Fusion Algorithms (All Complete ✅)

- **ComplementaryFilter**: Simple weighted fusion algorithm
  - `ComplementaryFilter.hpp/cpp`: Filter class with configurable alpha coefficient (0.79) and sampling rate (0.02s)
  - Implements: `setIMUData()`, `calculateRoll()`, `calculatePitch()`
  - Output: Roll/pitch estimations stored as Eigen vectors
  - Results: `Results/Results/ComplementaryRoll_a_0_79.txt`, `ComplementaryPitch_a_0_79.txt`

- **MahonyFilter**: Passive complementary filter with SO(3) representation
  - `MahonyFilter.hpp/cpp`: Rotation matrix-based filter with proportional gain (kp = 9)
  - Implements: `setIMUData()`, `predictForAllData()` with automatic orthonormalization
  - Features: Error correction in rotation space using skew-symmetric matrices
  - Results: `Results/Results/MahonyRoll_kp_9.txt`, `MahonyPitch_kp_9.txt`

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
   - Mahony: kp gain (9), dt (0.02s)
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
make complemntaryFilter   # Build complementary filter
make mahonyFilter          # Build Mahony filter
make ekfFilter             # Build EKF
make all                   # Build all three filters
```

### Run Filters
```bash
./bin/complmentary.out     # Run complementary filter
./bin/mahony.out           # Run Mahony filter
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
- `mahonyFilterMain.cpp` - Mahony filter entry point
- `ekfFilterMain.cpp` - EKF entry point

All three follow the same pattern:
1. Load CSV data (gyro, accel, ground truth angles)
2. Initialize filter with parameters
3. Call `setIMUData()` and `predictForAllData()` (or equivalent)
4. Convert results to degrees and calculate RMSE/MEA
5. Save results to `Results/Results/`

### Implementation Files
- `src/ComplementaryFilter.cpp` & `include/ComplementaryFilter.hpp`
- `src/MahonyFilter.cpp` & `include/MahonyFilter.hpp`
- `src/ExtendedKalmanFilter.cpp` & `include/ExtendedKalmanFilter.hpp`
- `include/Utils.hpp` - Shared utilities (RMSE, angle conversion, file I/O)
- `include/csvreader.hpp` - CSV data loading

### Visualization Scripts (Python)
- `Results/plotDiagrams.py` - Individual filter plots (configurable)
- `Results/plotDiagrams_zoomed.py` - Zoomed view for detailed analysis
- `Results/analyze_dynamics.py` - **Comprehensive 3-filter comparison with dynamics analysis**

### Documentation
- **`MDFiles/EKF_Complete_Mathematical_Reference.md`** - **PRIMARY EKF REFERENCE**
  - Complete mathematical documentation with all EKF equations
  - Quaternion mathematics, kinematics, and rotation representations
  - Detailed Jacobian derivations (F: 7×7 state transition, H: 3×7 measurement)
  - Tuning guide for process/measurement noise covariance matrices
  - Troubleshooting common EKF issues
- `MDFiles/MahonyFilter_Mathematical_Documentation.md` - Mahony filter theory
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

| Filter | Roll RMSE | Pitch RMSE | Parameters | Strengths | Limitations |
|--------|-----------|------------|------------|-----------|-------------|
| **Complementary** | 0.820° | 0.771° | α = 0.79, dt = 0.02s | Simple, computationally efficient, intuitive tuning | No bias estimation, fixed weighting |
| **Mahony** | 0.589° | 1.430° ⚠️ | kp = 9, dt = 0.02s | Better dynamics handling, rotation space correction | kp tuned for roll only, pitch suffers |
| **EKF** | **0.298°** | **0.720°** | Q, R covariance matrices | **Best accuracy**, automatic bias estimation, optimal fusion | Most complex, requires tuning multiple parameters |

**Winner: EKF** achieves best performance on both axes through optimal state estimation and gyroscope bias correction.

**Note:** Mahony filter kp was optimized for roll only (see mahonyFilterMain.cpp:55). Re-tuning for combined roll+pitch metric should improve pitch RMSE from 1.43° to ~0.9°.

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

✅ **Completed:**
- All three sensor fusion algorithms implemented and tested
- Consistent API across all filters (`setIMUData()`, `predictForAllData()` pattern)
- Comprehensive visualization and analysis tools
- Full mathematical documentation for all algorithms
- Performance benchmarking complete

🎯 **Current Investigations:**
- **Data availability check:** Investigating if more IMU data exists (potentially 1+ hour vs current 28 seconds)
- **Mahony optimization:** Re-tune kp for combined roll+pitch performance (currently optimized for roll only)
- **RNN/ML exploration:** Evaluating feasibility of deep learning approaches
  - See `MDFiles/RNN_Feasibility_Analysis.md` for comprehensive analysis
  - Options: Simple LSTM, Attention-LSTM, Hybrid EKF+LSTM
  - Viability depends on data availability and professor approval

🔧 **Pending Tasks:**
- Fix Mahony pitch tuning (1 day, guaranteed improvement)
- Verify total available IMU dataset size
- Consult professor on ML approach for thesis
- Decide: Classical optimization vs Deep learning exploration