# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ongoing C++ diploma thesis project for attitude estimation using IMU data. Currently implements complementary filter algorithms, with plans to extend to Kalman filtering approaches.

**Current Status**: Complementary filter implementation complete. Research phase completed - all relevant EKF papers have been read. **EKF implementation in progress** - foundation complete with 7D state vector design.
**Next Phase**: Complete quaternion-based Extended Kalman Filter (EKF) implementation. Foundation established: constructor, quaternion utilities, and mathematical framework ready. Next steps: implement predict() function with quaternion kinematics and measurement update.

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
cmake --build build
cmake --build build --target run
```

## Architecture

### Core Components

- **ComplementaryFilter**: Main algorithm implementation that fuses gyroscope and accelerometer data
  - `ComplementaryFilter.hpp/cpp`: Filter class with configurable alpha coefficient and sampling rate
  - Calculates roll and pitch using complementary filtering technique

- **ExtendedKalmanFilter**: Quaternion-based EKF implementation (in progress)
  - `ExtendedKalmanFilter.hpp/cpp`: 7D state vector EKF (quaternion + gyro bias estimation)
  - Foundation complete: constructor, quaternion utilities, mathematical framework
  - Pending: predict() function, measurement update, Jacobian computations

- **CsvReader**: Data input handler for CSV files containing sensor data
  - `csvreader.hpp`: Template-based CSV parser that converts data to Eigen matrices
  - Handles gyro, accelerometer, and ground truth angle data from Data/ directory

- **Utils**: Mathematical utilities and file I/O operations  
  - `Utils.hpp`: Static utility functions for angle conversion, RMSE/MEA calculations, vector operations
  - File output functions for results storage in Results/ directory

### Data Flow

1. Read sensor data from CSV files (gyro.csv, accel.csv, angles.csv in Data/ directory)
2. Initialize ComplementaryFilter with alpha coefficient (typically 0.8) and time step (0.02s)
3. Process data through complementary filter to estimate roll/pitch
4. Compare estimates with ground truth and calculate error metrics
5. Output results to Results/ directory as text files
6. Visualize results using Python plotting script

## Dependencies

- **Eigen3**: Matrix operations library (must be installed in system paths)
- **Standard C++14**: Required compiler standard
- **Python with matplotlib**: For visualization (plotDiagrams.py)

## Development Workflow

1. Modify algorithm parameters in main.cpp (alpha, dt)
2. Build and run: `make run`
3. View numerical results in console output
4. Generate plots: `python plotDiagrams.py`
5. Check Results/ directory for output files

## Key Files

- `main.cpp`: Entry point with data loading and filter execution
- `pch.h`: Precompiled header with common includes (Eigen, iostream, etc.)
- `plotDiagrams.py`: Visualization script for comparing predicted vs expected results
- `compile_commands.json`: Generated compilation database for IDE support

### EKF Implementation Files
- `ExtendedKalmanFilter.hpp/cpp`: Core EKF implementation (7D state: quaternion + bias)
- `jacobian.cpp`: Standalone example demonstrating Jacobian computation with Eigen
- `MDFiles/KalmanEquations.md`: Standard linear Kalman filter equations reference
- `MDFiles/EKFDescriptionEquation.md`: Extended Kalman filter equations with mathematical derivations

The project expects Data/ directory with sensor CSV files and creates Results/ directory for output files.
## Research Materials

- **Papers Read**: All EKF research materials have been studied in the following order:
  1. `kalman_intro.pdf` - Basic Kalman filter theory and fundamentals
  2. `kalaman_intermediate.pdf` - Advanced Kalman filter concepts
  3. `Quaternion-based_extended_Kalman_filter_for_determining_orientation_by_inertial_and_magnetic_sensing.pdf` - Main reference for quaternion-based EKF implementation
- **Reference Document**: `EKF_Equations_Reference.md` - Detailed mathematical reference with all EKF equations and symbol definitions
- All relevant papers for this thesis project are located in the Pappers/ folder