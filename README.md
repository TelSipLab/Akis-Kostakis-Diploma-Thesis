# Akis-Kostakis-Diploma-Thesis

**Diploma Thesis: Attitude Estimation Using IMU Data**
*Author: Akis Kostakis*

## Overview

This project implements and compares multiple attitude estimation algorithms for Inertial Measurement Unit (IMU) data:
- **Complementary Filter**: Basic sensor fusion of gyroscope and accelerometer data
- **Mahony Filter**: Passive complementary filter with proportional feedback correction
- **Extended Kalman Filter (EKF)**: Quaternion-based probabilistic state estimation (in progress)

The algorithms estimate roll and pitch angles from 6-DOF IMU measurements and are evaluated against ground truth data.

## Prerequisites

- **C++ Compiler**: g++ with C++14 support
- **Eigen3**: Linear algebra library (must be in system include paths)
- **Make**: Build system
- **Python 3** with matplotlib: For result visualization (optional)
- **bear**: For generating compile_commands.json (optional, for IDE support)

## Project Structure

```
├── Data/              # IMU sensor data and ground truth
├── Results/           # Algorithm outputs and plots
├── include/           # Header files (.hpp)
├── src/               # Implementation files (.cpp)
├── bin/               # Compiled executables
├── build/             # Build artifacts
├── Pappers/           # Research papers and references
├── MDFiles/           # Documentation and equations
└── *FilterMain.cpp    # Entry points for each filter
```

## Data

All sensor data is located in the `Data/` directory from a 6-DOF IMU sensor.

### Input Files

| File | Description | Format | Units |
|------|-------------|--------|-------|
| `gyro.csv` | Angular velocity (3-axis) | 3 columns: X, Y, Z | rad/s |
| `accel.csv` | Linear acceleration (3-axis) | 3 columns: X, Y, Z | m/s² |
| `angles.csv` | Ground truth angles | Column 1: Roll, Column 2: Pitch | radians |

**Note**: Movement profile during data collection: (??) | Ground truth captured using: (??) 

## Build & Run

### Installation

Ensure Eigen3 is installed and available in system include paths (`/usr/local/include` or `/usr/include`):

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen
```
You can also download it from the release and extract to your preffered location, it's easy since it is only header library


### Build

```bash
# Clean previous builds
make clean

# Build all filters (parallel compilation)
make all -j

# Or build individual filters
make mahonyFilter        # Mahony filter only
make complemntaryFilter  # Complementary filter only
make ekfFilter           # EKF filter only
```

### Run

```bash
# Run Mahony filter (recommended - best current performance)
./bin/mahony.out

# Run Complementary filter
./bin/complmentary.out

# Run EKF filter (work in progress)
./bin/ekf.out
```

### Visualize Results

After running a filter, visualize the results:

```bash
cd Results
python plotDiagrams.py
```

Output plots are saved in `Results/Figures/`.


## Development Setup

### clangd for VSCode (Recommended)

VSCode's default C++ IntelliSense can be slow. For better performance, use clangd:

1. Install the **clangd** extension in VSCode
2. Keep the **C/C++** extension installed (needed for debugging)
3. Add to VSCode `settings.json`:  
```json
{
    // Disable Microsoft C++ IntelliSense
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled",
    
    // clangd settings
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--header-insertion=iwyu",
        "--completion-style=detailed",
        "--function-arg-placeholders",
        "--fallback-style=llvm"
    ],
    "clangd.path": "/usr/bin/clangd",
    
    // Keep C++ debugger functionality
    "C_Cpp.debugShortcut": true
}
```


### Generate compile_commands.json

For clangd to work properly, generate the compilation database:

```bash
# Install bear (if not already installed)
sudo apt install bear

# Generate compile_commands.json
make clean  # Important: must rebuild from scratch
bear -- make all
cat compile_commands.json  # Verify generation
```
