# RNN Hybrid Architecture for Attitude Estimation

**Status:** 🚀 PLANNING PHASE
**Date:** November 9, 2025
**Implementation Language:** C++ with libtorch (PyTorch C++ API)

---

## Overview

This document describes the hybrid RNN-based approach for attitude estimation that combines classical sensor fusion filters with deep learning for improved prediction accuracy.

---

## Architecture Design

### Hybrid Approach Concept

**Philosophy:** Use classical filter output as a feature for an RNN that learns temporal dynamics and refines predictions.

```
┌─────────────┐
│   IMU Data  │
│ (accel+gyro)│
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       v                 v
┌─────────────┐   ┌─────────────┐
│  Classical  │   │   Raw IMU   │
│   Filter    │   │  Features   │
│  (EKF/CF)   │   │             │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                │
                v
        ┌───────────────┐
        │  State Vector │
        │   (Input)     │
        │               │
        │ • Roll (k)    │
        │ • Pitch (k)   │
        │ • ax, ay, az  │
        │ • ωx, ωy, ωz  │
        └───────┬───────┘
                │
                v
        ┌───────────────┐
        │      RNN      │
        │   (LSTM/GRU)  │
        │               │
        │  Hidden State │
        │   Temporal    │
        │   Learning    │
        └───────┬───────┘
                │
                v
        ┌───────────────┐
        │ Predicted     │
        │ State (k+1)   │
        │               │
        │ • Roll (k+1)  │
        │ • Pitch (k+1) │
        │ • ωx, ωy, ωz  │
        └───────────────┘
```

---

## Input/Output Specification

### Input State Vector (at time k)

**Dimension:** 8D vector

| Index | Component | Source | Units | Description |
|-------|-----------|--------|-------|-------------|
| 0 | roll(k) | Filter output | radians | Current roll angle from classical filter |
| 1 | pitch(k) | Filter output | radians | Current pitch angle from classical filter |
| 2 | ax(k) | Accelerometer | m/s² | X-axis acceleration |
| 3 | ay(k) | Accelerometer | m/s² | Y-axis acceleration |
| 4 | az(k) | Accelerometer | m/s² | Z-axis acceleration |
| 5 | ωx(k) | Gyroscope | rad/s | X-axis angular velocity |
| 6 | ωy(k) | Gyroscope | rad/s | Y-axis angular velocity |
| 7 | ωz(k) | Gyroscope | rad/s | Z-axis angular velocity |

**Normalization:** All inputs should be normalized (details TBD)

### Output State Vector (at time k+1)

**Dimension:** 5D vector

| Index | Component | Units | Description |
|-------|-----------|-------|-------------|
| 0 | roll(k+1) | radians | Predicted roll angle at next timestep |
| 1 | pitch(k+1) | radians | Predicted pitch angle at next timestep |
| 2 | ωx(k+1) | rad/s | Predicted X-axis angular velocity |
| 3 | ωy(k+1) | rad/s | Predicted Y-axis angular velocity |
| 4 | ωz(k+1) | rad/s | Predicted Z-axis angular velocity |

**Rationale:** Predicting angular velocities helps the model learn dynamics and can be used for consistency checks.

---

## Classical Filter Selection

### Which Filter to Use?

We need to decide which classical filter's output to use as input features. Options:

| Filter | Roll RMSE | Pitch RMSE | Pros | Cons | Recommendation |
|--------|-----------|------------|------|------|----------------|
| **EKF** | 0.298° | 0.720° | Best accuracy, bias estimation | Most complex, might leave less for RNN to learn | ✅ **START HERE** |
| **Explicit CF** | 0.554° | 0.752° | Good accuracy, bias estimation | Middle ground | 🔄 **TRY SECOND** |
| **Mahony Passive** | 0.614° | 0.756° | Simpler, more room for improvement | No bias estimation | 🔄 **ALTERNATIVE** |
| **Complementary** | 0.820° | 0.771° | Simplest, most room for RNN | No bias, highest error | ❌ **LAST RESORT** |

### Recommended Strategy

1. **Phase 1:** Start with **EKF** as baseline
   - Best filter performance establishes upper bound
   - If RNN improves on EKF, we have strong contribution
   - EKF bias estimates provide rich features

2. **Phase 2:** Try **Explicit CF** as baseline
   - Compare if simpler filter + RNN outperforms pure EKF
   - More interesting for thesis (shows RNN can compensate for simpler filter)

3. **Phase 3 (Optional):** Try **Mahony Passive**
   - Explore if RNN can learn bias correction
   - Could show RNN replacing complex components

---

## Network Architecture

### RNN Type Options

**Option 1: LSTM (Recommended for start)**
```
Input (8D) → LSTM(hidden_size=64) → LSTM(hidden_size=32) → FC(5D)
```
- **Pros:** Handles long-term dependencies, well-established
- **Cons:** More parameters, slower training

**Option 2: GRU (Alternative)**
```
Input (8D) → GRU(hidden_size=64) → GRU(hidden_size=32) → FC(5D)
```
- **Pros:** Fewer parameters, faster training
- **Cons:** Slightly less expressive than LSTM

**Option 3: Simple RNN (Baseline)**
```
Input (8D) → RNN(hidden_size=64) → RNN(hidden_size=32) → FC(5D)
```
- **Pros:** Simplest, fastest
- **Cons:** Vanishing gradient issues

### Hyperparameters (Initial Guesses)

| Parameter | Initial Value | Notes |
|-----------|---------------|-------|
| **LSTM layers** | 2 | Stack two LSTM layers |
| **Hidden size (layer 1)** | 64 | Larger for more capacity |
| **Hidden size (layer 2)** | 32 | Smaller for refinement |
| **Output FC size** | 5 | Predicted state dimension |
| **Dropout** | 0.2 | Between LSTM layers |
| **Learning rate** | 0.001 | Adam optimizer |
| **Batch size** | 32 | Adjust based on sequence length |
| **Sequence length** | 20 | ~0.4 seconds at 50Hz |
| **Epochs** | 100-200 | Monitor validation loss |

---

## Data Preparation

### Dataset Structure

**Current Data:**
- Total samples: 1409
- Sample rate: 50 Hz (dt = 0.02s)
- Duration: ~28 seconds

**Train/Val/Test Split:**
```
Total: 1409 samples
├── Train: 985 samples (70%) - First 19.7 seconds
├── Validation: 211 samples (15%) - Next 4.2 seconds
└── Test: 213 samples (15%) - Last 4.2 seconds
```

**Important:** Use temporal split (not random) to avoid data leakage!

### Sequence Creation

For each sequence of length `seq_len = 20`:

**Input:**
```python
X[i] = [state[i], state[i+1], ..., state[i+seq_len-1]]  # Shape: (seq_len, 8)
```

**Target:**
```python
Y[i] = state[i+seq_len]  # Shape: (5,) - next state angles + velocities
```

**Number of sequences:**
- Train: 985 - 20 = 965 sequences
- Val: 211 - 20 = 191 sequences
- Test: 213 - 20 = 193 sequences

### Preprocessing Pipeline

```cpp
// Pseudocode
for each sample k:
    // 1. Get filter output
    roll[k], pitch[k] = selected_filter.getEstimation(k)

    // 2. Get IMU readings
    accel[k] = accelerometer_data[k]  // (ax, ay, az)
    gyro[k] = gyroscope_data[k]       // (ωx, ωy, ωz)

    // 3. Construct input vector
    input_state[k] = [roll[k], pitch[k], accel[k], gyro[k]]  // 8D

    // 4. Construct target vector
    target_state[k] = [roll[k+1], pitch[k+1], gyro[k+1]]  // 5D

    // 5. Normalize (z-score normalization)
    input_state[k] = (input_state[k] - mean) / std
    target_state[k] = (target_state[k] - target_mean) / target_std
```

---

## Loss Function

### Primary Loss: MSE on Angles

```python
loss_angles = MSE(predicted_angles, true_angles)
```

### Secondary Loss: MSE on Angular Velocities

```python
loss_velocities = MSE(predicted_velocities, true_velocities)
```

### Combined Loss

```python
total_loss = α × loss_angles + β × loss_velocities
```

**Suggested weights:**
- α = 1.0 (angles are primary objective)
- β = 0.1 (velocities are auxiliary, help learn dynamics)

### Evaluation Metrics

1. **Roll RMSE** (primary)
2. **Pitch RMSE** (primary)
3. **Combined RMSE** = (Roll RMSE + Pitch RMSE) / 2
4. **Angular velocity MAE** (secondary)

**Goal:** Beat EKF's 0.298° roll RMSE and 0.720° pitch RMSE

---

## Implementation Plan

### Phase 1: Data Preparation (C++)

**Files to create:**
- `include/RNNDataPreparator.hpp`
- `src/RNNDataPreparator.cpp`

**Tasks:**
1. Load filter outputs (roll, pitch) from existing results
2. Load IMU data (accel, gyro) from CSV files
3. Compute statistics (mean, std) on training set
4. Create sequences with sliding window
5. Normalize data
6. Save to format readable by libtorch (tensors or CSV)

### Phase 2: Network Definition (C++ with libtorch)

**Files to create:**
- `include/AttitudeRNN.hpp`
- `src/AttitudeRNN.cpp`

**Tasks:**
1. Define LSTM network architecture
2. Implement forward pass
3. Setup loss function
4. Setup optimizer (Adam)

### Phase 3: Training Loop (C++)

**Files to create:**
- `rnnTrainingMain.cpp`

**Tasks:**
1. Load preprocessed data
2. Create DataLoader for batching
3. Implement training loop
   - Forward pass
   - Loss computation
   - Backpropagation
   - Optimizer step
4. Implement validation loop
5. Save checkpoints
6. Log metrics

### Phase 4: Evaluation (C++)

**Files to create:**
- `rnnEvaluationMain.cpp`

**Tasks:**
1. Load trained model
2. Run inference on test set
3. Compute RMSE metrics
4. Compare with classical filters
5. Generate prediction plots
6. Save results

---

## libtorch Setup Requirements

### Installation

**Ubuntu/Linux:**
```bash
# Download libtorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Or GPU version (if CUDA available)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

**Windows (WSL):**
Same as Linux above.

### CMakeLists.txt Configuration

```cmake
cmake_minimum_required(VERSION 3.18)
project(AttitudeEstimationRNN)

set(CMAKE_CXX_STANDARD 14)

# Find libtorch
list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")
find_package(Torch REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# RNN Training executable
add_executable(rnnTraining
    rnnTrainingMain.cpp
    src/RNNDataPreparator.cpp
    src/AttitudeRNN.cpp
)
target_link_libraries(rnnTraining ${TORCH_LIBRARIES})

# RNN Evaluation executable
add_executable(rnnEval
    rnnEvaluationMain.cpp
    src/AttitudeRNN.cpp
)
target_link_libraries(rnnEval ${TORCH_LIBRARIES})
```

### Makefile Integration (Optional)

```makefile
# Add to existing Makefile
TORCH_PATH = /path/to/libtorch
TORCH_LIBS = -L$(TORCH_PATH)/lib -ltorch -lc10 -ltorch_cpu
TORCH_INCLUDES = -I$(TORCH_PATH)/include -I$(TORCH_PATH)/include/torch/csrc/api/include

rnnTraining: rnnTrainingMain.cpp src/RNNDataPreparator.cpp src/AttitudeRNN.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TORCH_INCLUDES) $^ -o bin/rnnTraining.out $(TORCH_LIBS)
```

---

## Expected Outcomes

### Success Criteria

**Minimum Viable:**
- RNN converges during training (loss decreases)
- Test RMSE better than baseline filter used as input
- Results are reproducible

**Good Success:**
- RNN beats best complementary filter (Explicit CF: 0.554° roll)
- RNN approaches EKF performance (0.298° roll)

**Excellent Success:**
- RNN beats EKF on roll (< 0.298°)
- RNN beats EKF on pitch (< 0.720°)
- RNN shows consistent improvement across different motion dynamics

### Potential Challenges

1. **Small dataset:** Only 1409 samples may lead to overfitting
   - **Solution:** Strong regularization (dropout, early stopping)
   - **Solution:** Data augmentation (if possible)

2. **Temporal dependencies:** Need to preserve sequence structure
   - **Solution:** Use temporal split, not random split

3. **Normalization:** Critical for stable training
   - **Solution:** Careful z-score normalization per feature

4. **Overfitting:** Model may memorize training sequences
   - **Solution:** Monitor validation loss closely
   - **Solution:** Use dropout, reduce model capacity if needed

---

## Alternative Approaches (Future Work)

### 1. Attention Mechanism
```
Input → LSTM → Attention → FC → Output
```
- Learn which timesteps are most important

### 2. Transformer Architecture
```
Input → Positional Encoding → Multi-Head Attention → FC → Output
```
- State-of-the-art for sequence modeling
- May require more data

### 3. Multi-Task Learning
```
        ┌─→ Angle Prediction (primary)
Input → RNN ┼─→ Velocity Prediction
        └─→ Bias Estimation
```
- Learn related tasks jointly

### 4. Ensemble
```
Filter 1 ──┐
Filter 2 ──┼─→ RNN → Weighted Fusion → Final Output
Filter 3 ──┘
```
- Use multiple filters as input

---

## File Structure (Proposed)

```
├── include/
│   ├── RNNDataPreparator.hpp      # Data loading and preprocessing
│   └── AttitudeRNN.hpp            # Network definition
├── src/
│   ├── RNNDataPreparator.cpp
│   └── AttitudeRNN.cpp
├── rnnTrainingMain.cpp            # Training script
├── rnnEvaluationMain.cpp          # Evaluation script
├── RNNData/                       # Preprocessed data for RNN
│   ├── train_sequences.pt         # Training sequences (libtorch tensor)
│   ├── val_sequences.pt           # Validation sequences
│   ├── test_sequences.pt          # Test sequences
│   └── normalization_stats.txt    # Mean and std for denormalization
├── RNNModels/                     # Saved models
│   ├── best_model.pt              # Best validation checkpoint
│   └── training_log.txt           # Training metrics
└── MDFiles/
    └── RNN_Hybrid_Architecture.md # This document
```

---

## Next Steps

### Immediate Actions

1. **Install libtorch** on your system
2. **Decide which filter** to use as baseline (recommend: EKF)
3. **Implement RNNDataPreparator** to create training sequences
4. **Define network architecture** in AttitudeRNN class
5. **Test forward pass** with dummy data

### Questions to Answer

- [ ] Which filter should we use as baseline? (EKF vs Explicit CF vs Mahony)
- [ ] Do we have access to more IMU data? (1409 samples is small for deep learning)
- [ ] Should we predict only angles, or also angular velocities?
- [ ] What sequence length should we use? (20 samples = 0.4s is initial guess)
- [ ] CPU or GPU training? (GPU much faster if available)

### Timeline Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Setup** | 1 day | Install libtorch, verify installation |
| **Data Prep** | 1-2 days | Implement RNNDataPreparator, create sequences |
| **Network** | 1 day | Define AttitudeRNN architecture |
| **Training** | 2-3 days | Implement training loop, hyperparameter tuning |
| **Evaluation** | 1 day | Test set evaluation, comparison plots |
| **Documentation** | 1 day | Update thesis, create result figures |
| **Total** | **7-10 days** | End-to-end RNN implementation |

---

## References

### libtorch Documentation
- Official Tutorial: https://pytorch.org/cppdocs/
- Installation Guide: https://pytorch.org/get-started/locally/
- C++ API Reference: https://pytorch.org/cppdocs/api/library_root.html

### RNN for IMU Papers
1. "Deep Learning for Sensor-based Activity Recognition" - Survey of RNN approaches
2. "IONet: Learning to Cure the Curse of Drift in Inertial Odometry" - RNN for IMU drift correction
3. "RIDI: Robust IMU Double Integration" - Learning-based IMU processing

---

**Document Version:** 1.0
**Last Updated:** November 9, 2025
**Status:** Ready for implementation
