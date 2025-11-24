# RNN for IMU Attitude Estimation: Complete Feasibility Analysis

**Date:** October 29, 2025
**Project:** MSc Thesis - Attitude Estimation using IMU Sensor Fusion
**Author:** Analysis for thesis decision-making

---

## Table of Contents

1. [Current Situation](#current-situation)
2. [Option 1: Python-Based RNN](#option-1-python-based-rnn)
3. [Option 2: C++ Based RNN](#option-2-c-based-rnn)
4. [Option 3: Alternative Routes](#option-3-alternative-routes-no-rnn)
5. [Recommendation Matrix](#recommendation-matrix)
6. [Recommended Strategy](#recommended-strategy)
7. [Detailed Implementation: Hybrid EKF+LSTM](#detailed-implementation-hybrid-ekflstm)
8. [Final Recommendations](#final-recommendations)

---

## Current Situation

### What You Have

- **Dataset:** 1409 samples (~28 seconds at 50Hz)
- **Motion Profile:** Single trajectory
- **Implemented Filters:** 3 classical algorithms
  - Complementary Filter (α = 0.79)
  - Mahony Passive Filter (kp = 9)
  - Extended Kalman Filter (EKF)
- **Ground Truth:** angles.csv provides labeled data
- **Codebase:** C++ with Eigen library
- **Strong Baseline:** EKF achieves 0.298° roll RMSE

### Current Performance (RMSE in degrees)

| Filter | Roll RMSE | Pitch RMSE |
|--------|-----------|------------|
| **Complementary** | 0.820° | 0.771° |
| **Mahony** | 0.589° | 1.430° |
| **EKF** | 0.298° | 0.720° |

### Constraints

- ❌ Limited training data diversity (one flight profile)
- ❌ MSc thesis timeline pressure
- ✅ Need to justify complexity vs EKF baseline
- ✅ Must demonstrate research awareness

---

## Option 1: Python-Based RNN

### 1A. Simple LSTM Proof-of-Concept ⭐ RECOMMENDED STARTER

**Implementation Time:** 2-3 days

#### Technology Stack
- **Framework:** PyTorch or TensorFlow/Keras
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Language:** Python

#### Architecture

```
Input: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
       Shape: (batch_size, sequence_length=10-50, 6 features)
       ↓
LSTM Layer (hidden_size=64, num_layers=2)
       ↓
Fully Connected Layer (64 → 2)
       ↓
Output: [roll, pitch] in degrees
```

#### Code Estimate
~100-150 lines of Python

#### Expected Performance
- **Roll RMSE:** 0.4-0.6° (likely similar to Mahony)
- **Pitch RMSE:** 0.8-1.2°
- **Outcome:** Unlikely to beat EKF due to limited training data

#### Pros
✅ Fast to implement
✅ Standard deep learning workflow
✅ Easy to visualize training curves (loss, RMSE over epochs)
✅ Demonstrates "modern ML approach" awareness
✅ Good for "Exploratory Analysis" thesis section
✅ Low risk implementation

#### Cons
❌ Language mismatch with C++ project
❌ Won't integrate with real-time C++ filters
❌ Likely overfits to single trajectory
❌ Hard to justify superiority over well-tuned EKF
❌ Limited novelty

#### Thesis Value
⭐⭐⭐ (Moderate - shows breadth of knowledge)

#### Sample Implementation Outline

```python
import torch
import torch.nn as nn
import numpy as np

# 1. Load Data
gyro = np.loadtxt('Data/gyro.csv', delimiter=',')
accel = np.loadtxt('Data/accel.csv', delimiter=',')
angles = np.loadtxt('Data/angles.csv', delimiter=',')

# 2. Create sliding windows
def create_sequences(imu_data, labels, window_size=20):
    X, y = [], []
    for i in range(len(imu_data) - window_size):
        X.append(imu_data[i:i+window_size])
        y.append(labels[i+window_size])  # Predict current angles
    return np.array(X), np.array(y)

# 3. Model
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=64,
                           num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)  # Roll, Pitch

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 4. Train/Val/Test split: 70/15/15
# 5. Train for 100-200 epochs
# 6. Evaluate RMSE vs EKF baseline
```

---

### 1B. Attention-LSTM ⚠️ HIGH RISK, HIGH REWARD

**Implementation Time:** 5-7 days

#### Technology Stack
- PyTorch (more flexible for custom attention)
- NumPy, Matplotlib

#### Architecture

```
Input: IMU window (batch, sequence_length, 6)
       ↓
1D-CNN (extract local temporal patterns)
       ↓
Bidirectional LSTM (capture temporal dependencies)
       ↓
Multi-Head Attention (focus on important timesteps)
       ↓
Fully Connected → [roll, pitch]
```

#### Code Estimate
~300-400 lines of Python

#### Expected Performance
- **Roll RMSE:** 0.35-0.5° (marginal improvement over simple LSTM)
- **Pitch RMSE:** 0.7-1.0°
- **Likely constrained by limited data diversity**

#### Pros
✅ State-of-the-art architecture (aligns with 2024-2025 research)
✅ Attention weights provide interpretability
✅ Strong thesis differentiator
✅ Can visualize which timesteps/sensors model focuses on
✅ Publishable if successful

#### Cons
❌ Complex implementation (debugging attention mechanisms)
❌ Needs 10-50x more training data to truly shine
❌ Risk: More development time, potentially similar results
❌ Hard to train effectively with only 1409 samples
❌ Overfitting likely without data augmentation

#### Thesis Value
⭐⭐⭐⭐ (High academic interest, execution risk)

#### Research Alignment
- **2025 Paper:** "Attention-CNN-LSTM for MEMS Gyroscope Denoising"
- **Trend:** Attention mechanisms for adaptive sensor weighting
- **Gap:** Most papers use 80-120 hours of flight data (you have ~28 seconds)

---

### 1C. Hybrid EKF + LSTM Correction ⭐⭐ BEST OPTION

**Implementation Time:** 4-5 days

#### Core Concept
Instead of replacing EKF, **learn to correct its errors:**
1. Run existing EKF implementation
2. Train LSTM to predict residual error
3. Final output = EKF prediction + LSTM correction

This is a **physics-informed machine learning** approach.

#### Technology Stack
- Python (PyTorch/TensorFlow)
- Load EKF results from existing txt files
- NumPy for data processing

#### Architecture

```
Input Features (per timestep):
  - EKF roll estimate
  - EKF pitch estimate
  - Gyro [x, y, z]
  - Accel [x, y, z]
  - (Optional) EKF uncertainty estimate

  Shape: (batch, window=20, 8-10 features)
       ↓
LSTM (learns when EKF systematically fails)
       ↓
Output: [roll_correction, pitch_correction]

Final Estimate = EKF + LSTM_correction
```

#### Expected Performance
- **Roll RMSE:** 0.25-0.35° (potentially beats EKF!)
- **Pitch RMSE:** 0.6-0.8° (likely beats EKF)
- **Most realistic chance of measurable improvement**

#### Why This Works Better

1. **Smaller learning problem:** Residual errors are smaller than absolute angles
   - Easier to learn: "When gyro_z spikes, EKF underestimates by 0.1°"
   - vs learning: "Given raw IMU, angle is 15.3°"

2. **Works with limited data:**
   - EKF provides strong prior
   - LSTM only learns systematic biases/patterns
   - Less prone to overfitting

3. **Physics + Data synergy:**
   - EKF handles well-modeled dynamics
   - LSTM handles unmodeled effects (sensor nonlinearities, temperature drift)

#### Pros
✅ **Highest probability of beating EKF baseline**
✅ Physics-informed approach (combines classical + ML)
✅ Learns only the "hard parts" (residuals)
✅ Works effectively with limited data
✅ Aligns with "Deep Kalman Filter" research trend
✅ Strong thesis contribution if successful
✅ Novel approach for MSc level

#### Cons
❌ Requires exporting/loading EKF predictions
❌ More complex experimental setup
❌ Still Python-only (no C++ integration)
❌ Interpretation requires understanding both EKF and LSTM

#### Thesis Value
⭐⭐⭐⭐⭐ (Best trade-off: feasible + novel + high success probability)

#### Success Probability
- **60%:** Modest improvement (5-15% RMSE reduction)
- **30%:** Neutral (similar to EKF, demonstrates approach viability)
- **10%:** Regression (overfitting, but still publishable as negative result)

---

## Option 2: C++ Based RNN

### 2A. Manual C++ LSTM Implementation ❌ NOT RECOMMENDED

**Implementation Time:** 10-15 days

#### Approach
- Implement LSTM forward pass from scratch in C++ with Eigen
- Manually code matrix operations (gates, activations, etc.)
- Train in Python (still needed for backpropagation)
- Export weights to JSON/binary format
- Load weights in C++ for inference only

#### Code Estimate
~800-1200 lines of C++ (highly error-prone)

#### Libraries Needed
- Eigen (matrix operations) - already have
- nlohmann/json (weight loading)
- Custom implementation of sigmoid, tanh activations

#### Pros
✅ Fully consistent with C++ project
✅ Real-time inference capable
✅ Educational (deep understanding of LSTM internals)
✅ No Python dependency at runtime
✅ Complete control over implementation

#### Cons
❌ **Extremely time-consuming** (2 weeks minimum)
❌ High error rate (manual gradient checking, dimension bugs)
❌ Training still requires Python/PyTorch anyway
❌ Debugging backpropagation in C++ is painful
❌ Not worth time investment for MSc thesis
❌ No scientific contribution (reimplementing existing methods)

#### Thesis Value
⭐⭐ (Low - lots of engineering work, minimal research value)

#### Verdict
**SKIP THIS.** Use Python for training, only consider C++ if inference integration is critical.

---

### 2B. C++ Inference with ONNX Runtime ⚠️ OVERKILL

**Implementation Time:** 5-7 days

#### Approach
1. Train LSTM in PyTorch or TensorFlow
2. Export model to ONNX format (standard interchange format)
3. Use ONNX Runtime library in C++ for inference
4. Integrate with existing C++ filter codebase

#### Technology Stack
- **Training:** Python + PyTorch
- **Export:** ONNX (Open Neural Network Exchange)
- **Inference:** C++ ONNX Runtime
- **Integration:** Eigen for data preprocessing

#### Code Estimate
- Python training: ~150 lines
- C++ inference wrapper: ~200-300 lines
- CMake integration: ~50 lines

#### Workflow

```
Python Side:
  train_model.py → trained_lstm.pth → export_to_onnx() → lstm_model.onnx

C++ Side:
  lstm_model.onnx → ONNXRuntime::LoadModel()
                 → PrepareInputTensor()
                 → RunInference()
                 → [roll, pitch]
```

#### Pros
✅ Best of both worlds (easy training, C++ integration)
✅ Production-quality approach (ONNX is industry standard)
✅ Consistent codebase (all executables in C++)
✅ Real-time capable
✅ Cross-platform (ONNX Runtime supports Windows/Linux)

#### Cons
❌ ONNX Runtime is a large dependency (~100MB library)
❌ Cross-platform build complexity (CMake configuration)
❌ Overkill for academic research thesis
❌ Still won't beat EKF with limited training data
❌ Adds significant build complexity

#### Thesis Value
⭐⭐⭐ (Good software engineering, questionable research value)

#### When to Use This
- If you plan to deploy on embedded hardware
- If C++ integration is thesis requirement
- If you have time for "bonus engineering section"

**For MSc thesis:** Probably not worth the complexity.

---

### 2C. Lightweight C++ Libraries ⚠️ MEDIOCRE OPTION

**Implementation Time:** 4-6 days

#### Library Options

**Option 1: tiny-dnn**
- Header-only C++ neural network library
- Can train and infer in C++
- GitHub: https://github.com/tiny-dnn/tiny-dnn

**Option 2: Frugally-Deep**
- Keras model loader for C++
- Inference only (train in Python/Keras)
- Header-only library
- GitHub: https://github.com/Dobiasd/frugally-deep

#### Approach (Frugally-Deep Example)

```
Python Side:
  1. Train in Keras
  2. model.save('lstm_model.h5')
  3. Convert: keras_to_fdeep.py lstm_model.h5 lstm_model.json

C++ Side:
  #include <fdeep/fdeep.hpp>

  auto model = fdeep::load_model("lstm_model.json");
  auto result = model.predict({imu_input_tensor});
```

#### Code Estimate
~200-300 lines C++

#### Pros
✅ Easier than manual LSTM implementation
✅ Header-only (simple CMake integration)
✅ Maintains C++ consistency
✅ Reasonable learning curve

#### Cons
❌ tiny-dnn: Less mature, limited LSTM support, slower
❌ Frugally-deep: Inference only, still need Python for training
❌ Less flexible than native PyTorch/TensorFlow
❌ Smaller community (fewer tutorials, support)
❌ May have version compatibility issues

#### Thesis Value
⭐⭐⭐ (Moderate - compromise solution)

#### Verdict
**Consider only if:** C++ integration is required but ONNX is too heavy.

---

## Option 3: Alternative Routes (No RNN)

### 3A. "Future Work" Section Only ⭐ SAFE FALLBACK

**Implementation Time:** 2-3 hours (writing only)

#### Approach

Write a comprehensive "Future Work" section that demonstrates research awareness without implementation.

#### Content to Include

1. **Literature Review:**
   - Cite recent papers (2024-2025) on attention-LSTM for IMU
   - Reference "Deep Learning for Inertial Positioning" surveys
   - Mention hybrid Deep Kalman Filter approaches

2. **Data Limitations:**
   - Acknowledge single trajectory limitation (1409 samples)
   - Specify needed diversity: 100+ hours, multiple flight profiles
   - Discuss overfitting risks with current dataset

3. **Proposed Architecture:**
   - Outline Hybrid EKF+LSTM approach
   - Explain physics-informed ML rationale
   - Describe expected benefits (bias learning, adaptive weighting)

4. **Implementation Roadmap:**
   - Data collection requirements
   - Training/validation/test split strategy
   - Evaluation metrics beyond RMSE

#### Example Section Outline

```markdown
## 7. Future Work: Deep Learning Approaches

Recent advances in RNN-based attitude estimation (Zhang et al. 2025)
show promise for adaptive sensor fusion. However, such approaches
require extensive training data across diverse motion profiles.

### 7.1 Hybrid EKF-LSTM Architecture

We propose a physics-informed approach where an LSTM learns to
correct systematic EKF errors...

### 7.2 Data Requirements

Effective training would require:
- 100+ hours of flight data
- Multiple aircraft/motion profiles
- Diverse environmental conditions

### 7.3 Expected Benefits

- Automatic gyro bias adaptation
- Learned sensor weighting based on motion context
- Potential 20-40% RMSE improvement over classical EKF
```

#### Pros
✅ Zero implementation risk
✅ Demonstrates research awareness
✅ Shows critical thinking about limitations
✅ Honest about data constraints
✅ Can cite cutting-edge papers
✅ Takes only a few hours

#### Cons
❌ No experimental validation
❌ No novel contribution
❌ May seem like avoiding the problem

#### Thesis Value
⭐⭐⭐⭐ (High - honest, well-researched analysis)

#### When to Use
- Limited time remaining on thesis
- Professor agrees current filters are sufficient
- Want to show ML awareness without implementation

---

### 3B. Focus on Mahony Pitch Improvement ⭐ QUICK WIN

**Implementation Time:** 1-2 days

#### The Problem

Your Mahony filter has:
- **Roll RMSE:** 0.589° (excellent, beats Complementary)
- **Pitch RMSE:** 1.430° (poor, worst of all filters)

**Root cause:** kp was tuned ONLY for roll performance (see mahonyFilterMain.cpp:55)

#### Approaches to Fix

**Approach 1: Re-tune for Combined Metric**

```cpp
// Current code (mahonyFilterMain.cpp:55)
double tmpBetRmse = Utils::rmse(rollGroundTruth, mahonyTmp.getRollEstimation());

// Improved: tune for both roll AND pitch
double rollRmse = Utils::rmse(rollGroundTruth, mahonyTmp.getRollEstimation());
double pitchRmse = Utils::rmse(pitchGroundTruth, mahonyTmp.getPitchEstimation());
double tmpBetRmse = (rollRmse + pitchRmse) / 2.0;  // Average
// Or weighted: 0.6*rollRmse + 0.4*pitchRmse if roll is more important
```

**Expected outcome:**
- kp will likely increase to 15-30
- Roll RMSE: 0.589° → 0.7° (slight degradation)
- Pitch RMSE: 1.430° → 0.9° (major improvement)
- **More balanced performance**

**Approach 2: Axis-Specific Gains (Advanced)**

Modify MahonyFilter to use different gains for roll vs pitch corrections:

```cpp
// In MahonyFilter.hpp
double kp_roll;
double kp_pitch;

// In update function
Eigen::Vector3d omega_mes = Utils::vexFromSkewMatrix(Pa_R_tilde);
// Weight differently based on which angle is being corrected
omega_mes(0) *= kp_roll;   // Roll axis
omega_mes(1) *= kp_pitch;  // Pitch axis
```

**Expected outcome:**
- kp_roll ≈ 9, kp_pitch ≈ 20-30
- Roll RMSE: ~0.6° (maintained)
- Pitch RMSE: 1.43° → 0.8-0.9° (major improvement)

**Approach 3: Add Integral Term (Full Mahony Filter)**

Implement the complete Mahony filter with bias estimation:

```cpp
// Add to MahonyFilter.hpp
Eigen::Vector3d bias;  // Gyro bias estimate
double ki;             // Integral gain

// In update function
bias += ki * omega_mes * dt;  // Accumulate bias
Eigen::Vector3d omega_corrected = omega_y - bias + kp * omega_mes;
```

**Expected outcome:**
- With proper tuning (kp=15, ki=0.1)
- Roll RMSE: 0.589° → 0.4-0.5° (improvement!)
- Pitch RMSE: 1.43° → 0.8-1.0° (major improvement)
- Now competitive with EKF

#### Pros
✅ Quick implementation (1-2 days max)
✅ Solid engineering analysis
✅ Demonstrates parameter sensitivity understanding
✅ Fixes obvious problem in current implementation
✅ Good thesis content ("Optimization and Comparative Analysis")

#### Cons
❌ Not cutting-edge (classical methods)
❌ Won't beat EKF significantly
❌ Less exciting than ML approaches

#### Thesis Value
⭐⭐⭐⭐ (High - demonstrates thoroughness)

#### Verdict
**DO THIS REGARDLESS.** It's a quick win that improves your baseline results.

---

## Recommendation Matrix

| Approach | Time | Feasibility | Beat EKF? | Thesis Impact | Risk Level | Overall Score |
|----------|------|-------------|-----------|---------------|------------|---------------|
| **Python Simple LSTM** | 2-3 days | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐ | Low | ✅ **Good Starter** |
| **Python Attention-LSTM** | 5-7 days | ⭐⭐⭐ | ❌ Unlikely | ⭐⭐⭐⭐ | High | ⚠️ Risky |
| **Hybrid EKF+LSTM (Python)** | 4-5 days | ⭐⭐⭐⭐ | ✅ Maybe | ⭐⭐⭐⭐⭐ | Medium | ✅✅ **BEST CHOICE** |
| **C++ Manual LSTM** | 10-15 days | ⭐ | ❌ No | ⭐⭐ | Very High | ❌ Avoid |
| **C++ ONNX Runtime** | 5-7 days | ⭐⭐⭐ | ❌ No | ⭐⭐⭐ | Medium | ⚠️ Overkill |
| **C++ Lightweight Libs** | 4-6 days | ⭐⭐ | ❌ No | ⭐⭐⭐ | Medium | ⚠️ Meh |
| **Future Work Only** | 3 hours | ⭐⭐⭐⭐⭐ | N/A | ⭐⭐⭐⭐ | None | ✅ **Safe Fallback** |
| **Improve Mahony** | 1-2 days | ⭐⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐ | Low | ✅ **Quick Win** |

### Legend
- ⭐⭐⭐⭐⭐ Excellent
- ⭐⭐⭐⭐ Good
- ⭐⭐⭐ Moderate
- ⭐⭐ Poor
- ⭐ Very Poor

---

## Recommended Strategy

### Two-Tier Approach: Low Risk + High Reward

---

### TIER 1: Essential Work (3 days total) - DO THIS FIRST

These tasks have **high value, low risk, and guaranteed results.**

#### Task 1: Fix Mahony Pitch Problem (1 day)

**File:** `mahonyFilterMain.cpp`

**Changes:**
1. Modify tuning loop to optimize for combined roll+pitch RMSE
2. Re-run parameter search: kp ∈ [1, 100]
3. Update results files

**Expected outcome:**
- New optimal kp ≈ 15-25
- Pitch RMSE: 1.43° → 0.8-0.9°
- Roll RMSE: 0.59° → 0.65-0.75° (acceptable trade-off)

**Deliverable:** Updated Mahony results, parameter sensitivity analysis

---

#### Task 2: Python Simple LSTM Experiment (2 days)

**Objective:** Demonstrate awareness of modern ML approaches

**Implementation:**
1. **Day 1:** Data preparation, model setup, initial training
   - Create sliding windows from IMU data
   - Implement basic LSTM in PyTorch/TensorFlow
   - Train/val/test split (70/15/15)

2. **Day 2:** Training, evaluation, documentation
   - Train for 100-200 epochs
   - Calculate RMSE on test set
   - Generate comparison plots
   - Write "Exploratory ML Analysis" section

**Expected RMSE:**
- Roll: 0.4-0.6°
- Pitch: 0.8-1.2°
- Likely between Mahony and EKF

**Deliverable:**
- Trained model
- Performance comparison chart
- 2-3 page thesis section
- Training/validation curves

**Success criteria:**
- RMSE within 20% of EKF (acceptable)
- Demonstrates understanding of ML limitations
- Honest discussion of overfitting risks

---

### TIER 2: Advanced Work (4-5 days) - IF TIME PERMITS

Only pursue this if:
- Tier 1 is complete and successful
- You have 1 week of dedicated time available
- Professor approves the approach

#### Task 3: Hybrid EKF+LSTM Correction (4-5 days)

**Objective:** Attempt to beat EKF baseline using physics-informed ML

**Why this has highest success probability:**
- Smaller learning problem (residuals vs absolute angles)
- EKF provides strong prior
- Works with limited data

**Implementation plan:** See [detailed section below](#detailed-implementation-hybrid-ekflstm)

**Expected RMSE:**
- Roll: 0.25-0.35° (10-20% improvement over EKF)
- Pitch: 0.6-0.8° (10-15% improvement)

**Deliverable:**
- Hybrid model implementation
- Comparative analysis with pure EKF
- Discussion of when/why corrections help
- Visualization of residual error patterns

**Risk mitigation:**
- Even if RMSE doesn't improve, approach is publishable
- Can analyze which scenarios benefit from ML correction
- Demonstrates sophisticated understanding of hybrid methods

---

### Timeline Summary

**Minimum viable thesis (Tier 1 only):**
- Day 1: Fix Mahony
- Days 2-3: Simple LSTM
- **Total: 3 days**
- **Result:** Solid classical implementation + ML exploration

**Enhanced thesis (Tier 1 + Tier 2):**
- Days 1-3: Tier 1 work
- Days 4-8: Hybrid EKF+LSTM
- **Total: 8 days**
- **Result:** Potential novel contribution, publishable

**What to skip:**
- ❌ All C++ RNN implementations (not worth time)
- ❌ Attention mechanisms (insufficient data)
- ❌ Manual LSTM from scratch (no research value)

---

### Decision Tree

```
START
  ↓
Have you fixed Mahony pitch?
  NO → Do Task 1 first (1 day)
  YES ↓
       ↓
Do you have 3+ days available?
  NO → Write "Future Work" section only
  YES ↓
       ↓
Implement Simple LSTM (Task 2, 2 days)
       ↓
Did it work reasonably well?
  NO → Analyze why, write up findings, STOP
  YES ↓
       ↓
Do you have 5+ additional days AND professor approval?
  NO → STOP, write thesis
  YES ↓
       ↓
Implement Hybrid EKF+LSTM (Task 3, 4-5 days)
       ↓
Write comprehensive comparative analysis
       ↓
DONE
```

---

## Detailed Implementation: Hybrid EKF+LSTM

This section provides a complete implementation guide for the most promising approach.

### Overview

**Core Idea:** Train an LSTM to learn and correct systematic EKF errors.

**Mathematical formulation:**
```
θ_final = θ_EKF + f_LSTM(θ_EKF, ω, a, history)

where:
  θ_final = corrected roll/pitch estimate
  θ_EKF = raw EKF estimate
  f_LSTM = learned correction function
  ω = gyro measurements
  a = accel measurements
  history = past window of measurements
```

---

### Step 1: Data Preparation (1 hour)

#### File: `hybrid_lstm_data_prep.py`

```python
import numpy as np

def prepare_hybrid_data(window_size=20):
    """
    Prepare data for EKF+LSTM hybrid approach.

    Returns:
        X: Input features (N, window_size, num_features)
        y: Target corrections (N, 2) for [roll, pitch]
    """

    # Load existing EKF predictions
    ekf_roll = np.loadtxt('Results/Results/EkfRoll.txt')
    ekf_pitch = np.loadtxt('Results/Results/EkfPitch.txt')

    # Load ground truth
    truth_roll = np.loadtxt('Results/ExpectedResults/expected_roll.txt')
    truth_pitch = np.loadtxt('Results/ExpectedResults/expected_pitch.txt')

    # Compute residual errors (what LSTM needs to learn)
    residual_roll = truth_roll - ekf_roll
    residual_pitch = truth_pitch - ekf_pitch

    print(f"Residual roll  - Mean: {np.mean(residual_roll):.4f}, Std: {np.std(residual_roll):.4f}")
    print(f"Residual pitch - Mean: {np.mean(residual_pitch):.4f}, Std: {np.std(residual_pitch):.4f}")

    # Load raw IMU data
    gyro = np.loadtxt('Data/gyro.csv', delimiter=',')   # (N, 3)
    accel = np.loadtxt('Data/accel.csv', delimiter=',') # (N, 3)

    # Feature engineering: include EKF estimates as features
    # Features per timestep: [ekf_roll, ekf_pitch, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
    features_per_timestep = np.column_stack([
        ekf_roll.reshape(-1, 1),
        ekf_pitch.reshape(-1, 1),
        gyro,
        accel
    ])  # Shape: (N, 8)

    # Create sliding windows
    X, y_roll, y_pitch = [], [], []

    for i in range(window_size, len(features_per_timestep)):
        # Input: window of past measurements
        X.append(features_per_timestep[i-window_size:i, :])

        # Target: residual error at current timestep
        y_roll.append(residual_roll[i])
        y_pitch.append(residual_pitch[i])

    X = np.array(X)  # (N-window_size, window_size, 8)
    y = np.column_stack([y_roll, y_pitch])  # (N-window_size, 2)

    print(f"\nDataset shape:")
    print(f"  X (inputs):  {X.shape}")
    print(f"  y (targets): {y.shape}")

    # Normalize features (important for neural network training)
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std

    # Save normalization parameters for inference
    np.save('models/X_mean.npy', X_mean)
    np.save('models/X_std.npy', X_std)

    return X_normalized, y

if __name__ == "__main__":
    X, y = prepare_hybrid_data(window_size=20)
    np.save('models/X_train.npy', X)
    np.save('models/y_train.npy', y)
    print("Data preparation complete!")
```

**Key insights:**
- Residuals are typically small (< 1°)
- Smaller targets → easier to learn
- EKF estimates included as features (model learns "when is EKF wrong?")

---

### Step 2: Model Architecture (30 minutes)

#### File: `hybrid_lstm_model.py`

```python
import torch
import torch.nn as nn

class EKF_LSTM_Corrector(nn.Module):
    """
    LSTM that learns to correct EKF residual errors.

    Architecture:
      Input: (batch, seq_len=20, features=8)
      LSTM: 2 layers, hidden_size=32
      Output: (batch, 2) for [roll_correction, pitch_correction]
    """

    def __init__(self, input_size=8, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 2)  # Output: [roll_correction, pitch_correction]

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        correction = self.fc2(out)  # (batch, 2)

        return correction

    def predict_correction(self, imu_window, X_mean, X_std):
        """
        Predict correction for a single window.

        Args:
            imu_window: numpy array (seq_len, features)
            X_mean, X_std: normalization parameters

        Returns:
            correction: [roll_corr, pitch_corr] in degrees
        """
        self.eval()
        with torch.no_grad():
            # Normalize
            x_norm = (imu_window - X_mean) / X_std

            # Convert to tensor
            x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)  # (1, seq_len, features)

            # Predict
            correction = self.forward(x_tensor)

            return correction.numpy()[0]  # (2,)

# Model summary
if __name__ == "__main__":
    model = EKF_LSTM_Corrector()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Test forward pass
    dummy_input = torch.randn(4, 20, 8)  # batch=4, seq=20, features=8
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (4, 2)
```

**Architecture justification:**
- **Small model** (~3K parameters) → less prone to overfitting
- **2-layer LSTM** → captures temporal dependencies
- **Dropout** → regularization for limited data
- **Simple FC head** → fast convergence

---

### Step 3: Training Script (1 hour)

#### File: `train_hybrid_lstm.py`

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hybrid_lstm_model import EKF_LSTM_Corrector

def train_hybrid_model():
    # Load prepared data
    X = np.load('models/X_train.npy')  # (N, 20, 8)
    y = np.load('models/y_train.npy')  # (N, 2)

    print(f"Dataset: {X.shape[0]} samples")

    # Train/Val/Test split: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EKF_LSTM_Corrector().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    num_epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_hybrid_lstm.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Hybrid EKF+LSTM Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('Results/Figures/hybrid_lstm_training.png', dpi=150)
    print("Training curve saved!")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('models/best_hybrid_lstm.pth'))
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()

    # Calculate test RMSE
    rmse_roll = np.sqrt(np.mean((y_pred[:, 0] - y_test[:, 0])**2))
    rmse_pitch = np.sqrt(np.mean((y_pred[:, 1] - y_test[:, 1])**2))

    print(f"\n=== Test Set Performance ===")
    print(f"Correction RMSE - Roll: {rmse_roll:.4f}°, Pitch: {rmse_pitch:.4f}°")

    return model

if __name__ == "__main__":
    model = train_hybrid_model()
```

**Training strategy:**
- **Early stopping** → prevent overfitting
- **Learning rate scheduling** → fine-tune convergence
- **MSE loss** → regression problem
- **Adam optimizer** → fast convergence

---

### Step 4: Evaluation and Comparison (2 hours)

#### File: `evaluate_hybrid_lstm.py`

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from hybrid_lstm_model import EKF_LSTM_Corrector

def evaluate_hybrid_approach():
    """
    Compare EKF vs Hybrid EKF+LSTM on full dataset.
    """

    # Load data
    ekf_roll = np.loadtxt('Results/Results/EkfRoll.txt')
    ekf_pitch = np.loadtxt('Results/Results/EkfPitch.txt')
    truth_roll = np.loadtxt('Results/ExpectedResults/expected_roll.txt')
    truth_pitch = np.loadtxt('Results/ExpectedResults/expected_pitch.txt')

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EKF_LSTM_Corrector().to(device)
    model.load_state_dict(torch.load('models/best_hybrid_lstm.pth'))
    model.eval()

    # Load normalization parameters
    X_mean = np.load('models/X_mean.npy')
    X_std = np.load('models/X_std.npy')

    # Load prepared features
    X = np.load('models/X_train.npy')

    # Predict corrections for all samples
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        corrections = model(X_tensor).cpu().numpy()  # (N, 2)

    # Apply corrections (skip first 'window_size' samples)
    window_size = 20
    hybrid_roll = ekf_roll.copy()
    hybrid_pitch = ekf_pitch.copy()

    hybrid_roll[window_size:] = ekf_roll[window_size:] + corrections[:, 0]
    hybrid_pitch[window_size:] = ekf_pitch[window_size:] + corrections[:, 1]

    # Calculate RMSE
    # For fair comparison, evaluate only on samples where hybrid has predictions
    eval_idx = slice(window_size, None)

    rmse_ekf_roll = np.sqrt(np.mean((ekf_roll[eval_idx] - truth_roll[eval_idx])**2))
    rmse_ekf_pitch = np.sqrt(np.mean((ekf_pitch[eval_idx] - truth_pitch[eval_idx])**2))

    rmse_hybrid_roll = np.sqrt(np.mean((hybrid_roll[eval_idx] - truth_roll[eval_idx])**2))
    rmse_hybrid_pitch = np.sqrt(np.mean((hybrid_pitch[eval_idx] - truth_pitch[eval_idx])**2))

    print("=== Performance Comparison ===")
    print(f"\nRoll Angle:")
    print(f"  EKF:           {rmse_ekf_roll:.4f}°")
    print(f"  Hybrid (EKF+LSTM): {rmse_hybrid_roll:.4f}°")
    print(f"  Improvement:   {((rmse_ekf_roll - rmse_hybrid_roll) / rmse_ekf_roll * 100):.1f}%")

    print(f"\nPitch Angle:")
    print(f"  EKF:           {rmse_ekf_pitch:.4f}°")
    print(f"  Hybrid (EKF+LSTM): {rmse_hybrid_pitch:.4f}°")
    print(f"  Improvement:   {((rmse_ekf_pitch - rmse_hybrid_pitch) / rmse_ekf_pitch * 100):.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_indices = np.arange(len(truth_roll))

    # Roll comparison
    axes[0, 0].plot(time_indices, truth_roll, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    axes[0, 0].plot(time_indices, ekf_roll, 'b--', linewidth=1.5, label=f'EKF (RMSE: {rmse_ekf_roll:.3f}°)', alpha=0.8)
    axes[0, 0].plot(time_indices, hybrid_roll, 'r:', linewidth=2, label=f'Hybrid (RMSE: {rmse_hybrid_roll:.3f}°)', alpha=0.8)
    axes[0, 0].set_ylabel('Roll Angle (deg)', fontsize=11)
    axes[0, 0].set_title('Roll Angle: EKF vs Hybrid EKF+LSTM', fontsize=12)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # Pitch comparison
    axes[0, 1].plot(time_indices, truth_pitch, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    axes[0, 1].plot(time_indices, ekf_pitch, 'b--', linewidth=1.5, label=f'EKF (RMSE: {rmse_ekf_pitch:.3f}°)', alpha=0.8)
    axes[0, 1].plot(time_indices, hybrid_pitch, 'r:', linewidth=2, label=f'Hybrid (RMSE: {rmse_hybrid_pitch:.3f}°)', alpha=0.8)
    axes[0, 1].set_ylabel('Pitch Angle (deg)', fontsize=11)
    axes[0, 1].set_title('Pitch Angle: EKF vs Hybrid EKF+LSTM', fontsize=12)
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Roll error comparison
    error_ekf_roll = np.abs(ekf_roll - truth_roll)
    error_hybrid_roll = np.abs(hybrid_roll - truth_roll)
    axes[1, 0].plot(time_indices, error_ekf_roll, 'b-', linewidth=1, label='EKF Error', alpha=0.7)
    axes[1, 0].plot(time_indices, error_hybrid_roll, 'r-', linewidth=1, label='Hybrid Error', alpha=0.7)
    axes[1, 0].set_ylabel('Absolute Error (deg)', fontsize=11)
    axes[1, 0].set_xlabel('Sample Index', fontsize=11)
    axes[1, 0].set_title('Roll Absolute Error', fontsize=12)
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    # Pitch error comparison
    error_ekf_pitch = np.abs(ekf_pitch - truth_pitch)
    error_hybrid_pitch = np.abs(hybrid_pitch - truth_pitch)
    axes[1, 1].plot(time_indices, error_ekf_pitch, 'b-', linewidth=1, label='EKF Error', alpha=0.7)
    axes[1, 1].plot(time_indices, error_hybrid_pitch, 'r-', linewidth=1, label='Hybrid Error', alpha=0.7)
    axes[1, 1].set_ylabel('Absolute Error (deg)', fontsize=11)
    axes[1, 1].set_xlabel('Sample Index', fontsize=11)
    axes[1, 1].set_title('Pitch Absolute Error', fontsize=12)
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Results/Figures/Hybrid_EKF_LSTM_Comparison.png', dpi=150)
    print("\nComparison plot saved: Hybrid_EKF_LSTM_Comparison.png")

    # Save hybrid results
    np.savetxt('Results/Results/HybridRoll.txt', hybrid_roll, fmt='%.6f')
    np.savetxt('Results/Results/HybridPitch.txt', hybrid_pitch, fmt='%.6f')
    print("Hybrid predictions saved!")

if __name__ == "__main__":
    evaluate_hybrid_approach()
```

---

### Step 5: Analysis and Interpretation (1 hour)

#### Questions to answer in thesis:

1. **When does LSTM correction help most?**
   - Analyze error reduction vs gyro magnitude
   - Check if corrections are larger during high dynamics
   - Identify systematic EKF biases the LSTM learned

2. **What patterns did the LSTM learn?**
   - Visualize correction magnitude over time
   - Correlate with sensor characteristics
   - Check if corrections are consistent or noisy

3. **Overfitting check:**
   - Compare train/val/test RMSE
   - If test >> val, model overfit
   - Discuss data augmentation needs

4. **Failure modes:**
   - Identify samples where hybrid is worse than EKF
   - Discuss when physics-based model should be trusted over ML

---

### Success Metrics

**Optimistic scenario (60% probability):**
- Roll improvement: 5-15% (0.298° → 0.25-0.28°)
- Pitch improvement: 10-20% (0.720° → 0.58-0.65°)
- **Result:** Publishable, strong thesis contribution

**Neutral scenario (30% probability):**
- Similar RMSE to pure EKF (within 5%)
- **Result:** Still valuable - demonstrates hybrid approach viability
- Discussion: "With more diverse data, approach shows promise"

**Pessimistic scenario (10% probability):**
- Worse than EKF (overfitting)
- **Result:** Honest negative result is still publishable
- Discussion: "Limited data insufficient for ML, classical methods superior"

---

## Final Recommendations

### What You Should Do

**Tier 1 (Essential) - 3 days:**
1. ✅ Fix Mahony pitch tuning (1 day)
2. ✅ Implement Simple LSTM in Python (2 days)

**After Tier 1:**
- Review results with professor
- Decide if Tier 2 is worth pursuing

**Tier 2 (If time permits) - 5 days:**
3. ✅ Hybrid EKF+LSTM (4-5 days)

### What You Should NOT Do

❌ C++ manual LSTM implementation
❌ Attention mechanisms (insufficient data)
❌ ONNX Runtime integration (overkill)
❌ Collect more flight data (out of scope for MSc)

### Thesis Structure Suggestion

```
Chapter 5: Classical Filter Comparison
  5.1 Complementary Filter
  5.2 Mahony Filter
    5.2.1 Parameter Sensitivity Analysis (kp tuning for roll+pitch)
  5.3 Extended Kalman Filter
  5.4 Performance Comparison

Chapter 6: Exploratory Machine Learning Analysis
  6.1 Motivation for Deep Learning Approaches
  6.2 LSTM Implementation
    6.2.1 Architecture
    6.2.2 Training Strategy
    6.2.3 Results and Limitations
  6.3 Hybrid EKF+LSTM Approach (if implemented)
    6.3.1 Physics-Informed ML Rationale
    6.3.2 Implementation
    6.3.3 Comparative Analysis
  6.4 Discussion
    6.4.1 Data Requirements for Effective ML
    6.4.2 When to Use Classical vs ML Methods

Chapter 7: Future Work
  7.1 Data Collection for Deep Learning
  7.2 Attention Mechanisms
  7.3 Multi-Sensor Fusion (IMU + GPS + Magnetometer)
```

### Final Decision Matrix

**If you have LIMITED time (< 1 week):**
→ Do Tier 1 + "Future Work" section
→ Focus on polishing existing filter implementations
→ Strong classical baseline is better than rushed ML

**If you have MODERATE time (1-2 weeks):**
→ Do Tier 1 + Simple LSTM
→ Demonstrates breadth
→ Honest about limitations

**If you have AMPLE time (2+ weeks) + professor approval:**
→ Do Tier 1 + Tier 2 (Hybrid EKF+LSTM)
→ Best chance of novel contribution
→ Publishable regardless of outcome

---

## Conclusion

The **Hybrid EKF+LSTM approach** offers the best balance of:
- Feasibility (works with limited data)
- Novelty (physics-informed ML)
- Success probability (60% chance of measurable improvement)
- Thesis impact (publishable regardless of outcome)

However, **classical filter optimization** (fixing Mahony pitch) should be done FIRST as it's a guaranteed quick win.

**Bottom line:** RNN is viable for your thesis, but approach it strategically:
1. Start with low-risk classical improvements
2. Add simple LSTM as "exploratory analysis"
3. Only pursue advanced hybrid approach if time permits

The key is honest discussion of limitations and data requirements - this demonstrates research maturity more than achieving state-of-the-art results with insufficient data.

---

**Document Version:** 1.0
**Last Updated:** October 29, 2025
**Status:** Ready for professor review and decision
