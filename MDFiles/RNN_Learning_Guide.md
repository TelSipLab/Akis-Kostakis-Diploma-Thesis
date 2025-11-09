# RNN Learning Guide - From CNNs to RNNs

**Target Audience:** Someone with CNN experience learning RNNs
**Date:** November 9, 2025
**Purpose:** Bridge CNN knowledge to understand RNNs for IMU time series prediction

---

## Table of Contents

1. [CNNs vs RNNs: High-Level Comparison](#cnns-vs-rnns-high-level-comparison)
2. [Why RNNs for Time Series](#why-rnns-for-time-series)
3. [Vanilla RNN: The Basics](#vanilla-rnn-the-basics)
4. [The Problem: Vanishing Gradients](#the-problem-vanishing-gradients)
5. [LSTM: The Solution](#lstm-the-solution)
6. [GRU: The Simpler Alternative](#gru-the-simpler-alternative)
7. [Training RNNs: Backpropagation Through Time](#training-rnns-backpropagation-through-time)
8. [Practical Implementation with libtorch](#practical-implementation-with-libtorch)
9. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
10. [RNNs for IMU Data: Our Use Case](#rnns-for-imu-data-our-use-case)

---

## CNNs vs RNNs: High-Level Comparison

### What You Know: Convolutional Neural Networks

**CNNs are for spatial patterns:**
```
Image (H × W × C) → Conv Layer → Feature Maps → ... → Classification
```

**Key properties:**
- **Parameter sharing** across spatial locations (same kernel everywhere)
- **Translation invariance** (cat in top-left = cat in bottom-right)
- **Fixed input size** (or with padding/pooling tricks)
- **No memory** between images (each image processed independently)

**Example:** Image classification
```
Input: [224×224×3] image
Conv1: 64 filters of 3×3 → [224×224×64]
Pool1: MaxPool 2×2 → [112×112×64]
Conv2: 128 filters of 3×3 → [112×112×128]
...
FC: Flatten → Classification
```

### What's New: Recurrent Neural Networks

**RNNs are for temporal patterns:**
```
Sequence (T timesteps) → RNN Layer → Hidden States → ... → Prediction
```

**Key properties:**
- **Parameter sharing** across time (same weights for all timesteps)
- **Temporal dependence** (order matters!)
- **Variable input length** (can handle sequences of any length)
- **Memory/Hidden state** (remembers previous timesteps)

**Example:** Sentence sentiment analysis
```
Input: "This movie is great" (4 words)
→ RNN processes word-by-word
→ Final hidden state → Classification (positive/negative)
```

---

## Why RNNs for Time Series

### Your Use Case: IMU Time Series Prediction

**Data structure:**
```
Time:  t=0    t=1    t=2    ...    t=k    → predict t=k+1
       ↓      ↓      ↓             ↓
State: [r,p,  [r,p,  [r,p,         [r,p,        [r,p,
        a,g]   a,g]   a,g]          a,g]         a,g]

Where: r = roll, p = pitch, a = accel (3D), g = gyro (3D)
```

**Why not CNNs?**
- CNNs treat all positions equally (good for images)
- Time series has directionality: past → future
- Order matters: [t=0, t=1, t=2] ≠ [t=2, t=1, t=0]
- Need to remember long-term dependencies

**Why RNNs?**
- Process sequence step-by-step (left to right)
- Maintain hidden state (memory of previous timesteps)
- Learn temporal patterns (e.g., acceleration → velocity → position)
- Can predict next state given history

---

## Vanilla RNN: The Basics

### Architecture

Think of an RNN as a **feedforward network that processes one timestep at a time, maintaining a hidden state**.

**Single timestep:**
```
Input at time t:  x_t ∈ ℝⁿ
Hidden state:     h_t ∈ ℝʰ  (memory from previous timesteps)
Output at time t: y_t ∈ ℝᵐ

Update equations:
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

**Unrolled over time:**
```
   x_0        x_1        x_2        x_3
    │          │          │          │
    ↓          ↓          ↓          ↓
┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│  RNN  │→│  RNN  │→│  RNN  │→│  RNN  │
└───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
    │          │          │          │
    ↓          ↓          ↓          ↓
   y_0        y_1        y_2        y_3
```

**Key insight:** Same RNN cell, applied repeatedly. The arrow → carries hidden state h_t.

### Comparison to CNN

| Aspect | CNN | RNN |
|--------|-----|-----|
| **Kernel/Cell** | Convolutional kernel (e.g., 3×3) | RNN cell with hidden state |
| **Weight sharing** | Across spatial locations | Across time steps |
| **Connection** | Local (receptive field) | Sequential (previous timestep) |
| **Output** | Feature map | Sequence of hidden states |

### Example: Simple Sequence

**Task:** Predict next number in sequence [1, 2, 3, 4, ?]

```python
# Pseudocode
h_0 = zeros(hidden_size)  # Initial hidden state

# Process sequence
x_0 = 1 → h_1 = RNN(x_0, h_0) → y_1 = predict(h_1) = 2
x_1 = 2 → h_2 = RNN(x_1, h_1) → y_2 = predict(h_2) = 3
x_2 = 3 → h_3 = RNN(x_2, h_2) → y_3 = predict(h_3) = 4
x_3 = 4 → h_4 = RNN(x_3, h_3) → y_4 = predict(h_4) = 5 ✓
```

**Key:** Each prediction uses information from ALL previous inputs (through hidden state).

---

## The Problem: Vanishing Gradients

### Why Vanilla RNN Fails for Long Sequences

Remember backpropagation? Gradients flow backward through the network.

**In CNNs:**
- Gradient flows through a few layers (e.g., 10-100 layers)
- Can use skip connections (ResNet) to help

**In RNNs:**
- Gradient flows through many timesteps (e.g., 100+ timesteps)
- Each timestep multiplies by W_hh
- Gradient = (W_hh)^T × (W_hh)^T × ... × (W_hh)^T (100 times!)

**Problem:**
- If eigenvalues of W_hh < 1: gradients **vanish** (go to 0)
- If eigenvalues of W_hh > 1: gradients **explode** (go to ∞)

**Result:** Vanilla RNN can't learn long-term dependencies (> 10 timesteps).

**Example:**
```
"The cat, which we found in the garden last week, ... [100 words] ... was gray."

RNN needs to remember "cat" (singular) → "was" (not "were")
But gradient from "was" can't reach back to "cat" through 100 timesteps!
```

---

## LSTM: The Solution

### Long Short-Term Memory

**Key idea:** Instead of simple hidden state, use a **memory cell** with **gates** to control information flow.

### Gates Intuition (Think of Water Flow)

Imagine information flowing through pipes:

1. **Forget Gate** (f_t): "Should I forget old memory?"
   - Like a valve that lets you drain the old water
   - f_t = 0: Forget everything
   - f_t = 1: Keep everything

2. **Input Gate** (i_t): "Should I add new information?"
   - Like a valve that lets new water in
   - i_t = 0: Ignore new input
   - i_t = 1: Fully incorporate new input

3. **Output Gate** (o_t): "What should I output?"
   - Like a tap that controls output flow
   - o_t = 0: Don't output anything
   - o_t = 1: Output full cell state

### LSTM Equations

**State vectors:**
- h_t: Hidden state (what we output to next layer)
- c_t: Cell state (long-term memory, protected by gates)

**Gate equations:**
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    # Input gate
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    # Output gate

c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c) # Candidate cell state

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       # Update cell state
h_t = o_t ⊙ tanh(c_t)                  # Update hidden state

Where:
σ = sigmoid function (0 to 1, perfect for gates)
⊙ = element-wise multiplication
tanh = activation function (-1 to 1)
```

### Visual Diagram

```
              ┌─────────────────────────┐
  c_{t-1} ────┤  Forget gate (f_t)     │
              └──────────┬──────────────┘
                         │ × f_t (element-wise)
                         ↓
              ┌──────────┴──────────────┐
              │  Cell State c_t         │←── i_t × c̃_t (new info)
              └──────────┬──────────────┘
                         │ tanh
                         ↓
              ┌──────────┴──────────────┐
              │  × o_t (output gate)    │
              └──────────┬──────────────┘
                         ↓
                        h_t
```

### Why LSTM Works

**Long-term memory preservation:**
- Cell state c_t can flow through many timesteps unchanged
- Forget gate protects from vanishing gradients
- If f_t ≈ 1, c_t ≈ c_{t-1} (memory preserved)

**Gradient flow:**
- Gradients can flow directly through cell state
- No repeated matrix multiplications!
- Can learn dependencies over 100+ timesteps

**Comparison to highway/ResNet:**
- Like skip connections in CNNs
- Direct path for gradient flow
- But with learned gates (more flexible)

---

## GRU: The Simpler Alternative

### Gated Recurrent Unit

**Key idea:** Simplify LSTM by combining gates.

**Changes from LSTM:**
- No separate cell state (only h_t)
- Only 2 gates instead of 3
- Fewer parameters (faster training)

### GRU Equations

```
z_t = σ(W_z × [h_{t-1}, x_t] + b_z)    # Update gate (like input gate)
r_t = σ(W_r × [h_{t-1}, x_t] + b_r)    # Reset gate (like forget gate)

h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t] + b_h)  # Candidate hidden state

h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         # Update hidden state
```

**Intuition:**
- Update gate z_t: "How much to update?" (like input + forget combined)
- Reset gate r_t: "How much to forget previous hidden state?"
- Only one state h_t (simpler than LSTM's h_t and c_t)

### LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| **Parameters** | More (3 gates + cell) | Fewer (2 gates) |
| **Training speed** | Slower | Faster |
| **Expressiveness** | Higher | Slightly lower |
| **Long dependencies** | Excellent | Very good |
| **When to use** | Complex tasks, lots of data | Simpler tasks, limited data |

**For your use case (IMU data):**
- Start with **LSTM** (more established, better performance)
- Try **GRU** if training is slow or overfitting occurs

---

## Training RNNs: Backpropagation Through Time

### Concept

**In CNNs:** Backprop through layers (spatial)
```
Loss → ∂L/∂W_layer3 → ∂L/∂W_layer2 → ∂L/∂W_layer1
```

**In RNNs:** Backprop through time (temporal)
```
Loss at t=T → ∂L/∂W_t=T → ∂L/∂W_t=T-1 → ... → ∂L/∂W_t=0
```

### Unrolled Network

**Forward pass:**
```
x_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3 → Loss
x_1 ──↗       x_2 ──↗       x_3 ──↗
```

**Backward pass:**
```
       ∂L/∂h_3 ← ∂L/∂h_2 ← ∂L/∂h_1 ← ∂L/∂h_0
         ↓         ↓         ↓
       ∂L/∂W    ∂L/∂W    ∂L/∂W     (accumulate gradients)
```

### Truncated BPTT

**Problem:** Long sequences require too much memory.

**Solution:** Only backprop through k timesteps (e.g., k=20).

```
Full sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...] (length 1000)

Truncated BPTT (k=4):
- Forward: Process all 1000 timesteps
- Backward: Only backprop through last 4 timesteps at a time
```

**Trade-off:**
- Less memory usage ✓
- Faster training ✓
- Can't learn dependencies longer than k ✗

**For your use case:**
- Sequence length: 20 timesteps (0.4 seconds)
- This is fine for truncated BPTT
- Most dynamics happen within this window

---

## Practical Implementation with libtorch

### 1. Basic LSTM Cell

```cpp
#include <torch/torch.h>

// LSTM layer definition
class LSTMNet : public torch::nn::Module {
public:
    LSTMNet(int input_size, int hidden_size, int num_layers, int output_size) {
        // LSTM layer
        lstm = register_module("lstm",
            torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .batch_first(true)));  // Input shape: (batch, seq, feature)

        // Fully connected output layer
        fc = register_module("fc", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x shape: (batch_size, seq_length, input_size)

        // LSTM forward pass
        auto lstm_out = lstm->forward(x);
        auto output = std::get<0>(lstm_out);  // Get output, ignore hidden states

        // output shape: (batch_size, seq_length, hidden_size)
        // Take last timestep
        auto last_output = output.select(1, -1);  // (batch_size, hidden_size)

        // Fully connected layer
        auto prediction = fc->forward(last_output);  // (batch_size, output_size)

        return prediction;
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};
```

### 2. Training Loop

```cpp
// Create model
auto model = std::make_shared<LSTMNet>(
    8,    // input_size: 8D state vector
    64,   // hidden_size
    2,    // num_layers: 2 LSTM layers
    5     // output_size: 5D prediction
);

// Loss and optimizer
torch::nn::MSELoss criterion;
torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : train_loader) {
        // Get data
        auto inputs = batch.data;   // (batch, seq_len, 8)
        auto targets = batch.target; // (batch, 5)

        // Forward pass
        auto predictions = model->forward(inputs);
        auto loss = criterion(predictions, targets);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Print loss
        if (batch_idx % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
}
```

### 3. Making Predictions

```cpp
// Load trained model
torch::load(model, "best_model.pt");
model->eval();  // Set to evaluation mode

torch::NoGradGuard no_grad;  // Disable gradient computation

// Prepare input sequence
torch::Tensor input_seq = torch::zeros({1, 20, 8});  // (batch=1, seq=20, features=8)

// Fill with your data
for (int t = 0; t < 20; ++t) {
    input_seq[0][t][0] = roll[t];
    input_seq[0][t][1] = pitch[t];
    // ... fill rest of features
}

// Predict
auto prediction = model->forward(input_seq);  // (1, 5)

// Extract predicted values
float pred_roll = prediction[0][0].item<float>();
float pred_pitch = prediction[0][1].item<float>();
// ...
```

---

## Common Pitfalls and Solutions

### 1. Exploding Gradients

**Symptom:** Loss becomes NaN during training.

**Solution:** Gradient clipping
```cpp
// After loss.backward(), before optimizer.step()
torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);  // Clip to max norm 1.0
```

### 2. Overfitting

**Symptom:** Training loss decreases, validation loss increases.

**Solutions:**
- **Dropout:** Already built into LSTM
  ```cpp
  torch::nn::LSTMOptions(input_size, hidden_size)
      .num_layers(2)
      .dropout(0.2)  // 20% dropout between layers
  ```
- **Early stopping:** Stop training when validation loss stops improving
- **Smaller model:** Reduce hidden_size or num_layers

### 3. Vanishing Gradients (Even with LSTM!)

**Symptom:** Model doesn't learn long-term dependencies.

**Solutions:**
- Use **LSTM or GRU** (not vanilla RNN)
- **Gradient clipping** (prevents explosion, helps vanishing)
- **Learning rate tuning:** Try different values (0.01, 0.001, 0.0001)
- **Batch normalization:** Normalize between layers

### 4. Slow Convergence

**Symptom:** Loss decreases very slowly.

**Solutions:**
- **Better initialization:** Xavier/He initialization (libtorch does this by default)
- **Learning rate schedule:** Reduce learning rate during training
  ```cpp
  auto scheduler = torch::optim::StepLR(optimizer,
      /*step_size=*/10, /*gamma=*/0.1);  // Reduce LR by 10x every 10 epochs
  ```
- **Batch normalization:** Stabilizes training
- **Pre-training:** Initialize with simpler task first

### 5. Wrong Input Shape

**Common error:** Tensor shape mismatch

**Solution:** Always print shapes during debugging
```cpp
std::cout << "Input shape: " << x.sizes() << std::endl;
// Expected: [batch_size, seq_length, input_size]
// Example: [32, 20, 8]
```

**Checklist:**
- ✅ batch_first=true in LSTM options
- ✅ Input shape: (batch, seq, features)
- ✅ Target shape: (batch, output_size)

---

## RNNs for IMU Data: Our Use Case

### Why RNN is Perfect for IMU

**IMU data characteristics:**
1. **Temporal structure:** Measurements are sequential
2. **High sampling rate:** 50 Hz → close correlation between adjacent samples
3. **Dynamics:** Angular velocity integrates to angle (temporal relationship)
4. **Noise:** RNN can learn to filter noise patterns

**What RNN will learn:**
- **Short-term dynamics:** ω_t → θ_{t+1} (angular velocity → angle)
- **Filter residuals:** Patterns in filter errors
- **Sensor noise patterns:** Systematic errors in accelerometer/gyro
- **Motion patterns:** Common movement sequences

### Architecture Design for IMU

```
Input Sequence (20 timesteps × 8 features):
  [roll, pitch, ax, ay, az, ωx, ωy, ωz] at t=0
  [roll, pitch, ax, ay, az, ωx, ωy, ωz] at t=1
  ...
  [roll, pitch, ax, ay, az, ωx, ωy, ωz] at t=19

         ↓

    LSTM Layer 1 (64 units)
    - Learns basic patterns
    - Integrates velocity → position

         ↓

    LSTM Layer 2 (32 units)
    - Refines predictions
    - Learns filter corrections

         ↓

    Fully Connected (5 outputs)
    - Maps hidden state to prediction

         ↓

Output (next state at t=20):
  [roll, pitch, ωx, ωy, ωz]
```

### Sequence Length Choice

**Why 20 timesteps?**
```
20 samples × 0.02s/sample = 0.4 seconds
```

**Intuition:**
- Human reactions: ~0.2-0.4s
- IMU dynamics: Most changes within 0.5s
- Memory: 20 × 8 = 160 values (manageable)
- Long enough: Capture dynamics
- Short enough: Avoid overfitting

**Too short (< 10 timesteps):**
- Can't learn temporal patterns
- Might as well use feedforward network

**Too long (> 50 timesteps):**
- Overfitting on small dataset
- Slower training
- Diminishing returns

### Data Normalization

**Critical for RNN training!**

```cpp
// Compute statistics on TRAINING set only
Eigen::VectorXd mean = train_data.colwise().mean();
Eigen::VectorXd std = train_data.colwise().std();

// Normalize (z-score)
normalized = (data - mean) / std;

// Later, denormalize predictions
prediction_denorm = prediction * std + mean;
```

**Why normalize each feature separately?**
```
Roll:        ~[-0.5, 0.5] rad
Pitch:       ~[-0.5, 0.5] rad
Accel:       ~[-20, 20] m/s²    ← Much larger scale!
Gyro:        ~[-5, 5] rad/s

Without normalization: Network only learns from accel (largest gradients)
With normalization: All features contribute equally
```

---

## Example: Complete Mini Training Script

```cpp
#include <torch/torch.h>
#include <iostream>

// 1. Define model
struct IMU_LSTM : torch::nn::Module {
    IMU_LSTM() {
        lstm1 = register_module("lstm1",
            torch::nn::LSTM(torch::nn::LSTMOptions(8, 64).num_layers(1).batch_first(true)));
        lstm2 = register_module("lstm2",
            torch::nn::LSTM(torch::nn::LSTMOptions(64, 32).num_layers(1).batch_first(true)));
        fc = register_module("fc", torch::nn::Linear(32, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out1 = std::get<0>(lstm1->forward(x));
        auto out2 = std::get<0>(lstm2->forward(out1));
        auto last = out2.select(1, -1);  // Get last timestep
        return fc->forward(last);
    }

    torch::nn::LSTM lstm1{nullptr}, lstm2{nullptr};
    torch::nn::Linear fc{nullptr};
};

int main() {
    // 2. Create model
    auto model = std::make_shared<IMU_LSTM>();

    // 3. Create optimizer
    torch::optim::Adam optimizer(model->parameters(), 0.001);

    // 4. Create dummy data (replace with real data)
    int batch_size = 32;
    int seq_length = 20;
    int input_size = 8;
    int output_size = 5;

    auto train_x = torch::randn({batch_size, seq_length, input_size});
    auto train_y = torch::randn({batch_size, output_size});

    // 5. Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        auto predictions = model->forward(train_x);
        auto loss = torch::mse_loss(predictions, train_y);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }

    // 6. Save model
    torch::save(model, "imu_lstm.pt");

    return 0;
}
```

---

## Key Takeaways

### CNNs vs RNNs Summary

| Aspect | CNN | RNN |
|--------|-----|-----|
| **Input** | Images (spatial) | Sequences (temporal) |
| **Key operation** | Convolution | Recurrence |
| **Memory** | None | Hidden state |
| **Order matters?** | No (translation invariance) | Yes! (temporal order) |
| **Variable length?** | No (fixed size) | Yes |

### RNN Type Recommendations

1. **Start with LSTM** - Most robust, well-established
2. **Try GRU** - If LSTM is slow or overfitting
3. **Avoid vanilla RNN** - Vanishing gradient issues

### Critical Points for IMU

1. ✅ **Normalize data** - Essential for convergence
2. ✅ **Temporal split** - Train/val/test must be sequential, not random
3. ✅ **Sequence length** - 20 timesteps (0.4s) is good starting point
4. ✅ **Gradient clipping** - Prevent exploding gradients
5. ✅ **Validation monitoring** - Watch for overfitting on small dataset

---

## Next Steps for Your Implementation

1. **Understand theory** ✓ (you're here!)
2. **Install libtorch** - Get PyTorch C++ API working
3. **Test LSTM forward pass** - Verify shapes with dummy data
4. **Load IMU + filter data** - Prepare sequences
5. **Train simple model** - 1-layer LSTM as baseline
6. **Iterate** - Add complexity, tune hyperparameters
7. **Evaluate** - Compare with classical filters

---

## Additional Resources

### Tutorials
- **Official libtorch tutorial:** https://pytorch.org/cppdocs/
- **Understanding LSTM:** http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **RNN effectiveness:** http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### Papers
- **Original LSTM paper:** Hochreiter & Schmidhuber (1997)
- **GRU paper:** Cho et al. (2014)
- **Sequence to sequence:** Sutskever et al. (2014)

### Practice
- Start with toy problem (predict sine wave)
- Then move to IMU data
- Debug with small sequences first (seq_len = 5)

---

**Document Version:** 1.0
**Last Updated:** November 9, 2025
**Status:** Complete learning guide
