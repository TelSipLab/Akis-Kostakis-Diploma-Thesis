# Future Work: RNN-Based Attitude Estimation

## Motivation

Current implementations (Complementary, Mahony, EKF) use classical sensor fusion techniques. Deep learning approaches, particularly Recurrent Neural Networks (RNNs), could potentially:

- Learn complex sensor error patterns from data
- Handle non-linear dynamics without explicit modeling
- Adapt to different motion regimes automatically
- Potentially outperform EKF in highly dynamic scenarios

## Proposed Approach

### Network Architecture Options

1. **LSTM (Long Short-Term Memory)**
   - Input: IMU window (gyro + accel measurements)
   - Output: Roll and pitch angles
   - Advantage: Learns temporal dependencies

2. **GRU (Gated Recurrent Unit)**
   - Simpler than LSTM, faster training
   - Similar performance potential

3. **Hybrid Approach**
   - Use EKF for baseline estimation
   - RNN learns residual corrections
   - Combines physics-based and data-driven methods

### Training Data Requirements

- Current dataset: 1500+ samples
- Need: Multiple motion scenarios (static, slow, fast, mixed)
- Split: 70% train, 15% validation, 15% test
- Ground truth: angles.csv provides labels

### Implementation Steps (Future)

1. Data preprocessing:
   - Normalize sensor inputs
   - Create sliding windows
   - Generate training sequences

2. Model design:
   - Define network architecture (PyTorch/TensorFlow)
   - Choose loss function (MSE for regression)
   - Set hyperparameters (hidden size, layers, learning rate)

3. Training:
   - Train on motion data
   - Monitor validation RMSE
   - Compare against EKF baseline (0.298°)

4. Evaluation:
   - Test on held-out data
   - Compare with classical filters
   - Analyze failure modes

### Expected Challenges

- Limited training data (single dataset)
- Generalization to new motion profiles
- Computational cost (vs real-time classical filters)
- Interpretability (black box vs physics-based)
- May not beat well-tuned EKF without more data

### Success Criteria

Would need to achieve **< 0.25° RMSE** to justify complexity over EKF.

## References to Add

- IONet: Learning to Cure the Curse of Drift in Inertial Odometry (2020)
- RIDI: Robust IMU Double Integration (2019)
- Deep learning for sensor fusion: A survey (2021)

---

**Status**: Deferred for future work. Current classical implementation (EKF) provides excellent baseline performance.

**Date**: 2025-10-28
