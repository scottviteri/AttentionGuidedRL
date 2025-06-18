# Summing vs Averaging Policy Gradients Across Trajectories

## Current Implementation: Averaging

```python
avg_policy_loss = policy_loss / count
```

## Alternative: Summing

```python
total_policy_loss = policy_loss  # No division
```

## Detailed Analysis

### 1. Length Invariance

**Averaging (Current)**
- ✓ **Length-invariant**: A trajectory with 5 good steps has the same loss scale as one with 10 good steps
- ✓ **Fair comparison**: Different length trajectories can be compared directly
- Example: Loss of 2.0 means "average badness per step" regardless of trajectory length

**Summing**
- ✗ **Length-dependent**: Longer trajectories accumulate more loss
- ✗ **Unfair bias**: Model learns to prefer shorter trajectories to minimize total loss
- Example: 5-step trajectory with loss 10 vs 10-step trajectory with loss 20

### 2. Gradient Magnitudes

**Averaging**
- Gradients scale with average advantage
- More stable training with consistent gradient magnitudes
- Learning rate doesn't need adjustment for different trajectory lengths

**Summing**
- Gradients scale with total accumulated advantage
- Variable gradient magnitudes based on trajectory length
- May need adaptive learning rates

### 3. Behavioral Incentives

**Averaging**
```python
# Incentive: Maximize average advantage per step
# Behavior: Consistent good performance throughout
Loss = -E[log π(a) × A⁺] / T
```

**Summing**
```python
# Incentive: Maximize total advantage
# Behavior: Longer trajectories with many good steps
Loss = -E[Σ log π(a) × A⁺]
```

### 4. Specific Scenarios

#### Variable-Length Trajectories
- **Averaging**: Essential for fair comparison
- **Summing**: Creates length bias

#### Early Termination
- **Averaging**: No penalty for shorter successful trajectories
- **Summing**: Encourages extending trajectories even if unnecessary

#### Fixed-Length Trajectories
- **Averaging**: Equivalent to summing with scaled learning rate
- **Summing**: Simpler, no normalization needed

### 5. With Positive Advantage Filtering

**Averaging + Clamping (Current)**
```python
effective_advantages = torch.clamp(advantages, min=0.0)
loss = -(log_probs * effective_advantages).mean()
```
- Focuses on quality: "What fraction of steps were good?"
- Robust to trajectory length variations

**Summing + Clamping**
```python
effective_advantages = torch.clamp(advantages, min=0.0)
loss = -(log_probs * effective_advantages).sum()
```
- Focuses on quantity: "How many good steps total?"
- Biased toward longer trajectories

### 6. Mathematical Properties

**Averaging**
- Loss ∈ [0, max_possible_loss_per_step]
- Bounded and predictable
- Natural for stochastic optimization

**Summing**
- Loss ∈ [0, max_possible_loss_per_step × T]
- Unbounded with trajectory length
- Requires careful normalization

## Recommendation

**Keep Averaging** for the following reasons:

1. **Robustness**: Works correctly with variable-length trajectories
2. **Stability**: Consistent gradient magnitudes improve training
3. **Interpretability**: Average loss per step is more meaningful
4. **No Length Bias**: Doesn't artificially favor short/long trajectories
5. **Standard Practice**: Most modern RL algorithms use averaging

## When to Consider Summing

1. **Fixed-length episodes**: When all trajectories have exactly the same length
2. **Total return focus**: When you explicitly want to maximize total return
3. **Specific reward structures**: Some domains may naturally align with summing

## Hybrid Approach

```python
# Weight by trajectory length but with diminishing returns
effective_count = count ** 0.5  # Square root weighting
weighted_loss = policy_loss / effective_count
```

This gives some preference to longer trajectories without full length dependence. 