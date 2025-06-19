# Periodic Spike Analysis

## Problem Description
The training metrics show periodic spikes with period 10 in:
- Trajectory log probabilities
- Average reward
- Model log probabilities
- Reward variance within trajectory

These spikes appear at episodes 10, 20, 30, 40, etc.

## Root Cause

The spikes are caused by the `old_model` update mechanism that happens every 10 episodes (`BASELINE_UPDATE_FREQUENCY = 10`).

### What Happens at Each Spike

1. **Episode 10, 20, 30, etc.**: The `old_model` is updated to be a copy of the current `adapter_model`
2. **Key Embeddings Change**: The data iterator is recreated with the new `old_model` for computing key embeddings
3. **Different Representations**: The new key embeddings are now computed with a model that has been trained for 10 episodes, making them significantly different from the previous embeddings

### Why This Causes Spikes

1. **Sudden Context Shift**: The key embeddings that the model sees suddenly change because they're now computed with the updated model
2. **Representation Mismatch**: The adapter model has been trained on key embeddings from the old version, but suddenly sees embeddings from the new version
3. **Reward Distribution Change**: The reward distribution shifts because the quality of matches between queries and keys changes with the new embeddings

## The "Offset" Pattern

The pattern looks periodic with an offset because:
- At update episodes (10, 20, 30...): Sudden spike due to representation change
- Episodes 11-19, 21-29, etc.: Gradual adaptation as the model learns to work with the new embeddings
- Just before next update: Model has adapted, metrics stabilize
- Then the cycle repeats

## Evidence from Logs

Looking at the rewards in the training logs:
- Episode 9: -2.8977
- Episode 10: **-2.1597** (spike - better reward after update)
- Episode 19: -2.9673  
- Episode 20: **-2.1162** (spike - better reward after update)
- Episode 29: -2.9961
- Episode 30: **-2.1953** (spike - better reward after update)

## Why This Design Exists

This mechanism serves important purposes:
1. **PPO KL Constraint**: The KL divergence is computed against `old_model`, which should be a recent but not identical version
2. **Key Embedding Consistency**: Keys need to be embedded with a consistent model during each phase
3. **Stability vs Progress**: Updating too frequently would be unstable, too infrequently would make KL constraint meaningless

## Potential Solutions

### 1. Smoother Transition (Recommended)
Instead of hard-switching the key embeddings, use a mixture:
```python
alpha = min(1.0, (episode % BASELINE_UPDATE_FREQUENCY) / BASELINE_UPDATE_FREQUENCY)
mixed_embeddings = alpha * new_embeddings + (1 - alpha) * old_embeddings
```

### 2. More Frequent Updates
Reduce `BASELINE_UPDATE_FREQUENCY` to 5 or even 2 to make transitions less dramatic.

### 3. Separate Key Embedding Model
Use a separate, slowly-updated model for key embeddings that updates via exponential moving average:
```python
key_model_params = 0.99 * key_model_params + 0.01 * adapter_model_params
```

### 4. Pre-compute All Embeddings
Compute all key embeddings once at the start with the initial model and never update them.

## Trade-offs

- **Current approach**: Clear phases, easy to debug, but causes spikes
- **Smooth transition**: More stable training, but more complex
- **Frequent updates**: Smaller spikes, but more computational cost
- **Separate model**: Most stable, but adds complexity
- **Fixed embeddings**: No spikes, but loses adaptation benefits 