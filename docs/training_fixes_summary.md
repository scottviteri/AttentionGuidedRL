# Training Fixes Summary

## Issues Fixed ✓

### 1. Missing similarity_scores During Filtering
**Problem**: `filter_trajectories_grpo` wasn't copying `similarity_scores` and `selected_idx` attributes.
**Fix**: Added code to copy these attributes after creating filtered QKVStep.

### 2. dtype Mismatch in compute_similarity  
**Problem**: einsum operation failed with bfloat16 tensors.
**Fix**: Added conversion to float32 in compute_similarity function.

### 3. Gradient Flow Issues
**Problem**: Using Python floats (0.0) for loss accumulation broke gradient flow.
**Fix**: Changed to use torch tensors with requires_grad=True for policy_loss, kl_loss, and entropy_bonus.

### 4. Zero Gradient Edge Case
**Problem**: When all advantages are zero, no gradients flow.
**Fix**: Added small epsilon when all effective advantages are zero.

## Issues Temporarily Disabled

### 1. KL Divergence Dimension Mismatch
**Problem**: similarity_scores shape changes as keys are removed, but all_key_embeddings stays constant.
**Status**: Temporarily disabled with `if False` to allow training to proceed.
**TODO**: Store available key embeddings at each step instead of globally.

## Current Issues

### 1. Negative Loss Values
The total loss is negative (e.g., -0.0214), which suggests:
- Entropy term is too large relative to policy loss
- May need to reduce ENTROPY_COEF from 0.01

### 2. Adapter Weights Not Updating
Despite non-zero loss and gradients, adapter weights aren't changing:
- Learning rate might be too small (currently 0.0002)
- Gradients might be very small due to high filtering rate

### 3. High Trajectory Filtering Rate
Many trajectories are filtered (e.g., "Filtered: 10/16"):
- GRPO baseline might be too aggressive
- Initial rewards are very similar, making advantages small

## Recommended Next Steps

1. **Re-enable KL divergence** with proper dimension handling
2. **Reduce entropy coefficient** to prevent negative losses
3. **Increase learning rate** to ensure weight updates
4. **Consider warmup period** before applying GRPO filtering
5. **Add gradient norm logging** to debug small gradients

## Configuration Changes Made

- `USE_POSITIVE_ADVANTAGES_ONLY = False` (temporarily disabled to reduce filtering) 