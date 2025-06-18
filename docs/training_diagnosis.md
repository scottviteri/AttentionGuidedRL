# Training Diagnosis Summary

## Issues Found

### 1. Missing similarity_scores During Filtering ✓ FIXED
**Problem**: In `filter_trajectories_grpo`, the `similarity_scores` and `selected_idx` attributes were not being copied to the filtered trajectory, causing "Skipping step - no similarity scores" messages.

**Solution**: Added code to copy these attributes after creating the filtered QKVStep.

### 2. dtype Mismatch in compute_similarity ✓ FIXED  
**Problem**: einsum operation failed with "expected scalar type Float but found BFloat16" error.

**Solution**: Added dtype conversion to float32 in compute_similarity function.

### 3. Dimension Mismatch in KL Divergence ❌ NEEDS FIX
**Problem**: The KL divergence computation fails because:
- `similarity_scores` has shape [batch, remaining_keys] where remaining_keys decreases (15, 14, 13...)
- `all_key_embeddings` has shape [batch, total_keys, hidden] where total_keys is always 15
- When recomputing similarities with previous model, dimensions don't match

**Root Cause**: The trajectory generation removes selected keys from the pool at each step, but the KL computation assumes all keys are always available.

### 4. High Filtering Rate Leading to Zero Loss
**Problem**: Many trajectories are filtered out (e.g., "Filtered: 9/16"), and those that remain often have all negative advantages, leading to zero policy loss.

**Possible Causes**:
- Rewards are very similar across trajectories
- GRPO baseline is too aggressive
- Initial policy is too random

## Recommended Fixes

### Fix for Dimension Mismatch
Instead of storing `all_key_embeddings` at the trajectory level, we should store the available key embeddings at each step in the QKVStep. This way, the KL computation can use the correct subset of keys.

### Fix for High Filtering Rate
1. Reduce initial filtering to allow more gradients to flow
2. Add a small epsilon to advantages to prevent all zeros
3. Consider using a softer baseline or warmup period 