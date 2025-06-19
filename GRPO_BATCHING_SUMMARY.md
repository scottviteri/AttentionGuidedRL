# GRPO Batching and Variable Renaming Summary

## Overview
Successfully implemented variable renaming for clarity and GRPO-style batching using a simple stream repeat operator - exactly as suggested by the user.

## 1. Variable Renaming

### **Clarity Improvements**
- **`baseline_model` → `old_model`** (pi_old)
  - The periodically updated model used for KL computation
  - Updated every `BASELINE_UPDATE_FREQUENCY` episodes (currently 10)
  - Used for computing probability ratios in PPO: `π_θ(a|s) / π_old(a|s)`

- **`base_model` → `ref_model`** (pi_ref)  
  - The reference model without LoRA
  - Used for reward computation and comparison baselines
  - Remains fixed throughout training

### **Updated Function Signatures**
```python
# Before
compute_policy_loss(trajectory, adapter_model, baseline_model, ...)
train_step(trajectory, adapter_model, base_model, baseline_model, ...)
compute_trajectory_rewards(trajectory, adapter_model, base_model, ...)

# After  
compute_policy_loss(trajectory, adapter_model, old_model, ...)
train_step(trajectory, adapter_model, ref_model, old_model, ...)
compute_trajectory_rewards(trajectory, adapter_model, ref_model, ...)
```

## 2. GRPO Batching Implementation (Simplified!)

### **The Elegant Solution**
Instead of creating a complex `generate_grpo_trajectory` function, we simply use a stream operator:

```python
def repeat_n_times(n: int, stream: Iterator) -> Iterator:
    """Repeat each item from the stream n times."""
    for item in stream:
        for _ in range(n):
            yield item
```

### **How It Works**

#### **Standard Batching (without GRPO)**
```
Stream: [A, B, C, D, E, F, ...]
Batch (size=4): [A, B, C, D]
```

#### **GRPO Batching (with repeat operator)**
```
Stream: [A, B, C, D, E, F, ...]
Repeat(4): [A, A, A, A, B, B, B, B, C, C, C, C, ...]
Batch (size=4): [A, A, A, A]
```

### **Implementation**
```python
if use_grpo_batching:
    # Generate single items
    base_iterator = iter_key_value_pairs_unified_with_tokenizer(
        dataset_name=args.dataset,
        batch_size=1,  # Single items
        tokenizer=tokenizer,
        embedding_fn=embedding_fn
    )
    # Repeat each item batch_size times
    kv_pair_generator = repeat_n_times(args.batch_size, base_iterator)
else:
    # Standard batching
    kv_pair_generator = iter_key_value_pairs_unified_with_tokenizer(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        embedding_fn=embedding_fn
    )
```

### **Benefits of This Approach**

1. **Simplicity**: No need for a separate `generate_grpo_trajectory` function
2. **Reusability**: Uses the existing `generate_trajectory` function unchanged
3. **Composability**: Stream operators can be easily composed
4. **Clarity**: The repeat operation is explicit and easy to understand
5. **Flexibility**: Easy to add other stream transformations if needed

### **Command Line Usage**

```bash
# Standard batching (different items in each batch)
python -m src.main --batch-size 4

# GRPO batching (same item repeated in batch)
python -m src.main --batch-size 4 --grpo-batching
```

## 3. Architecture Comparison

### **Previous Complex Approach**
- Separate `generate_grpo_trajectory` function (170+ lines)
- Different trajectory generation logic
- Complex tensor reshaping for single trajectory
- Hard to maintain two different code paths

### **New Simple Approach**
- Just a repeat operator (5 lines)
- Reuses existing `generate_trajectory`
- Data transformation happens at the stream level
- Single code path with configurable behavior

## 4. Mathematical Equivalence

Both approaches achieve the same GRPO objective:
- Each batch contains multiple copies of the same data point
- Reduces variance in gradient estimates
- Follows established GRPO methodology

The difference is purely in implementation elegance.

## 5. Testing and Validation

✅ **Tests Pass**:
- Repeat operator correctly duplicates items
- `generate_grpo_trajectory` successfully removed
- Standard `generate_trajectory` handles both modes
- Command line flag `--grpo-batching` works

## 6. Summary

The user's suggestion to use a simple repeat operator was brilliant. It achieves the same GRPO batching effect with:
- **~170 fewer lines of code**
- **No specialized trajectory function**
- **Clear, composable stream operations**
- **Easier to understand and maintain**

This is a perfect example of how functional programming concepts (stream transformations) can dramatically simplify complex implementations.

## 7. Configuration Updates

### **Update Frequency**
- `BASELINE_UPDATE_FREQUENCY = 10` (reduced from 50)
- More frequent updates since `old_model` no longer affects reward computation

### **Compatibility**
- Maintains backward compatibility with existing training pipeline
- All plotting and logging functions work with renamed variables
- PPO clipping and advantage computation unchanged

## 8. Model Usage Clarification

### **Current Model Roles**
```python
# adapter_model: Trainable LoRA adapter (π_θ)
# ref_model: Fixed reference model (π_ref) - for rewards
# old_model: Periodically updated model (π_old) - for KL computation
```

### **PPO Computation**
```python
# Probability ratio for PPO clipping
ratio = π_θ(action|context) / π_old(action|context)

# Reward computation  
reward = adapter_log_prob - ref_log_prob  # (if SUBTRACT_BASE_MODEL_LOGPROBS=True)
# OR
reward = adapter_log_prob  # (if SUBTRACT_BASE_MODEL_LOGPROBS=False)
```

## 9. Usage

### **Environment Variables**
```bash
export BASELINE_UPDATE_FREQUENCY=10  # Update old_model frequency
export PPO_CLIP_EPSILON=0.2         # PPO clipping parameter
```

### **Training Command**
```bash
python -m src.main --batch-size 4 --verbose
```

The batch size now represents the size of the key pool rather than the number of parallel trajectories. 