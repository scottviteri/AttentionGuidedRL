# Clarification on `compute_policy_loss` Conditional Cases

## Current Understanding

### 1. When would tokenizer be None?

Looking at the code and tests:
- In production (`main.py`), tokenizer is ALWAYS passed to `compute_policy_loss`
- In tests, some older tests call `compute_policy_loss` without tokenizer parameter
- For vector queries (the current implementation), tokenizer is REQUIRED to reconstruct context

**Recommendation**: Since we're using vector queries, tokenizer should never be None. The function should fail fast with a clear error.

### 2. When would qkv_step not have available_key_embeddings?

Looking at the data flow:
- `available_key_embeddings` is added via `setattr` in `main.py` line 254
- It's not part of the QKVStep dataclass definition (added dynamically)
- Tests sometimes create QKVStep without this field

**Recommendation**: For PPO with vector queries, we need available_key_embeddings to compute KL divergence properly. The function should fail with a clear error if missing.

### 3. When would current_similarities not be in locals()?

This is a code smell. The use of `'current_similarities' in locals()` checks whether a variable was defined in a previous conditional branch. This happens when:
- The code doesn't enter the branch that computes `current_similarities` (due to missing tokenizer or available_key_embeddings)
- The complex nested conditionals make it unclear which variables are available

**Recommendation**: Eliminate this pattern by restructuring the code to have clear, linear flow.

## Improved Code Structure

```python
def compute_policy_loss(...):
    # 1. Validate required inputs upfront
    if tokenizer is None:
        raise ValueError("Tokenizer is required for vector queries")
    
    # 2. Process each trajectory step
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        # Check required fields
        if not hasattr(qkv_step, 'similarity_scores') or qkv_step.similarity_scores is None:
            continue  # Skip invalid steps
            
        if not hasattr(qkv_step, 'available_key_embeddings') or qkv_step.available_key_embeddings is None:
            raise ValueError(f"Step {t} missing available_key_embeddings")
        
        # Now we know we have all required data - compute everything linearly
        current_similarities = compute_similarity(...)
        old_similarities = compute_similarity(...)
        
        # No need to check if variables exist - they're guaranteed to exist here
```

## Summary of Changes Made

1. **Fail fast on missing tokenizer**: Added explicit check at function start
2. **Fail fast on missing available_key_embeddings**: Added explicit check for each step
3. **Eliminated `in locals()` checks**: Restructured code to have linear flow
4. **Clearer error messages**: Specify exactly what's missing and why it's required

This makes the code more maintainable and helps catch configuration errors early rather than silently falling back to potentially incorrect behavior. 