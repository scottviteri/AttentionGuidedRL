# Recent Changes Summary

This document summarizes the major improvements made to the Attention-Guided RL project.

## 🔧 **Fixes Applied**

### 1. Standard Token Support for Improved Log Probabilities
**Issue**: Model log probabilities around -53 due to unknown special token `<VECTOR_QUERY>`
**Root Cause**: Special tokens not seen during pre-training confuse the model
**Solution**: Added option to use standard vocabulary tokens instead of special tokens
**Result**: Dramatic improvement in log probabilities (from -17.49 to -4.96, +12.53 improvement)
**Usage**: `export USE_STANDARD_QUERY_TOKEN=true` (now default)
**Best Tokens**: 'Search' (-4.96), 'Find' (-5.82), 'Query' (-7.30)

### 2. Simplified Model Architecture for Meaningful KL Divergence
**Issue**: KL Loss consistently showed 0.0000 due to overly complex 4-model setup
**Root Cause**: `previous_model` was reset to be identical to `adapter_model` after each episode
**Solution**: Simplified to 3-model architecture with single configurable baseline
**Architecture**: 
- `base_model` (original, for rewards)
- `adapter_model` (trainable with LoRA) 
- `baseline_model` (for both key embeddings AND KL computation)
**Result**: KL divergence now accumulates over episodes (0.000 → 0.056 → 0.102 → 0.000 reset)
**Configuration**: `export BASELINE_UPDATE_FREQUENCY=25` (default)

### 3. Policy Gradient Sign Convention Improvement  
**Issue**: Confusing sign conventions with multiple negations throughout computation  
**Solution**: Work with positive gradients throughout, single sign flip only at the end  
**Benefits**: More intuitive reasoning (positive = reinforce good actions)  
**Commit**: `b64eb83`

### 4. Machine-Dependent Test Behavior Fix
**Issue**: Tests behaved differently based on GPU memory (<12GB = GPT-2, ≥12GB = Llama)  
**Solution**: Manual model configuration via `MODEL_TYPE` environment variable  
**Benefits**: Deterministic tests across all machines, explicit user control  
**New Tests**: 7 comprehensive configuration tests added  
**Commit**: `40f89a0`

## 📚 **Documentation Updates**

### README.md
- Added model selection section with usage examples
- Documented `MODEL_TYPE` environment variable usage
- Provided clear guidance for choosing models based on GPU memory

### design_doc.md  
- Updated to reflect manual model selection approach
- Removed references to automatic GPU-based switching

**Commit**: `dbf2210`

## 🧪 **Testing Improvements**

### New Test Suite
- **tests/test_config.py**: 7 new tests for configuration validation
  - Default model type testing
  - Explicit model type setting
  - Case insensitivity verification
  - Invalid model type error handling
  - Configuration consistency across models
  - Device configuration validation

### Test Determinism
- Tests now behave consistently regardless of machine GPU memory
- No more machine-dependent failures
- All existing tests remain functional

## 🔄 **Enhanced Monitoring**

### Plotting Improvements
- Added policy gradient visualization (positive = reinforcement)
- Enhanced 6-panel dashboard for comprehensive monitoring
- Improved loss component labeling for clarity
- Log probability tracking and improvement metrics

### Training Insights
- Better gradient flow understanding
- Clearer visualization of what the model is learning
- Enhanced debugging capabilities

## 📋 **GitHub Issues Created & Resolved**

Created and immediately resolved documentation issues:
- **Issue #11**: KL divergence timing bug ✅ RESOLVED
- **Issue #12**: Machine-dependent test behavior ✅ RESOLVED  
- **Issue #13**: Policy gradient sign convention improvement ✅ IMPLEMENTED
- **Issue #14**: Enhanced training visualization ✅ IMPLEMENTED

## 🚀 **Usage**

### Model Selection
```bash
# Default (GPT-2)
python -m src.main

# Explicit GPT-2
MODEL_TYPE=gpt2 python -m src.main

# Llama for larger GPUs  
MODEL_TYPE=llama python -m src.main --dataset twenty_questions

# Combined with other parameters
MODEL_TYPE=gpt2 python -m src.main --batch-size 8 --episodes 2000
```

### Query Token Configuration
```bash
# Use standard tokens (RECOMMENDED - default)
export USE_STANDARD_QUERY_TOKEN=true
python -m src.main

# Use specific standard token for best results
export USE_STANDARD_QUERY_TOKEN=true
export QUERY_TOKEN='Search'  # Best performing token
python -m src.main

# Combined model + token configuration
MODEL_TYPE=llama USE_STANDARD_QUERY_TOKEN=true QUERY_TOKEN='Search' python -m src.main --dataset twenty_questions
```

### Baseline Model Configuration
```bash
# Default baseline update frequency (every 25 episodes)
export BASELINE_UPDATE_FREQUENCY=25
python -m src.main

# More KL accumulation (update every 50 episodes)
export BASELINE_UPDATE_FREQUENCY=50
python -m src.main

# Combined with other options
MODEL_TYPE=gpt2 USE_STANDARD_QUERY_TOKEN=true BASELINE_UPDATE_FREQUENCY=30 python -m src.main
```

### Running Tests
```bash
# All tests (now deterministic)
python -m pytest

# Configuration tests specifically
python -m pytest tests/test_config.py -v
```

## ✅ **Benefits Achieved**

1. **Dramatically Improved Log Probabilities**: Standard tokens give 10-12 point improvements (e.g., -17.49 → -4.96)
2. **Meaningful KL Divergence**: Simplified architecture allows KL to accumulate over episodes (no longer always 0)
3. **Reduced Complexity**: Simplified from 4 models to 3 models with clearer responsibilities
4. **Configurable Regularization**: Baseline update frequency controls KL accumulation vs. adaptation speed
5. **Deterministic Testing**: Tests behave consistently across all machines
6. **User Control**: Explicit model selection instead of automatic guessing
7. **Better Training Insights**: Enhanced monitoring and visualization
8. **Conceptual Clarity**: Intuitive sign conventions and clear documentation
9. **Comprehensive Testing**: 7 new tests ensure configuration reliability

## 📈 **Impact**

- **Reliability**: No more machine-dependent test failures
- **Usability**: Clear documentation and user control
- **Debugging**: Better training insights and visualization
- **Maintainability**: Cleaner code with intuitive conventions
- **Performance**: Proper KL regularization now functional 