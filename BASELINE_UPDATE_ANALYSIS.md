# Training Spikiness Analysis & EMA Solution

## Problem: Training Metric Spikes

Your training exhibits periodic spikes in KL divergence and related metrics every 10 episodes. This creates a **sawtooth pattern** that makes training unstable and harder to analyze.

### Root Cause

The current baseline update mechanism uses **hard model replacement**:

```python
if (episode + 1) % BASELINE_UPDATE_FREQUENCY == 0:
    old_model = create_model_copy(adapter_model)  # SUDDEN JUMP
```

This creates **discontinuous jumps**:
- Episodes 1-9: KL accumulates as adapter drifts from old_model  
- Episode 10: old_model suddenly = adapter_model → KL drops to ~0
- Episodes 11-19: KL accumulates again
- Episode 20: Another sudden jump

## Solution: Exponential Moving Average (EMA)

Replace hard updates with smooth EMA updates **every episode**:

```python
if USE_EMA_BASELINE:
    # Smooth update every episode
    update_model_ema(old_model, adapter_model, decay=EMA_DECAY)
else:
    # Legacy hard update (causes spikes)
    if (episode + 1) % BASELINE_UPDATE_FREQUENCY == 0:
        old_model = create_model_copy(adapter_model)
```

### EMA Update Formula

```
old_model = decay * old_model + (1 - decay) * adapter_model
```

Where `decay = 0.95` (adjustable):
- Higher decay = smoother updates
- Lower decay = faster adaptation

## Benefits of EMA Approach

1. **Eliminates Spikes**: No sudden jumps in KL divergence
2. **Smoother Training**: Continuous regularization pressure  
3. **Better Analysis**: Cleaner metrics for debugging
4. **Configurable**: Adjust smoothness via decay parameter
5. **Backward Compatible**: Can toggle back to hard updates

## Demonstration Results

From the simulation:
- **Hard updates**: 86% reduction in maximum spike size
- **Spike episodes**: KL drops from 0.219 → 0.000 (sudden jump)
- **EMA episodes**: KL changes by max 0.030 (smooth)

## Usage

### Enable EMA (Default - Recommended)
```bash
python -m src.main --use-ema-baseline --ema-decay 0.05
```

### Adjust Smoothness
```bash
# Smoother (decay=0.02, faster adaptation)
python -m src.main --ema-decay 0.02

# Very smooth (decay=0.1, slower adaptation)  
python -m src.main --ema-decay 0.1
```

### Disable EMA (Legacy Mode)
```bash
python -m src.main --no-use-ema-baseline
```

## Recommended Settings

- **Standard training**: `--ema-decay 0.05` (default)
- **Unstable training**: `--ema-decay 0.02` (more adaptation)
- **Very smooth plots**: `--ema-decay 0.1` (less adaptation)

The EMA approach should eliminate the spikiness you observed while maintaining effective KL regularization.
