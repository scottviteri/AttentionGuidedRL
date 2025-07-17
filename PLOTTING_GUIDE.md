# Plotting System Guide

## Overview

The AttentionGuidedRL project includes a comprehensive plotting system that tracks and visualizes training metrics. Both real-time plotting during training and post-hoc visualization are supported.

## Key Features

### 1. Automatic Plot Generation During Training
- Plots are automatically saved every 25 episodes and at the end of training
- Data is saved as pickle files for later re-generation
- Plots are saved in `logs/<timestamp>/plots/`

### 2. Metrics Tracked (20+ metrics)
- **Loss Components**: Total loss, policy loss, KL penalty
- **Rewards**: Average rewards with trend lines and baseline update markers
- **Model Comparisons**: Adapter, baseline (old), and reference model log probabilities
- **Advantages**: Raw advantages with distribution analysis
- **Trajectory Metrics**: Trajectory-level log probabilities
- **Gradient Health**: Overall gradient norms + per-layer LoRA gradient flow
- **Wikipedia Order**: Order consistency metric (0=reverse, 0.5=random, 1=perfect)
- **Policy Gradients**: Shows reinforcement direction
- **KL Divergence**: From reference model (π_ref)
- **Reward Variance**: Within-trajectory variance
- **Step Analysis**: Log probabilities by step position (early/mid/late training)
- **PPO Metrics**: Clipping ratios (π_θ/π_old)
- **NEW: LoRA Layer Analysis**: Per-layer gradient magnitudes (identifies which layers are learning)
- **NEW: Advantage Distribution**: Percentage of positive vs negative advantages (step-level learning signal)
- **NEW: Baseline Update Detection**: Vertical markers showing periodic model updates
- **NEW: Similarity Analysis**: Query-key similarity evolution, entropy, and discrimination ability

### 3. Plot Types Generated

#### Training Metrics (12 subplots)
- Comprehensive overview in a 3x4 grid with enhanced debugging focus
- **Plot 11**: LoRA Layer Gradient Flow (replaces redundant KL Loss Raw Values)
- **Plot 12**: Advantage Distribution Analysis (replaces Batch Selection Entropy) 
- Each metric has its own subplot with appropriate scaling
- Trend lines added where relevant
- Baseline update markers on key plots (rewards, advantages, etc.)

#### Loss Breakdown
- Stacked area chart showing policy loss vs KL penalty contribution
- Generated when 20+ episodes of data available

#### Similarity Analysis (NEW)
- Separate 2x2 detailed analysis of query-key similarity evolution
- Shows mean similarity, entropy (specificity), range (discrimination), and variability
- Includes baseline update markers to correlate with training phases

#### Step-Level Learning Analysis (NEW)
- Separate 2x2 analysis of step-level learning dynamics
- Learning signal strength (positive advantage percentage over time)
- Advantage magnitude evolution and variability analysis 
- Combined learning health score with baseline update correlations

### 4. Standalone Plot Generation

Use `generate_plots.py` to regenerate plots from saved data:

```bash
# Basic usage - saves in same directory as pickle file
python generate_plots.py logs/20250619-073435/plots/plot_data_episode_49.pkl

# Specify output directory
python generate_plots.py data.pkl --output-dir custom_plots/

# Override KL coefficient in labels
python generate_plots.py data.pkl --kl-coef 0.05
```

**Note**: No backward compatibility - pickle files must contain all enhanced debugging metrics. Older pickle files from before the plotting improvements will not work.

### 5. Custom Plotting

Use `custom_plot_example.py` as a template for creating custom visualizations:

```bash
python custom_plot_example.py logs/path/to/plot_data.pkl
```

This creates:
- Custom reward plots with smoothing
- Detailed loss analysis
- Advantage distribution histograms
- Multi-run comparison plots

### 6. Technical Details

- **Backend**: Uses matplotlib 'Agg' backend (non-interactive, suitable for servers)
- **Data Format**: Pickle files containing all metrics + metadata
- **Consistency**: Both training and standalone scripts produce identical plots
- **Resolution**: 150 DPI for all plots
- **File Formats**: PNG for all visualizations

## Common Use Cases

### Regenerate Plots After Training
```bash
# Find the latest plot data
latest_data=$(find logs -name "plot_data_latest.pkl" | sort | tail -1)
python generate_plots.py $latest_data
```

### Compare Multiple Runs
```bash
# The custom_plot_example.py automatically finds and compares multiple runs
python custom_plot_example.py logs/*/plots/plot_data_latest.pkl
```

### Create Publication-Quality Figures
```bash
# Modify generate_plots.py to adjust:
# - DPI (change from 150 to 300)
# - Figure size (adjust figsize parameter)
# - Font sizes (modify fontsize parameters)
# - Colors and styles
```

## Plot Data Structure

Each pickle file contains:
```python
{
    'training_steps': [...],
    'total_losses': [...],
    'policy_losses': [...],
    'kl_losses': [...],
    'avg_rewards': [...],
    # ... 17+ more existing metrics ...
    
    # NEW: Enhanced debugging metrics
    'lora_layer_gradients': {0: [...], 1: [...], 10: [...]},  # Per-layer gradient magnitudes
    'advantage_distributions': [{'positive_percentage': 65.2, 'negative_percentage': 34.8, ...}],
    'similarity_score_stats': [{'mean': 0.23, 'std': 0.18, 'entropy': 2.4, ...}],
    'metadata': {
        'episode': 49,
        'timestamp': '2025-06-19T07:37:21',
        'config': {
            'KL_PENALTY_COEFFICIENT': 0.1,
            'GAMMA': 0.99,
            # ... other config values ...
        }
    }
}
```

## Troubleshooting

### No Plots Generated During Training
- Check that episodes ≥ 25 (plots save every 25 episodes)
- Verify write permissions in logs directory
- Check for matplotlib errors in training.log

### Empty Plots
- Ensure training ran for at least a few episodes
- Check that data arrays in pickle file are non-empty
- Verify matplotlib backend is set correctly

### Different Results Between Training and Standalone
- Both now use identical code and backends
- Ensure using same pickle file for comparison
- Check for any custom configuration overrides 