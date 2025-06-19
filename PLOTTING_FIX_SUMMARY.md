# Plotting System Fix Summary

## Problem
The `generate_plots.py` script worked correctly when called manually, but the automatic plotting during training produced empty or incorrect plots.

## Root Cause
The training code had its own implementation of plotting logic that was potentially out of sync with `generate_plots.py`, leading to inconsistencies.

## Solution
Replaced the `plot_metrics()` function in `src/main.py` to directly call `generate_plots.py` via subprocess, ensuring both manual and automatic plotting use exactly the same code.

## Implementation Details

### Changes to `src/main.py`
The `plot_metrics()` function now:
1. Locates the latest pickle file (`plot_data_latest.pkl`)
2. Finds the `generate_plots.py` script path
3. Calls it via `subprocess.run()` with proper error handling
4. Captures and logs all output from the script

### Benefits
- **Consistency**: Both automatic and manual plotting use identical code
- **Maintainability**: Only one plotting implementation to maintain
- **Debugging**: All output from plotting is captured in training logs
- **Reliability**: Subprocess isolation prevents any matplotlib state issues

## Verification
Tested with a training run:
```bash
python src/main.py --episodes 26 --batch-size 10
```

Results show:
- Plot data saved at episode 25: ✓
- `generate_plots.py` called automatically: ✓
- Plots generated successfully: ✓
- Output logged to training.log: ✓

## Files Generated
- `training_metrics.png`: Main 12-subplot comprehensive view
- `loss_breakdown.png`: Detailed loss composition (when 20+ episodes)
- Both saved in `logs/<timestamp>/plots/` 