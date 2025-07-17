# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

This repository implements an attention-guided reinforcement learning framework that enables a base language model to autonomously guide its training by sequencing non-overlapping key-value pairs from Wikipedia articles.

## Overview

The system uses a base language model (Llama-3.2-3B or GPT-2, manually configurable) to generate queries, and an attention mechanism to select the most relevant key-value pairs from a pool of options. The model is then trained using reinforcement learning, with rewards based on the improvement in predicting values given the context and query.

Key features:
- Attention-guided selection of key-value pairs using embeddings from the model's last attention layer
- Support for both Llama and GPT-2 architectures with manual configuration
- Parameter-efficient training using LoRA adapters
- Self-directed curriculum learning via reinforcement learning
- Extensive test coverage for reliability (66 tests covering all components)
- Support for multiple datasets (Wikipedia and Twenty Questions)

## Requirements

**⚠️ CUDA GPU Required**: This project requires a CUDA-compatible GPU and does not support CPU-only execution.

- Python 3.8+
- CUDA-compatible GPU with CUDA 11.8+ or 12.x
- PyTorch 2.0+ (CUDA-enabled)
- Transformers 4.45+ (required for proper quantization support)
- PEFT (Parameter-Efficient Fine-Tuning) 0.4+
- Datasets 2.13+
- tqdm
- bitsandbytes 0.40+ (for 8-bit quantization, requires CUDA)
- accelerate 0.22+ (for device management)

## Setup

1. **Verify CUDA availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/attention-guided-rl.git
cd attention-guided-rl
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Run the tests to ensure everything is set up correctly:
```bash
python -m pytest
```

## Usage

### Model Selection

- Choose which backbone to use with the **`--model-type`** CLI flag (preferred) or the `MODEL_TYPE` environment variable.

```bash
# GPT-2 (default; lighter GPU requirements)
python -m src.main --model-type gpt2

# Llama-3.2-3B (≈12 GB GPU VRAM required)
python -m src.main --model-type llama
```

If neither the flag nor the env-var is supplied, GPT-2 is used by default.

### Query Token Configuration
- **Deprecated**: The project now always uses the standard vocabulary token "Query" for the vector-query placeholder, so no extra configuration is required.

### Baseline Model Configuration

**NEW**: The project now uses a simplified 3-model architecture with configurable baseline update frequency to fix KL divergence issues.

The previous 4-model setup caused KL loss to always be 0. The new simplified architecture uses:
1. **`base_model`** - Original model (for reward computation)
2. **`adapter_model`** - Trainable model with LoRA
3. **`baseline_model`** - Single baseline for both key embeddings AND KL computation

#### Baseline Update Configuration:

```bash
# Default baseline update frequency (every **10** episodes)
python -m src.main --baseline-update-freq 10

# Less frequent updates (e.g. every 30 episodes) – more regularisation
python -m src.main --baseline-update-freq 30

# More frequent updates (e.g. every 5 episodes) – faster policy adaptation
python -m src.main --baseline-update-freq 5
```

**KL Divergence Behavior**: 
- KL accumulates over episodes until baseline update (no longer always 0!)
- Higher frequencies = faster adaptation but less regularization
- Lower frequencies = more regularization but slower adaptation to learning progress

### KL Regularization Configuration
```bash
# Default KL penalty coefficient (0.1 - 10x stronger than original)
export KL_PENALTY_COEFFICIENT=0.1
python -m src.main

# Stronger KL regularization for more stability
export KL_PENALTY_COEFFICIENT=0.2
python -m src.main

# Weaker KL regularization for faster learning
export KL_PENALTY_COEFFICIENT=0.05
python -m src.main

# Combined KL and baseline configuration
export KL_PENALTY_COEFFICIENT=0.15
export BASELINE_UPDATE_FREQUENCY=75
python -m src.main
```

### GRPO-style Batching (NEW)

By default the trainer uses the **GRPO** batching strategy in which each data item is repeated `batch_size` times. This is enabled with `--grpo-batching` (default **True**). Disable it to sample distinct items per batch position:

```bash
# Standard GRPO batching (repeat items)
python -m src.main --grpo-batching

# Classic distinct-sample batching
python -m src.main --no-grpo-batching
```

### Training with Wikipedia

Run the training with default parameters (Wikipedia dataset):
```bash
python -m src.main
```

### Training with Twenty Questions Dataset

Run training with the Twenty Questions dataset:
```bash
python -m src.main --dataset twenty_questions
```

### Custom Parameters

With custom parameters:
```bash
python -m src.main --batch-size 4 --episodes 1000 --dataset wikipedia
```

You can also combine model selection with custom parameters:
```bash
# Train Llama with Twenty Questions dataset
MODEL_TYPE=llama python -m src.main --dataset twenty_questions --episodes 500

# Train GPT-2 with custom batch size
MODEL_TYPE=gpt2 python -m src.main --batch-size 8 --episodes 2000
```

Available dataset options:
- `wikipedia` (default): Uses Wikipedia articles split into key-value pairs
- `twenty_questions`: Uses a structured 20 questions game dataset

## Project Structure

```
attention-guided-rl/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py                 # Entry point for Wikipedia training
│   ├── model.py                # Model setup with LoRA adaptation
│   ├── embeddings.py           # Embedding extraction and similarity computation
│   ├── data.py                 # Functional iterator-based dataloader
│   ├── training.py             # RL training loop and policy optimization
│   └── config.py               # Configuration parameters
├── data/                       # Generated datasets
├── visualizations/             # Analysis visualizations
└── tests/
    ├── test_model.py           # Tests for model setup
    ├── test_embeddings.py      # Tests for embedding extraction
    ├── test_data.py            # Tests for data loading
    ├── test_training.py        # Tests for training loop
    └── test_main.py            # Tests for main entry point
```

## Implementation Details

### Embedding Extraction

Embeddings are extracted from the last attention layer of the model, with different implementation strategies for Llama (which uses grouped query attention) and GPT-2 architectures.

### Training Loop

1. Generate a query based on the current context
2. Extract query embeddings
3. Compute similarity with available key embeddings
4. Sample a key-value pair based on similarity
5. Add the pair to the context
6. Repeat to build a trajectory
7. Compute rewards by comparing log probabilities
8. Update the policy using REINFORCE with KL regularization

### Checkpointing

The model is saved periodically (every 100 episodes by default) and at the end of training. Training can be resumed from the latest checkpoint using the `--resume` flag.

## Plotting and Visualization

### Automatic Plot Data Saving

**NEW**: The training now automatically saves all plotting data to pickle files, allowing you to regenerate plots or create custom visualizations without re-running training. Additionally, text-based analysis optimized for LM consumption is automatically generated alongside plots.

During training, plot data is saved:
- Every 15 episodes and at end of training: `logs/<timestamp>/plots/plot_data.pkl` (overwrites previous)
- Text analysis: `plot_data_analysis.txt` and `plot_data_analysis.json` (automatically generated)

**Note**: Now uses clean `PlotData` dataclass structure with single file approach. No backward compatibility with older pickle files.

### Regenerating Plots

Use the standalone plotting script to regenerate plots from saved data:

```bash
# Generate plots from the latest run
python generate_plots.py logs/*/plots/plot_data.pkl

# Generate plots with custom output directory
python generate_plots.py logs/*/plots/plot_data_episode_100.pkl -o custom_plots/

# Override configuration for labels
python generate_plots.py data.pkl --kl-coef 0.2
```

### Text Analysis Generation

Generate LM-friendly analysis from saved plot data:

```bash
# Generate text analysis from latest run
python generate_text_analysis.py logs/*/plots/plot_data.pkl

# Generate analysis with custom output directory
python generate_text_analysis.py data.pkl --output-dir analysis_results/
```

This creates two files optimized for language model consumption:
- `plot_data_analysis.txt`: Human-readable training report with learning health overview
- `plot_data_analysis.json`: Structured data for programmatic analysis

### Creating Custom Plots

Use the example script to create custom visualizations:

```bash
# Run the custom plotting examples
python custom_plot_example.py
```

This will generate:
- `custom_reward_plot.png` - Reward plot with variance shading and smoothing
- `loss_analysis.png` - Detailed loss component analysis
- `advantage_dist.png` - Advantage distribution analysis
- `comparison_plot.png` - Multi-run comparison (if multiple runs exist)

### Custom Plot Examples

```python
import pickle
import matplotlib.pyplot as plt

# Load saved data
with open('logs/latest/plots/plot_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access any metric
training_steps = data['training_steps']
rewards = data['avg_rewards']
advantages = data['avg_advantages']

# Create your own plots
plt.figure(figsize=(10, 6))
plt.plot(training_steps, rewards, 'b-', label='Rewards')
plt.plot(training_steps, advantages, 'r--', label='Advantages')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.legend()
plt.savefig('my_custom_plot.png')
```

### Available Metrics in Saved Data

The pickle files contain all training metrics:
- `training_steps` - Episode numbers
- `total_losses`, `policy_losses`, `kl_losses` - Loss components
- `avg_rewards` - Average rewards per episode
- `adapter_log_probs`, `baseline_log_probs`, `base_log_probs` - Model log probabilities
- `avg_advantages` - Average advantages
- `trajectory_log_probs` - Trajectory-level log probabilities
- `wikipedia_order_consistency` - Order consistency metric (Wikipedia dataset)
- `kl_penalty_terms` - KL penalty values
- `reward_variance` - Variance within trajectories
- `gradient_magnitudes` - Gradient norms
- `step_log_probs` - Log probabilities by step index
- `policy_gradients` - Policy gradient values
- `clipping_ratios` - PPO clipping ratios
- `kl_from_ref` - KL divergence from reference model
- `metadata` - Training configuration and timestamp

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
