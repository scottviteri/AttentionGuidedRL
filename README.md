# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

This repository implements a novel **attention-guided reinforcement learning framework** where language models learn to autonomously curate their own training data. The system uses attention mechanisms to select relevant key-value pairs from Wikipedia articles, creating a self-directed curriculum learning approach.

## Research Overview

The core innovation is an RL system where:

1. **Query Generation**: The model generates vector queries using embeddings from its attention layers
2. **Attention-Guided Selection**: Multi-head attention computes similarity between queries and available key embeddings
3. **Sequential Decision Making**: The model builds trajectories by iteratively selecting key-value pairs
4. **Self-Supervised Rewards**: Training progress is measured by improvement in predicting values given context
5. **Policy Optimization**: Advantage-based policy gradients with a chain-rule reward term train the model to make better selections

This creates a feedback loop where the model's internal representations directly guide what it learns next, enabling autonomous curriculum learning.

## Key Features

- **Attention-guided selection**: Uses embeddings from the model's attention layers for content selection
- **Self-directed learning**: Model autonomously sequences its training data based on internal representations  
- **Multi-model architecture**: Support for both Llama-3.2-3B and GPT-2 with unified interface
- **Robust RL training**: Advantage-based policy gradients with GRPO-style baselines and a differentiable reward term
- **Parameter efficiency**: LoRA adapters enable efficient training of large models
- **Comprehensive evaluation**: 66+ tests covering all components, extensive visualization tools
- **Multiple datasets**: Wikipedia articles and Twenty Questions for different learning scenarios

## Requirements

**⚠️ CUDA GPU Required**: This project requires a CUDA-compatible GPU and does not support CPU-only execution.

- Python 3.8+
- CUDA-compatible GPU with CUDA 11.8+ or 12.x
- PyTorch 2.0+ (CUDA-enabled)
- Transformers 4.45+ (required for proper quantization support)
- PEFT (Parameter-Efficient Fine-Tuning) 0.4+
- Datasets 2.13+
- Additional dependencies in `requirements.txt`

## Quick Start

1. **Setup:**
```bash
git clone https://github.com/yourusername/attention-guided-rl.git
cd attention-guided-rl
pip install -r requirements.txt
python -m pytest  # Verify installation
```

2. **Basic training:**
```bash
# Train with GPT-2 on Wikipedia (default)
python -m src.main

# Train with Llama on Twenty Questions dataset
python -m src.main --model-type llama --dataset twenty_questions
```

3. **View results:**
Training automatically saves plots and analysis to `logs/<timestamp>/plots/`

## Usage

### Model Selection

```bash
# GPT-2 (default, lighter GPU requirements)
python -m src.main --model-type gpt2

# Llama-3.2-3B (requires ~12GB GPU VRAM)  
python -m src.main --model-type llama
```

### Training Configuration

```bash
# Custom training parameters
python -m src.main --batch-size 8 --episodes 5000 --learning-rate 1e-4

# Reinforcement learning parameters
# Choose reward aggregation for advantages: average (default) or discounted
python -m src.main --reward-aggregation average
python -m src.main --reward-aggregation discounted --use-grpo-baseline --batch-size 8

# Enable differentiable reward term (chain rule)
python -m src.main --differentiable-rewards

# GRPO baseline (per-step batch mean) for variance reduction
python -m src.main --use-grpo-baseline
```

### Dataset Options

```bash
# Wikipedia articles (default)
python -m src.main --dataset wikipedia

# Twenty Questions structured data
python -m src.main --dataset twenty_questions  
```

### Training Modes

```bash
# Standard training
python -m src.main

# (Memory-efficient mode removed; default training uses a single adapter model)

# Resume from checkpoint
python -m src.main --resume
```

## Core Algorithm

### 1. Trajectory Generation
```python
# Simplified conceptual flow
for step in range(num_kv_pairs):
    query_vector = generate_query(context, model)
    similarities = compute_attention_similarity(query_vector, available_keys)
    selected_key_idx = sample_from_policy(similarities)
    context = append_key_value_pair(context, selected_key_idx)
```

### 2. Reward Computation
Rewards are based on the model's ability to predict value tokens:
```python
reward = log_prob_adapter(value | context, key) - log_prob_base(value | context, key)
```

### 3. Policy Optimization
Uses advantage-weighted policy gradient with an additional differentiable reward term (chain rule):
```python
loss = -sum_t(A_t * logpi_t) - λ * sum_t(r_t)
```

## Implementation Details

### Attention Mechanism
- Embeddings extracted from second-to-last attention layer
- Supports both Multi-Head Attention (GPT-2) and Grouped Query Attention (Llama)
- Temperature-scaled similarity computation for policy distribution

### Memory Efficiency
The system uses LoRA (Low-Rank Adaptation) for efficient training:
- LoRA adapters reduce memory usage by 60-90% compared to full fine-tuning
- Enables training larger models on smaller GPUs
- Only trains small adapter weights while keeping base model frozen

### Visualization and Analysis
Comprehensive plotting and analysis tools:
```bash
# Generate plots from saved training data
python scripts/generate_plots.py logs/*/plots/plot_data.pkl

# Generate text analysis for LLM consumption
python scripts/generate_text_analysis.py logs/*/plots/plot_data.pkl
```

## Project Structure

```
attention-guided-rl/
├── README.md
├── requirements.txt
├── scripts/                  # Standalone utilities (plotting, analysis, runners)
├── examples/                 # Example and walkthrough scripts
├── src/
│   ├── main.py               # Training entry point
│   ├── config.py             # TrainingConfig dataclass and utilities
│   ├── model.py              # Model setup with LoRA adaptation
│   ├── embeddings.py         # Attention-based embedding extraction
│   ├── data.py               # Dataset iterators and preprocessing
│   ├── training.py           # Training utilities and policy optimization
├── tests/                    # Comprehensive test suite (66+ tests)
├── logs/                     # Training outputs and visualizations
└── docs/                     # Additional documentation (math, plotting, design)
```

## Advanced Usage

### Checkpointing and Resuming
```bash
# Resume from latest checkpoint
python -m src.main --resume

# Training automatically saves checkpoints every 100 episodes
```

### Visualization and Analysis
Training automatically generates comprehensive visualizations:

```bash
# Generate plots from saved training data
python scripts/generate_plots.py logs/*/plots/plot_data.pkl

# Generate text analysis for LLM consumption  
python scripts/generate_text_analysis.py logs/*/plots/plot_data.pkl
```

### Custom Configuration
Advanced users can create custom configurations:

```bash
# Advanced RL parameters
python -m src.main \
  --kl-penalty-coef 0.15 \
  --baseline-update-freq 20 \
  --ppo-clip-epsilon 0.25 \
  --use-ema-baseline \
  --ema-decay 0.95
```

For complete documentation of all available options, see the TrainingConfig dataclass in `src/config.py`.

## Research Applications

This framework enables research into:
- **Curriculum learning**: How models can autonomously sequence training data
- **Attention mechanisms**: Using internal representations to guide learning
- **Self-supervised RL**: Learning without external reward signals
- **Meta-learning**: Models learning how to learn more effectively

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
