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

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- PEFT (Parameter-Efficient Fine-Tuning) 0.4+
- Datasets 2.13+
- tqdm

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/attention-guided-rl.git
cd attention-guided-rl
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Run the tests to ensure everything is set up correctly:
```bash
python -m pytest
```

## Usage

### Model Selection

Choose which model to use by setting the `MODEL_TYPE` environment variable:

```bash
# Use GPT-2 (default, works on smaller GPUs)
export MODEL_TYPE=gpt2
python -m src.main

# Use Llama-3.2-3B (requires more GPU memory, typically 12GB+)
export MODEL_TYPE=llama
python -m src.main
```

If no `MODEL_TYPE` is specified, GPT-2 is used by default.

### Query Token Configuration

**NEW**: The project now supports using standard vocabulary tokens instead of special tokens for much better log probabilities.

By default, the system uses standard tokens from the existing vocabulary (e.g., "Query", "Search") instead of adding new special tokens like `<VECTOR_QUERY>`. This provides dramatically better log probabilities since the model has seen these tokens during pre-training.

#### Token Configuration Options:

```bash
# Use standard tokens (default - RECOMMENDED)
export USE_STANDARD_QUERY_TOKEN=true  # Default: true
python -m src.main

# Use specific standard token
export USE_STANDARD_QUERY_TOKEN=true
export QUERY_TOKEN='Search'  # Options: Query, Search, Find, What, ?
python -m src.main

# Use original special token approach (not recommended - gives poor log probabilities)
export USE_STANDARD_QUERY_TOKEN=false
python -m src.main
```

**Performance Impact**: Using standard tokens improves log probabilities by 10-12 points compared to special tokens (e.g., from -17.49 to -4.96), which should significantly improve training quality.

### Baseline Model Configuration

**NEW**: The project now uses a simplified 3-model architecture with configurable baseline update frequency to fix KL divergence issues.

The previous 4-model setup caused KL loss to always be 0. The new simplified architecture uses:
1. **`base_model`** - Original model (for reward computation)
2. **`adapter_model`** - Trainable model with LoRA
3. **`baseline_model`** - Single baseline for both key embeddings AND KL computation

#### Baseline Update Configuration:

```bash
# Default: Update baseline every 25 episodes
export BASELINE_UPDATE_FREQUENCY=25
python -m src.main

# More frequent updates (faster adaptation, less KL accumulation)
export BASELINE_UPDATE_FREQUENCY=10
python -m src.main

# Less frequent updates (more KL accumulation, more stability)
export BASELINE_UPDATE_FREQUENCY=50
python -m src.main
```

**KL Divergence Behavior**: 
- KL accumulates over episodes until baseline update (no longer always 0!)
- Higher frequencies = faster adaptation but less regularization
- Lower frequencies = more regularization but slower adaptation to learning progress

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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
