# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

This repository implements an attention-guided reinforcement learning framework that enables a base language model to autonomously guide its training by sequencing non-overlapping key-value pairs from Wikipedia articles.

## Overview

The system uses a base language model (Llama-3.2-3B or GPT-2 depending on available GPU resources) to generate queries, and an attention mechanism to select the most relevant key-value pairs from a pool of options. The model is then trained using reinforcement learning, with rewards based on the improvement in predicting values given the context and query.

Key features:
- Attention-guided selection of key-value pairs using embeddings from the model's last attention layer
- Support for both Llama and GPT-2 architectures with seamless switching based on available GPU resources
- Parameter-efficient training using LoRA adapters
- Self-directed curriculum learning via reinforcement learning
- Extensive test coverage for reliability

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

### Training

Run the training with default parameters:
```bash
python -m src.main
```

With custom parameters:
```bash
python -m src.main --batch-size 4 --trajectory-length 5 --episodes 1000
```

### Command-line Arguments

- `--batch-size`: Number of examples to process in parallel (default: 2)
- `--resume`: Resume training from the latest checkpoint
- `--episodes`: Number of episodes to train for (default: 1000)
- `--trajectory-length`: Number of key-value pairs in each trajectory (default: 3)
- `--log-interval`: Interval for logging statistics (default: 10)

## Project Structure

```
attention-guided-rl/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py              # Entry point for training
│   ├── model.py             # Model setup with LoRA adaptation
│   ├── embeddings.py        # Embedding extraction and similarity computation
│   ├── data.py              # Functional iterator-based dataloader
│   ├── training.py          # RL training loop and policy optimization
│   └── config.py            # Configuration parameters
└── tests/
    ├── test_model.py        # Tests for model setup
    ├── test_embeddings.py   # Tests for embedding extraction
    ├── test_data.py         # Tests for data loading
    ├── test_training.py     # Tests for training loop
    └── test_main.py         # Tests for main entry point
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
