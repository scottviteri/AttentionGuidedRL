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
- 20 Questions dataset generation for training and evaluation

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

### Training with Wikipedia

Run the training with default parameters:
```bash
python -m src.main
```

With custom parameters:
```bash
python -m src.main --batch-size 4 --trajectory-length 5 --episodes 1000
```

### Training with Twenty Questions Dataset

The repository includes a modified training procedure that uses the twenty questions dataset with a different reinforcement learning task:

```bash
python -m src.twenty_questions_train --trajectory-length 10 --episodes 2000
```

#### Command-line Arguments for Twenty Questions Training

- `--dataset`: Path to the dataset file (default: uses data/20q_dataset.json)
- `--episodes`: Number of episodes to train for (default: 1000)
- `--batch-size`: Batch size for training (default: 1)
- `--trajectory-length`: Number of question-answer pairs in each trajectory (default: 5)
- `--learning-rate`: Learning rate for optimization (default: from config)
- `--kl-penalty`: KL penalty coefficient (default: from config)
- `--log-interval`: Interval for logging statistics (default: from config)
- `--resume`: Resume training from the latest checkpoint

### Dataset Generation

The repository includes scripts for generating a dataset for the 20 Questions game using Claude API:

```bash
python scripts/twenty_questions/generate_20q_dataset.py
```

For analyzing the generated dataset:

```bash
python scripts/twenty_questions/analyze_20q_dataset.py --visualize
```

See `scripts/twenty_questions/README.md` for more details.

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
│   ├── config.py               # Configuration parameters
│   ├── twenty_questions_data.py  # Data handling for 20Q dataset
│   └── twenty_questions_train.py # Training script for 20Q dataset
├── scripts/
│   └── twenty_questions/       # 20 Questions dataset generation
│       ├── generate_20q_dataset.py
│       ├── analyze_20q_dataset.py
│       └── README.md
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

### Twenty Questions Training

The twenty questions training procedure follows the same general reinforcement learning approach, but adapted to simulate a 20 questions game:

1. Start with a generic prompt that doesn't reveal the object ("I am thinking of an object...")
2. Generate a question based on current context
3. Use attention to select the most relevant predefined question from the dataset
4. Retrieve the corresponding YES/NO answer for the selected question
5. Add the question and its answer to the context
6. Repeat to build a trajectory of question-answer pairs
7. Compute rewards based on improvement in predicting answers
8. Update the policy using REINFORCE with KL regularization

The model learns to ask effective questions that efficiently narrow down the possible objects. The reward function encourages the model to select questions that provide the most information gain about the hidden object.

### Checkpointing

The model is saved periodically (every 100 episodes by default) and at the end of training. Training can be resumed from the latest checkpoint using the `--resume` flag.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
