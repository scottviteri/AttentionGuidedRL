# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

This repository implements an RL-based active learning framework that enables a base language model (Llama-3.2-3B) to autonomously guide its training by sequencing non-overlapping key–value pairs from Wikipedia articles.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

## Features

- Self-directed curriculum learning via RL
- Attention-guided key-value pair selection
- Efficient context window utilization with non-overlapping pairs
- Trajectory filtering for high-quality updates
- LoRA for efficient fine-tuning

## Testing

```bash
pytest tests/
``` 