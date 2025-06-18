"""
Configuration module for the Attention-Guided RL project.

Contains all the configuration constants used throughout the project.
"""

import os
import torch
from transformers import AutoConfig, AutoTokenizer

# Model configuration - manually configurable via environment variable
# Set MODEL_TYPE environment variable to choose model:
# - "gpt2": Use GPT-2 (default, works on smaller GPUs)
# - "llama": Use Llama-3.2-3B (requires more GPU memory)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "gpt2").lower()

if MODEL_TYPE == "llama":
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-3B"
elif MODEL_TYPE == "gpt2":
    MODEL_NAME = "gpt2"
    TOKENIZER_NAME = "gpt2"
else:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'gpt2' or 'llama'")

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = device
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Query configuration - Vector queries only
QUERY_VEC_TOKEN = "<VECTOR_QUERY>"

# Option to use standard vocabulary tokens instead of special tokens
# This should give much better log probabilities since the model has seen these during pre-training
USE_STANDARD_QUERY_TOKEN = os.environ.get("USE_STANDARD_QUERY_TOKEN", "true").lower() == "true"

# Standard query tokens for each model type
# These are common tokens that both models should handle well
STANDARD_QUERY_TOKENS = {
    "gpt2": "Query",      # Simple, semantic word that GPT-2 knows well
    "llama": "Query",     # Same for Llama - consistency across models
}

# Alternative options (can be changed by setting environment variable)
ALTERNATIVE_STANDARD_TOKENS = {
    "gpt2": ["Query", "?", "Find", "Search", "What"],
    "llama": ["Query", "?", "Find", "Search", "What"],
}

# Allow override via environment variable
CUSTOM_QUERY_TOKEN = os.environ.get("QUERY_TOKEN", None)

# Determine the actual token to use
if USE_STANDARD_QUERY_TOKEN:
    if CUSTOM_QUERY_TOKEN:
        QUERY_VEC_TOKEN = CUSTOM_QUERY_TOKEN
    else:
        QUERY_VEC_TOKEN = STANDARD_QUERY_TOKENS.get(MODEL_TYPE, "Query")
else:
    # Use the original special token
    QUERY_VEC_TOKEN = "<VECTOR_QUERY>"

# Prefix tokens for context building
KEY_PREFIX = "Key: "
VALUE_PREFIX = "Value: "

# Core token counts - these are the actual content tokens
TOKENS_PER_KEY = 10
TOKENS_PER_VALUE = 10

# Initialize tokenizer to calculate prefix lengths
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Calculate actual token lengths for the prefixes
PREFIX_TOKENS_PER_KEY = len(tokenizer.encode(KEY_PREFIX, add_special_tokens=False))
PREFIX_TOKENS_PER_VALUE = len(tokenizer.encode(VALUE_PREFIX, add_special_tokens=False))

# Total tokens per round (no query tokens for vector queries)
TOKENS_PER_ROUND = (
    PREFIX_TOKENS_PER_KEY + TOKENS_PER_KEY +
    PREFIX_TOKENS_PER_VALUE + TOKENS_PER_VALUE
)

# Initial prompt for vector queries
INITIAL_PROMPT = "Search for relevant information using learned vector queries."
INITIAL_PROMPT_TOKENS = len(tokenizer.encode(INITIAL_PROMPT, add_special_tokens=False))

# Context window configuration
model_config = AutoConfig.from_pretrained(MODEL_NAME)
if MODEL_TYPE == "llama":
    MAX_CONTEXT_LENGTH = model_config.max_position_embeddings
else:
    MAX_CONTEXT_LENGTH = model_config.n_positions

# Spacing between key-value pairs when extracting from text
KV_EVERY_N = 4  # Skip 4 chunks between each extraction for diversity

# Number of key-value pairs
available_context = MAX_CONTEXT_LENGTH - INITIAL_PROMPT_TOKENS
NUM_KV_PAIRS = available_context // TOKENS_PER_ROUND
NUM_KV_PAIRS = min(NUM_KV_PAIRS, 15)  # Cap at 15 for reasonable trajectory length

# LoRA configuration
LORA_RANK = 8 
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training configuration
LEARNING_RATE = 5e-4
GRADIENT_CLIP_NORM = 1.0
NUM_EPISODES = 10000
CHECKPOINT_INTERVAL = 100
TRAINING_BATCH_SIZE = 40  # Used for trajectory generation and training

# Reward computation - no scaling needed

# KL penalty
KL_PENALTY_COEFFICIENT = 0.01  # Beta coefficient for KL penalty

# Directory configuration
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Wandb configuration
ENABLE_WANDB = os.environ.get("ENABLE_WANDB", "false").lower() == "true"
WANDB_PROJECT = "attention-guided-rl"

# Logging
LOG_INTERVAL = 10

# Policy gradient configuration
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # For GAE (if using value function)


# Always use GRPO baseline - no warmup needed
USE_GRPO_BASELINE = True

# Temperature for softmax in similarity computation
TEMPERATURE = 1.0 

# Step-level advantage filtering
USE_POSITIVE_ADVANTAGES_ONLY = True  # Changed to True - only positive advantages contribute, but keeps all trajectories

# Baseline model update frequency (how often to update the single baseline model)
# Higher values = more KL divergence accumulation, more stability
# Lower values = faster adaptation to learning progress
BASELINE_UPDATE_FREQUENCY = int(os.environ.get("BASELINE_UPDATE_FREQUENCY", "25"))  # Default: every 25 episodes