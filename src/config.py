"""
Configuration module for the Attention-Guided RL project.

Contains all the configuration constants used throughout the project.
"""

import torch
from transformers import AutoConfig, AutoTokenizer

MODEL_TYPE = 'gpt2'
MODEL_NAME = 'gpt2'
TOKENIZER_NAME = 'gpt2'

DEVICE = "cuda" 
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Query configuration - Vector queries only
QUERY_VEC_TOKEN = 'Query'  # Hardcoded to 'Query'

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
NUM_KV_PAIRS = min(NUM_KV_PAIRS, 10)  # Cap at 15 for reasonable trajectory length

# Key embedding batch size - number of keys to process together in a single forward pass
# Higher values improve GPU utilization but use more memory
KEY_EMBEDDING_BATCH_SIZE = 4  # Process 4 keys at once by default

# LoRA configuration
LORA_RANK = 8 
LORA_ALPHA = 16
LORA_DROPOUT = 0.0  # Disabled dropout to avoid non-determinism during inference

# Training configuration
LEARNING_RATE = 5e-4
GRADIENT_CLIP_NORM = 1.0
NUM_EPISODES = 10000
CHECKPOINT_INTERVAL = 100
TRAINING_BATCH_SIZE = 4  # Used for trajectory generation and training

# Reward computation - no scaling needed

# KL penalty - increased significantly to provide meaningful regularization
# Higher values mean stronger KL penalty, more stable but potentially slower learning
KL_PENALTY_COEFFICIENT = 0.1  # Increased from 0.01 to 0.1 (10x stronger)

# Directory configuration
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Wandb configuration
ENABLE_WANDB = False
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

# PPO clipping parameter (epsilon) for clipped surrogate objective
PPO_CLIP_EPSILON = 0.2  # Standard PPO clipping range

# Baseline model update frequency (how often to update the single baseline model)
# Since baseline model is now only used for key embeddings (not KL), we can update more frequently
BASELINE_UPDATE_FREQUENCY = 20  # More frequent updates (reduced from 50)

# Exponential Moving Average (EMA) for smooth baseline updates
# Set to 0.0 to use the old hard update mechanism (create_model_copy every N episodes)
# Set to 0.01-0.1 for smooth updates (recommended: 0.05 for stable training)
EMA_DECAY = 0.01  # Higher = smoother updates, Lower = faster adaptation

# When using EMA, update baseline every episode instead of every N episodes
USE_EMA_BASELINE = False  # Set to False to use the old hard update method

# Reward computation configuration
# Whether to subtract base model log probabilities from adapter log probabilities when computing rewards
# True: reward = adapter_log_prob - base_log_prob (classic baseline subtraction)
# False: reward = adapter_log_prob (raw adapter performance, let GRPO handle baselines)
SUBTRACT_BASE_MODEL_LOGPROBS = True 