"""
Configuration parameters for the Attention-Guided RL project.
"""
import torch

# Model parameters
MODEL_NAME = "meta-llama/Llama-3.2-3B"
TOKENIZER_NAME = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# LoRA parameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Data parameters
TOKENS_PER_KEY = 20
TOKENS_PER_VALUE = 20
TOKENS_PER_KV_PAIR = TOKENS_PER_KEY + TOKENS_PER_VALUE
NUM_KV_PAIRS = 8
KV_SUBSET_FRACTION = 0.25

# Training parameters
NUM_EPISODES = 10000
WARMUP_EPISODES = 15
LEARNING_RATE = 2e-4
KL_PENALTY_COEFFICIENT = 0.1
GRADIENT_CLIP_NORM = 1.0

# Generation parameters
GENERATION_BATCH_SIZE = 64
TRAINING_BATCH_SIZE = 16
TEMPERATURE = 1.0
TOP_P = 0.9

# Prompt formatting
QUERY_PREFIX = " Query: "
RESPONSE_PREFIX = " Response: "

# Checkpoint parameters
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 100

# Logging
ENABLE_WANDB = False
LOG_INTERVAL = 10 