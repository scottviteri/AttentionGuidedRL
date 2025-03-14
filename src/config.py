"""
Configuration parameters for the Attention-Guided RL project.
"""
import torch
from transformers import AutoConfig

# Determine device and choose model based on available GPU memory.
if torch.cuda.is_available():
    device = "cuda"
    # Get CUDA device properties.
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory = gpu_props.total_memory  # in bytes
    # 12 GB = 12 * 1024 * 1024 * 1024 bytes
    if total_memory < 12 * 1024 * 1024 * 1024:
        # Less than 12GB of GPU memory, use a lighter model: GPT-2.
        MODEL_NAME = "gpt2"
        TOKENIZER_NAME = "gpt2"
    else:
        MODEL_NAME = "meta-llama/Llama-3.2-3B"
        TOKENIZER_NAME = "meta-llama/Llama-3.2-3B"
else:
    device = "cpu"
    # Use GPT-2 on CPU for resource reasons.
    MODEL_NAME = "gpt2"
    TOKENIZER_NAME = "gpt2"

DEVICE = device
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# LoRA parameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Data parameters
TOKENS_PER_KEY = 20
TOKENS_PER_VALUE = 20
TOKENS_PER_KV_PAIR = TOKENS_PER_KEY + TOKENS_PER_VALUE

KV_EVERY_N = 4

model_config = AutoConfig.from_pretrained(MODEL_NAME)
if MODEL_NAME.startswith("meta-llama"):
    context_length = model_config.max_position_embeddings
else:
    context_length = model_config.n_positions

NUM_KV_PAIRS = context_length // (TOKENS_PER_KV_PAIR * KV_EVERY_N)

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