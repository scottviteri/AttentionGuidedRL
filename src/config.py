"""
Configuration parameters for the Attention-Guided RL project.
"""
import torch
from transformers import AutoConfig, AutoTokenizer

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
        MODEL_TYPE = "gpt2"
    else:
        MODEL_NAME = "meta-llama/Llama-3.2-3B"
        TOKENIZER_NAME = "meta-llama/Llama-3.2-3B"
        MODEL_TYPE = "llama"
else:
    device = "cpu"
    # Use GPT-2 on CPU for resource reasons.
    MODEL_NAME = "gpt2"
    TOKENIZER_NAME = "gpt2"
    MODEL_TYPE = "gpt2"

DEVICE = device
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# LoRA parameters
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Data parameters
TOKENS_PER_QUERY = 10
TOKENS_PER_KEY = 10
TOKENS_PER_VALUE = 10

# Prompt formatting
QUERY_PREFIX = " Query: "
KEY_PREFIX = " Key: "
VALUE_PREFIX = " Value: "

# Initialize tokenizer to calculate prefix lengths
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Calculate actual token lengths for the prefixes
PREFIX_TOKENS_PER_QUERY = len(tokenizer.encode(QUERY_PREFIX, add_special_tokens=False))
PREFIX_TOKENS_PER_KEY = len(tokenizer.encode(KEY_PREFIX, add_special_tokens=False))
PREFIX_TOKENS_PER_VALUE = len(tokenizer.encode(VALUE_PREFIX, add_special_tokens=False))
TOKENS_PER_KV_PAIR = TOKENS_PER_KEY + TOKENS_PER_VALUE

# Calculate initial prompt token count
INITIAL_PROMPT = f"I am learning to pick the order of my training data by producing a natural language query of {TOKENS_PER_QUERY} tokens which will be used to select similar keys in a key-value dataset within a Wikipedia article. The goal is to predict the values, which are consecutive tokens from the article, over the whole multi-turn trajectory. "
INITIAL_PROMPT_TOKENS = len(tokenizer.encode(INITIAL_PROMPT, add_special_tokens=False))

# Total tokens per round calculation
TOKENS_PER_ROUND = (
    PREFIX_TOKENS_PER_QUERY + TOKENS_PER_QUERY + 
    PREFIX_TOKENS_PER_KEY + TOKENS_PER_KEY + 
    PREFIX_TOKENS_PER_VALUE + TOKENS_PER_VALUE
)

KV_EVERY_N = 4

model_config = AutoConfig.from_pretrained(MODEL_NAME)
if MODEL_TYPE == "llama":
    context_length = model_config.max_position_embeddings
else:
    context_length = model_config.n_positions

# Calculate max number of KV pairs that can fit in the context window
available_context_length = context_length - INITIAL_PROMPT_TOKENS
NUM_KV_PAIRS = available_context_length // (TOKENS_PER_ROUND * KV_EVERY_N)

# Training parameters
NUM_EPISODES = 10000
WARMUP_EPISODES = 1
LEARNING_RATE = 2e-4
KL_PENALTY_COEFFICIENT = 0.1
GRADIENT_CLIP_NORM = 1.0

# Generation parameters
GENERATION_BATCH_SIZE = 64
TRAINING_BATCH_SIZE = 16
TEMPERATURE = 1.0
TOP_P = 0.9

# Checkpoint parameters
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 5 

# Logging
ENABLE_WANDB = False
LOG_INTERVAL = 10 