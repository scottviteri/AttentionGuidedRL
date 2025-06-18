# Configuration Parameters Usage Audit

## Summary

This document tracks which configuration parameters from `src/config.py` are actually used in the codebase.

## ✅ Used Parameters

### Model Configuration
- **MODEL_NAME**: Used in model loading
- **TOKENIZER_NAME**: Used for tokenizer setup
- **MODEL_TYPE**: Used to determine target modules for LoRA
- **DEVICE**: Used throughout for tensor placement
- **DTYPE**: Used for model precision

### LoRA Parameters
- **LORA_RANK**: Used in LoRA config
- **LORA_ALPHA**: Used in LoRA config
- **LORA_DROPOUT**: Used in LoRA config

### Data Parameters
- **TOKENS_PER_QUERY**: Used in query generation
- **TOKENS_PER_KEY**: Used in data loading
- **TOKENS_PER_VALUE**: Used in data loading
- **KV_EVERY_N**: Used in data loading to control stride

### Prompt Formatting
- **QUERY_PREFIX**: Used in trajectory generation
- **KEY_PREFIX**: Used in trajectory generation
- **VALUE_PREFIX**: Used in trajectory generation
- **INITIAL_PROMPT**: Used as context initialization

### Training Parameters
- **NUM_EPISODES**: Used in training loop
- **WARMUP_EPISODES**: Used in filtering logic
- **LEARNING_RATE**: Used for optimizer (via CLI args)
- **KL_PENALTY_COEFFICIENT**: Used in loss computation
- **GRADIENT_CLIP_NORM**: Used in training step
- **GAMMA**: Used in return computation
- **GAE_LAMBDA**: Used in advantage computation (after our fix)
- **ENTROPY_COEF**: Used in loss computation
- **POLICY_SIGMA**: Used for Gaussian policy
- **USE_GRPO_BASELINE**: Used in advantage computation
- **TRAINING_BATCH_SIZE**: Used as default for CLI --batch-size

### Generation Parameters
- **TEMPERATURE**: Used in query generation and similarity computation
- **TOP_P**: Used in query generation

### Checkpoint Parameters
- **CHECKPOINT_DIR**: Used for saving checkpoints
- **CHECKPOINT_INTERVAL**: Used to control checkpoint frequency

### Logging
- **ENABLE_WANDB**: Used to control W&B logging
- **LOG_INTERVAL**: Used via CLI args for logging frequency

### Vector Query Parameters
- **QUERY_VEC_TOKEN**: Used when adding special tokens
- **USE_VECTOR_QUERIES**: Controlled via CLI, determines query mode

## ⚠️ Not Currently Used

### Generation Parameters
- **GENERATION_BATCH_SIZE**: Originally intended for larger batch generation during trajectory creation, but currently the same batch size is used throughout (controlled by --batch-size CLI argument)

## Calculated Parameters

These are computed dynamically based on other parameters:
- **PREFIX_TOKENS_PER_QUERY**: Calculated from tokenizer
- **PREFIX_TOKENS_PER_KEY**: Calculated from tokenizer
- **PREFIX_TOKENS_PER_VALUE**: Calculated from tokenizer
- **TOKENS_PER_KV_PAIR**: Sum of key and value tokens
- **INITIAL_PROMPT_TOKENS**: Calculated from tokenizer
- **TOKENS_PER_ROUND**: Total tokens per QKV round
- **NUM_KV_PAIRS**: Calculated based on context window

## Recommendations

1. **GENERATION_BATCH_SIZE**: Could be implemented to allow larger batch generation during trajectory creation for efficiency
2. All other parameters are properly used throughout the codebase
3. The config file is well-organized and parameters are appropriately grouped 