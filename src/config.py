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

# Core constants - basic values that don't require computation
TOKENS_PER_KEY = 10
TOKENS_PER_VALUE = 10
KV_EVERY_N = 4  # Skip 4 chunks between each extraction for diversity

# Fixed text constants
INITIAL_PROMPT = "Search for relevant information using learned vector queries."

# NOTE: All derived values (token counts, context lengths, num_kv_pairs) are now
# computed in TrainingConfig.from_args_and_defaults() to eliminate redundancy.

# Configuration constants (for backwards compatibility and TrainingConfig defaults)
LORA_RANK = 8 
LORA_ALPHA = 16
LEARNING_RATE = 5e-4
GRADIENT_CLIP_NORM = 1.0
NUM_EPISODES = 10000
CHECKPOINT_INTERVAL = 100
TRAINING_BATCH_SIZE = 4
KL_PENALTY_COEFFICIENT = 0.1
ENABLE_WANDB = False
LOG_INTERVAL = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
USE_GRPO_BASELINE = True
TEMPERATURE = 1.0 
PPO_CLIP_EPSILON = 0.2
USE_PPO = True
BASELINE_UPDATE_FREQUENCY = 50
EMA_DECAY = 0.99
USE_EMA_BASELINE = True
SUBTRACT_BASE_MODEL_LOGPROBS = False 
MEMORY_EFFICIENT_LORA = True 

# === Frozen Configuration Management ===

from dataclasses import dataclass
from typing import Any, Dict
import argparse

@dataclass(frozen=True)
class TrainingConfig:
    """
    Frozen configuration that resolves all values once from config.py defaults + CLI overrides.
    
    This eliminates the confusing pattern of:
    1. Import from config.py
    2. Update config.py values from CLI args  
    3. Mix usage of original imports vs config.X references
    
    Instead: Create one immutable config object with final resolved values.
    """
    # Core model configuration
    model_name: str = MODEL_NAME
    model_type: str = MODEL_TYPE
    tokenizer_name: str = TOKENIZER_NAME
    device: str = DEVICE
    
    # Training hyperparameters
    learning_rate: float = LEARNING_RATE
    num_episodes: int = NUM_EPISODES
    batch_size: int = TRAINING_BATCH_SIZE
    
    # RL-specific parameters
    kl_penalty_coefficient: float = KL_PENALTY_COEFFICIENT
    ppo_clip_epsilon: float = PPO_CLIP_EPSILON
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    temperature: float = TEMPERATURE
    
    # Training behavior
    use_grpo_baseline: bool = USE_GRPO_BASELINE
    use_ema_baseline: bool = USE_EMA_BASELINE
    ema_decay: float = EMA_DECAY
    baseline_update_frequency: int = BASELINE_UPDATE_FREQUENCY
    subtract_base_model_logprobs: bool = SUBTRACT_BASE_MODEL_LOGPROBS
    use_ppo: bool = True  # vs vanilla policy gradient (default to PPO)
    memory_efficient_lora: bool = MEMORY_EFFICIENT_LORA
    grpo_batching: bool = True  # GRPO-style batching
    
    # Training infrastructure
    checkpoint_interval: int = CHECKPOINT_INTERVAL
    log_interval: int = LOG_INTERVAL
    enable_wandb: bool = ENABLE_WANDB
    
    # Token configuration (computed - will be filled by create_training_config_from_args)
    prefix_tokens_per_key: int = 0
    prefix_tokens_per_value: int = 0
    tokens_per_round: int = 0
    initial_prompt_tokens: int = 0
    num_kv_pairs: int = 0
    max_context_length: int = 0
    
    # Fixed constants (rarely changed)
    tokens_per_key: int = TOKENS_PER_KEY
    tokens_per_value: int = TOKENS_PER_VALUE
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    kv_every_n: int = KV_EVERY_N
    initial_prompt: str = INITIAL_PROMPT
    key_prefix: str = KEY_PREFIX
    value_prefix: str = VALUE_PREFIX


# === Standalone functions for TrainingConfig ===

def create_training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Create TrainingConfig from CLI args, using config.py as defaults.
    
    This is the SINGLE point where all configuration gets resolved.
    After this, everything uses the frozen config object.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        TrainingConfig: Immutable configuration with all values resolved
    """
    # Start with defaults from dataclass, then override with CLI args
    base_config = TrainingConfig()
    
    # Model configuration (may be overridden by CLI args)
    model_type = getattr(args, 'model_type', None) or base_config.model_type
    if model_type == 'llama':
        model_name = 'meta-llama/Llama-3.2-3B'
        tokenizer_name = 'meta-llama/Llama-3.2-3B'
    elif model_type == 'gpt2':
        model_name = 'gpt2'
        tokenizer_name = 'gpt2'
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    
    # Initialize tokenizer for calculations
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate token counts
    prefix_tokens_per_key = len(tokenizer.encode(KEY_PREFIX, add_special_tokens=False))
    prefix_tokens_per_value = len(tokenizer.encode(VALUE_PREFIX, add_special_tokens=False))
    tokens_per_round = (
        prefix_tokens_per_key + TOKENS_PER_KEY +
        prefix_tokens_per_value + TOKENS_PER_VALUE
    )
    initial_prompt_tokens = len(tokenizer.encode(INITIAL_PROMPT, add_special_tokens=False))
    
    # Context window calculation
    model_config = AutoConfig.from_pretrained(model_name)
    if model_type == "llama":
        max_context_length = model_config.max_position_embeddings
    else:
        max_context_length = model_config.n_positions
        
    # Number of KV pairs
    available_context = max_context_length - initial_prompt_tokens
    num_kv_pairs = available_context // tokens_per_round
    num_kv_pairs = min(num_kv_pairs, 10)  # Cap for reasonable trajectory length
    
    # Apply CLI overrides with fallback to defaults
    return TrainingConfig(
        # Model configuration
        model_name=model_name,
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        device=base_config.device,
        
        # Training hyperparameters (CLI overrides or defaults)
        learning_rate=getattr(args, 'learning_rate', None) or base_config.learning_rate,
        num_episodes=getattr(args, 'episodes', None) or base_config.num_episodes,
        batch_size=getattr(args, 'batch_size', None) or base_config.batch_size,
        
        # RL parameters (CLI overrides or defaults)
        kl_penalty_coefficient=getattr(args, 'kl_penalty_coef', None) or base_config.kl_penalty_coefficient,
        ppo_clip_epsilon=getattr(args, 'ppo_clip_epsilon', None) or base_config.ppo_clip_epsilon,
        gamma=base_config.gamma,
        gae_lambda=base_config.gae_lambda,
        temperature=base_config.temperature,
        
        # Training behavior (CLI overrides or defaults)
        use_grpo_baseline=getattr(args, 'use_grpo_baseline', False) or base_config.use_grpo_baseline,
        use_ema_baseline=getattr(args, 'use_ema_baseline', False) or base_config.use_ema_baseline,
        ema_decay=getattr(args, 'ema_decay', None) or base_config.ema_decay,
        baseline_update_frequency=getattr(args, 'baseline_update_freq', None) or base_config.baseline_update_frequency,
        subtract_base_model_logprobs=getattr(args, 'subtract_base_logprobs', False) or base_config.subtract_base_model_logprobs,
        use_ppo=not getattr(args, 'vanilla_pg', False),  # PPO unless --vanilla-pg
        memory_efficient_lora=getattr(args, 'memory_efficient', False) or base_config.memory_efficient_lora,
        grpo_batching=getattr(args, 'grpo_batching', False) or base_config.grpo_batching,
        
        # Infrastructure
        checkpoint_interval=base_config.checkpoint_interval,
        log_interval=getattr(args, 'log_interval', None) or base_config.log_interval,
        enable_wandb=getattr(args, 'enable_wandb', False) or base_config.enable_wandb,
        
        # Computed token configuration
        prefix_tokens_per_key=prefix_tokens_per_key,
        prefix_tokens_per_value=prefix_tokens_per_value,
        tokens_per_round=tokens_per_round,
        initial_prompt_tokens=initial_prompt_tokens,
        num_kv_pairs=num_kv_pairs,
        max_context_length=max_context_length,
    )


def training_config_to_dict(config: TrainingConfig) -> Dict[str, Any]:
    """Convert TrainingConfig to dictionary for serialization/logging."""
    return {
        'model_name': config.model_name,
        'model_type': config.model_type,
        'learning_rate': config.learning_rate,
        'num_episodes': config.num_episodes,
        'batch_size': config.batch_size,
        'kl_penalty_coefficient': config.kl_penalty_coefficient,
        'ppo_clip_epsilon': config.ppo_clip_epsilon,
        'gamma': config.gamma,
        'gae_lambda': config.gae_lambda,
        'temperature': config.temperature,
        'use_grpo_baseline': config.use_grpo_baseline,
        'use_ema_baseline': config.use_ema_baseline,
        'ema_decay': config.ema_decay,
        'baseline_update_frequency': config.baseline_update_frequency,
        'subtract_base_model_logprobs': config.subtract_base_model_logprobs,
        'use_ppo': config.use_ppo,
        'memory_efficient_lora': config.memory_efficient_lora,
        'grpo_batching': config.grpo_batching,
        'enable_wandb': config.enable_wandb,
        'num_kv_pairs': config.num_kv_pairs,
        'max_context_length': config.max_context_length,
        'tokens_per_round': config.tokens_per_round,
    }


def log_training_config(config: TrainingConfig, logger) -> None:
    """Log the resolved configuration in a structured way."""
    logger.info("=== RESOLVED TRAINING CONFIGURATION ===")
    logger.info(f"Model: {config.model_name} ({config.model_type})")
    logger.info(f"Device: {config.device}")
    
    logger.info(f"Training: {config.num_episodes} episodes, batch_size={config.batch_size}, lr={config.learning_rate}")
    
    logger.info(f"RL: kl_penalty={config.kl_penalty_coefficient}, ppo_clip={config.ppo_clip_epsilon}")
    logger.info(f"    gamma={config.gamma}, gae_lambda={config.gae_lambda}, temperature={config.temperature}")
    
    if config.use_ema_baseline:
        logger.info(f"Baseline: EMA updates (decay={config.ema_decay:.3f}) - smooth")
    else:
        logger.info(f"Baseline: Hard updates every {config.baseline_update_frequency} episodes - may spike")
        
    if config.memory_efficient_lora:
        logger.info("Memory: Memory-efficient LoRA state management (60-90% reduction)")
    else:
        logger.info("Memory: Traditional model copying approach")
        
    logger.info(f"Context: {config.num_kv_pairs} KV pairs, {config.tokens_per_round} tokens/round")
    logger.info(f"         {config.max_context_length} max context, {config.initial_prompt_tokens} prompt tokens")
    logger.info("=" * 45)  
 