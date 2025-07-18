"""
Configuration module for the Attention-Guided RL project.

Contains all the configuration constants used throughout the project.
"""

import torch
import argparse
from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoConfig, AutoTokenizer

# Only torch imports needed here


@dataclass(frozen=True)
class TrainingConfig:
    """
    Frozen configuration that resolves all values once from defaults + CLI overrides.
    
    This eliminates the confusing pattern of importing constants and reassigning them.
    Instead: Create one immutable config object with final resolved values.
    """
    # Core model configuration
    model_name: str = 'gpt2'
    model_type: str = 'gpt2'
    tokenizer_name: str = 'gpt2'
    device: str = "cuda"
    
    # Training hyperparameters
    learning_rate: float = 5e-4
    num_episodes: int = 10000
    batch_size: int = 4
    
    # RL-specific parameters
    kl_penalty_coefficient: float = 0.1
    ppo_clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    temperature: float = 1.0
    
    # Training behavior
    use_grpo_baseline: bool = True
    use_ema_baseline: bool = True
    ema_decay: float = 0.99
    baseline_update_frequency: int = 50
    subtract_base_model_logprobs: bool = False
    use_ppo: bool = True  # vs vanilla policy gradient
    memory_efficient_lora: bool = True  # Default to true per user request
    grpo_batching: bool = True  # GRPO-style batching
    
    # Training infrastructure
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    enable_wandb: bool = False
    
    # Token configuration (computed - will be filled by create_training_config_from_args)
    prefix_tokens_per_key: int = 0
    prefix_tokens_per_value: int = 0
    tokens_per_round: int = 0
    initial_prompt_tokens: int = 0
    num_kv_pairs: int = 0
    max_context_length: int = 0
    
    # Fixed constants (rarely changed)
    tokens_per_key: int = 10
    tokens_per_value: int = 10
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # Set to 0 per user request
    kv_every_n: int = 4  # Skip 4 chunks between each extraction for diversity
    initial_prompt: str = "Search for relevant information using learned vector queries."
    key_prefix: str = "Key: "
    value_prefix: str = "Value: "
    query_vec_token: str = 'Query'  # Hardcoded to 'Query'
    dtype: str = "bfloat16"  # Will be converted to torch.dtype
    gradient_clip_norm: float = 1.0


# === Standalone functions for TrainingConfig ===

def create_training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Create TrainingConfig from CLI args, using dataclass defaults.
    
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate token counts
    prefix_tokens_per_key = len(tokenizer.encode(base_config.key_prefix, add_special_tokens=False))
    prefix_tokens_per_value = len(tokenizer.encode(base_config.value_prefix, add_special_tokens=False))
    tokens_per_round = (
        prefix_tokens_per_key + base_config.tokens_per_key +
        prefix_tokens_per_value + base_config.tokens_per_value
    )
    initial_prompt_tokens = len(tokenizer.encode(base_config.initial_prompt, add_special_tokens=False))
    
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
        checkpoint_dir=base_config.checkpoint_dir,
        log_interval=getattr(args, 'log_interval', None) or base_config.log_interval,
        enable_wandb=getattr(args, 'enable_wandb', False) or base_config.enable_wandb,
        
        # Computed token configuration
        prefix_tokens_per_key=prefix_tokens_per_key,
        prefix_tokens_per_value=prefix_tokens_per_value,
        tokens_per_round=tokens_per_round,
        initial_prompt_tokens=initial_prompt_tokens,
        num_kv_pairs=num_kv_pairs,
        max_context_length=max_context_length,
        
        # Pass through fixed constants
        tokens_per_key=base_config.tokens_per_key,
        tokens_per_value=base_config.tokens_per_value,
        lora_rank=base_config.lora_rank,
        lora_alpha=base_config.lora_alpha,
        lora_dropout=base_config.lora_dropout,
        kv_every_n=base_config.kv_every_n,
        initial_prompt=base_config.initial_prompt,
        key_prefix=base_config.key_prefix,
        value_prefix=base_config.value_prefix,
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


# ============================================================================
# SINGLETON CONFIGURATION PATTERN
# 
# The CONFIG object acts as a module-level singleton that is set once at
# startup. This provides a clean migration path:
# 1. Replace constant imports with CONFIG imports
# 2. Access values as CONFIG.key_prefix instead of KEY_PREFIX
# 3. Tests can set their own CONFIG for isolation
# ============================================================================

_default_config = TrainingConfig()

class _ConfigProxy:
    """
    A proxy that provides attribute access to the current configuration.
    This allows CONFIG.attribute syntax while deferring to either the
    runtime config (if set) or default config.
    """
    def __init__(self):
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, '_runtime_config', None)
    
    def __getattr__(self, name):
        # Use the runtime config if available, otherwise default
        runtime_config = object.__getattribute__(self, '_runtime_config')
        config = runtime_config or _default_config
        return getattr(config, name)
    
    def set_config(self, config: TrainingConfig):
        """Set the runtime configuration."""
        object.__setattr__(self, '_runtime_config', config)
    
    def reset_to_default(self):
        """Reset to default configuration (useful for tests)."""
        object.__setattr__(self, '_runtime_config', None)
    
    def __repr__(self):
        runtime_config = object.__getattribute__(self, '_runtime_config')
        config = runtime_config or _default_config
        return f"CONFIG({config})"


# Create the singleton CONFIG object
CONFIG = _ConfigProxy() 