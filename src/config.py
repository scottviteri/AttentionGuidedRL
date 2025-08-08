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
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate temperature is positive
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        # Validate other critical parameters
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")
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
    gamma: float = 0.99
    temperature: float = 1.0
    reward_aggregation: str = "average"  # "average" or "discounted"
    differentiable_rewards: bool = True   # Enable chain-rule reward gradients
    
    # Training behavior
    subtract_base_model_logprobs: bool = False
    
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
        gamma=base_config.gamma,
        temperature=base_config.temperature,
        reward_aggregation=getattr(args, 'reward_aggregation', None) or base_config.reward_aggregation,
        differentiable_rewards=getattr(args, 'differentiable_rewards', False) or base_config.differentiable_rewards,
        
        # Training behavior (CLI overrides or defaults)
        subtract_base_model_logprobs=getattr(args, 'subtract_base_logprobs', False) or base_config.subtract_base_model_logprobs,
        
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
        'gamma': config.gamma,
        'temperature': config.temperature,
        'subtract_base_model_logprobs': config.subtract_base_model_logprobs,
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
    
    logger.info(f"RL: gamma={config.gamma}, temperature={config.temperature}")
    
    logger.info("Baseline: None (trajectory average rewards)")
        
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