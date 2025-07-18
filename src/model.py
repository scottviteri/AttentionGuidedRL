
"""
Model setup module for the Attention-Guided RL project.

Contains functions for loading language models and applying LoRA for efficient training.
"""

import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import logging

from src.config import CONFIG


def load_base_model():
    """
    Load the base language model.
    
    Returns:
        The loaded language model
    """
    # Configure quantization for reduced memory usage
    # Assumes CUDA is available as an explicit dependency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Load the model with appropriate configurations
    # Assumes CUDA is available as an explicit dependency
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name,
        device_map=CONFIG.device,
        torch_dtype=CONFIG.dtype,
        quantization_config=quantization_config,
    )
    
    # Disable gradients for the base model
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def get_target_modules():
    """
    Get the target modules for LoRA based on the model type.
    
    Returns:
        List of target module names
    """
    if CONFIG.model_type == "llama":
        # For Llama models, target the attention projection layers
        return ["q_proj", "k_proj", "v_proj"]
    elif CONFIG.model_type == "gpt2":
        # For GPT-2, target the attention layers
        return ["c_attn"]
    else:
        raise ValueError(f"Unsupported model type: {CONFIG.model_type}")


def apply_lora_adapter(model):
    """
    Apply LoRA adapter to the model for parameter-efficient fine-tuning.
    
    Args:
        model: The base language model
        
    Returns:
        The model with LoRA adapter applied
    """
    # Create a deep copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=CONFIG.lora_rank,
        lora_alpha=CONFIG.lora_alpha,
        target_modules=get_target_modules(),
        lora_dropout=CONFIG.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",  # Use Gaussian initialization for more randomness
    )
    
    # Apply LoRA adapter to the copy
    model_copy = get_peft_model(model_copy, lora_config)
    
    # Note: LoRA weights are automatically initialized by PEFT's "gaussian" setting
    # No manual initialization needed - PEFT handles this correctly
    
    return model_copy


def save_model_adapter(model, path):
    """
    Save the model adapter state.
    
    Args:
        model: The model with LoRA adapter
        path: Path to save the state
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_adapter(model, path):
    """
    Load the model adapter state.
    
    Args:
        model: The model with LoRA adapter
        path: Path to load the state from
        
    Returns:
        The model with loaded adapter state
    """
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def save_lora_state(model):
    """
    Extract and return only the LoRA adapter state from a model.
    
    Args:
        model: The model with LoRA adapter
        
    Returns:
        Dict containing only LoRA adapter parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:  # Only LoRA parameters
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state(model, lora_state):
    """
    Load LoRA adapter state into a model.
    
    Args:
        model: The model with LoRA adapter to update
        lora_state: Dict containing LoRA parameters
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data.copy_(lora_state[name])


def update_lora_ema(target_model: torch.nn.Module, source_model: torch.nn.Module, decay: float = 0.95) -> None:
    """
    Update target_model LoRA parameters using exponential moving average from source_model.
    Memory-efficient: only updates LoRA parameters, not base model weights.
    
    Args:
        target_model: Model to update (old_model/baseline) 
        source_model: Model to copy from (current adapter_model)
        decay: EMA decay factor (0.9-0.99 typical, higher = smoother)
               target = decay * target + (1 - decay) * source
    """
    import logging
    
    updated_count = 0
    
    with torch.no_grad():
        for (target_name, target_param), (source_name, source_param) in zip(
            target_model.named_parameters(), source_model.named_parameters()
        ):
            # Only update LoRA parameters
            if 'lora_' in target_name and 'lora_' in source_name:
                if target_param.dtype.is_floating_point and source_param.dtype.is_floating_point:
                    target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)
                    updated_count += 1
    
    logging.debug(f"LoRA EMA update: {updated_count} LoRA parameters updated")


def create_model_copy(model):
    """
    Create a deep copy of the model with adapter parameters.
    
    Args:
        model: The model with LoRA adapter
        
    Returns:
        A copy of the model with the same parameters
    """
    return copy.deepcopy(model)


def update_model_ema(target_model: torch.nn.Module, source_model: torch.nn.Module, decay: float = 0.95) -> None:
    """
    Update target_model parameters using exponential moving average from source_model.
    
    This provides smooth parameter updates instead of hard replacements, reducing training spikiness.
    Memory-efficient: only updates LoRA parameters, not quantized base model weights.
    
    Args:
        target_model: Model to update (old_model/baseline)
        source_model: Model to copy from (current adapter_model)
        decay: EMA decay factor (0.9-0.99 typical, higher = smoother)
               target = decay * target + (1 - decay) * source
    """
    # Use the memory-efficient LoRA-only EMA update
    update_lora_ema(target_model, source_model, decay)


def setup_model_and_tokenizer():
    """
    Set up the model and tokenizer.
    
    Returns:
        Tuple of (base_model, adapter_model, tokenizer)
    """
    # Load the base model
    base_model = load_base_model()
    
    # Apply LoRA adapter
    adapter_model = apply_lora_adapter(base_model)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    
    # Verify the token exists in the vocabulary
    token_ids = tokenizer.encode(CONFIG.query_vec_token, add_special_tokens=False)
    if len(token_ids) != 1:
        logging.warning(f"Query token '{CONFIG.query_vec_token}' tokenizes to {len(token_ids)} tokens: {token_ids}")
        logging.warning("This may affect embedding extraction. Consider using a single-token word.")
    else:
        logging.info(f"Query token '{CONFIG.query_vec_token}' has token ID: {token_ids[0]}")
        
    return base_model, adapter_model, tokenizer


def get_checkpoint_path(episode):
    """
    Get the checkpoint path for a specific episode.
    
    Args:
        episode: The episode number or "latest"
        
    Returns:
        The checkpoint path
    """
    if episode == "latest":
        return os.path.join(CONFIG.checkpoint_dir, "model_latest.pt")
    else:
        return os.path.join(CONFIG.checkpoint_dir, f"model_episode_{episode}.pt")


def save_checkpoint(model, episode):
    """
    Save a checkpoint of the model.
    
    Args:
        model: The model to save
        episode: The current episode number
    """
    path = get_checkpoint_path(episode)
    save_model_adapter(model, path)


def load_checkpoint(model, episode):
    """
    Load a checkpoint of the model.
    
    Args:
        model: The model to load into
        episode: The episode number to load or "latest"
        
    Returns:
        The loaded model
    """
    path = get_checkpoint_path(episode)
    if os.path.exists(path):
        load_model_adapter(model, path)
        return True
    return False 