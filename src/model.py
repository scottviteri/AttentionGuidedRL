"""
Model setup module for the Attention-Guided RL project.

Contains functions for loading language models and applying LoRA for efficient training.
"""

import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from src.config import (
    MODEL_NAME,
    TOKENIZER_NAME,
    MODEL_TYPE,
    DEVICE,
    DTYPE,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    CHECKPOINT_DIR,
)


def load_base_model():
    """
    Load the base language model.
    
    Returns:
        The loaded language model
    """
    # Configure quantization for reduced memory usage
    quantization_config = None
    
    if torch.cuda.is_available():
        # Use 8-bit quantization if CUDA is available
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    # Load the model with appropriate configurations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE if torch.cuda.is_available() else None,
        torch_dtype=DTYPE,
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
    if MODEL_TYPE == "llama":
        # For Llama models, target the attention projection layers
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif MODEL_TYPE == "gpt2":
        # For GPT-2, target the attention layers
        return ["c_attn", "c_proj"]
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def apply_lora_adapter(model):
    """
    Apply LoRA adapter to the model for parameter-efficient fine-tuning.
    
    Args:
        model: The base language model
        
    Returns:
        The model with LoRA adapter applied
    """
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=get_target_modules(),
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA adapter
    model = get_peft_model(model, lora_config)
    
    return model


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
    """
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def create_model_copy(model):
    """
    Create a deep copy of the model's adapter parameters.
    
    Args:
        model: The model with LoRA adapter
        
    Returns:
        A copy of the model's state
    """
    return copy.deepcopy(model.state_dict())


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
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    return base_model, adapter_model, tokenizer


def get_checkpoint_path(episode):
    """
    Get the checkpoint path for a specific episode.
    
    Args:
        episode: The episode number
        
    Returns:
        The checkpoint path
    """
    return os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pt")


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
        episode: The episode number to load
        
    Returns:
        The loaded model
    """
    path = get_checkpoint_path(episode)
    if os.path.exists(path):
        load_model_adapter(model, path)
        return True
    return False 