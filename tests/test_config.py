"""
Tests for configuration module.
"""

import pytest
from src.config import CONFIG
from src.config import TrainingConfig


def test_default_model_configuration():
    """Test that default model configuration is correct."""
    assert CONFIG.model_type == "gpt2"
    assert CONFIG.model_name == "gpt2"
    assert CONFIG.tokenizer_name == "gpt2"


def test_device_configuration():
    """Test that DEVICE is properly configured for CUDA if available, otherwise CPU."""
    import torch
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert CONFIG.device == expected, f"Expected DEVICE to be '{expected}' but got '{CONFIG.device}'."


def test_token_configuration_consistency():
    """Test that token configuration values are reasonable."""
    CONFIG.set_config(TrainingConfig(num_kv_pairs=10))
    assert isinstance(CONFIG.tokens_per_key, int) and CONFIG.tokens_per_key > 0
    assert isinstance(CONFIG.tokens_per_value, int) and CONFIG.tokens_per_value > 0
    assert isinstance(CONFIG.num_kv_pairs, int) and CONFIG.num_kv_pairs > 0
    
    # Ensure we have reasonable limits
    assert CONFIG.num_kv_pairs <= 15, "NUM_KV_PAIRS should be capped for reasonable trajectory length"
    assert CONFIG.tokens_per_key <= 50, "CONFIG.tokens_per_key should be reasonable for context window"
    assert CONFIG.tokens_per_value <= 50, "CONFIG.tokens_per_value should be reasonable for context window" 