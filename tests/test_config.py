"""
Tests for configuration module.
"""

import pytest
from src.config import MODEL_TYPE, MODEL_NAME, TOKENIZER_NAME, DEVICE, NUM_KV_PAIRS, TOKENS_PER_KEY, TOKENS_PER_VALUE


def test_default_model_configuration():
    """Test that default model configuration is correct."""
    assert MODEL_TYPE == "gpt2"
    assert MODEL_NAME == "gpt2"
    assert TOKENIZER_NAME == "gpt2"


def test_device_configuration():
    """Test that DEVICE is properly configured for CUDA."""
    # CUDA is now a hard requirement, so DEVICE should be 'cuda'
    assert DEVICE == 'cuda', f"Expected DEVICE to be 'cuda' but got '{DEVICE}'. This project requires CUDA."


def test_token_configuration_consistency():
    """Test that token configuration values are reasonable."""
    assert isinstance(TOKENS_PER_KEY, int) and TOKENS_PER_KEY > 0
    assert isinstance(TOKENS_PER_VALUE, int) and TOKENS_PER_VALUE > 0
    assert isinstance(NUM_KV_PAIRS, int) and NUM_KV_PAIRS > 0
    
    # Ensure we have reasonable limits
    assert NUM_KV_PAIRS <= 15, "NUM_KV_PAIRS should be capped for reasonable trajectory length"
    assert TOKENS_PER_KEY <= 50, "TOKENS_PER_KEY should be reasonable for context window"
    assert TOKENS_PER_VALUE <= 50, "TOKENS_PER_VALUE should be reasonable for context window" 