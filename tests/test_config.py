"""
Tests for configuration module with manual model selection.
"""

import os
import pytest
from unittest.mock import patch


def test_default_model_type():
    """Test that default model type is GPT-2."""
    # Clear any existing MODEL_TYPE environment variable
    with patch.dict(os.environ, {}, clear=True):
        # Remove MODEL_TYPE if it exists
        if 'MODEL_TYPE' in os.environ:
            del os.environ['MODEL_TYPE']
        
        # Re-import config to get fresh values
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.MODEL_TYPE == "gpt2"
        assert src.config.MODEL_NAME == "gpt2"
        assert src.config.TOKENIZER_NAME == "gpt2"


# The default configuration now ignores environment variables.
# Tests that relied on env vars have been removed.


def test_config_consistency():
    """Test that configuration is consistent across different model types."""
    # Test that some core configs remain the same regardless of model
    with patch.dict(os.environ, {'MODEL_TYPE': 'gpt2'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        gpt2_tokens_per_key = src.config.TOKENS_PER_KEY
        gpt2_tokens_per_value = src.config.TOKENS_PER_VALUE
        gpt2_learning_rate = src.config.LEARNING_RATE
    
    with patch.dict(os.environ, {'MODEL_TYPE': 'llama'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        llama_tokens_per_key = src.config.TOKENS_PER_KEY
        llama_tokens_per_value = src.config.TOKENS_PER_VALUE
        llama_learning_rate = src.config.LEARNING_RATE
    
    # These should be the same regardless of model
    assert gpt2_tokens_per_key == llama_tokens_per_key
    assert gpt2_tokens_per_value == llama_tokens_per_value
    assert gpt2_learning_rate == llama_learning_rate


def test_device_configuration():
    """Test that the device is correctly configured."""
    from src.config import DEVICE
    
    # Check that device is defined
    assert DEVICE is not None
    
    # Check that it's a valid device string
    assert isinstance(DEVICE, str)
    assert DEVICE in ['cuda', 'cpu', 'mps'] 