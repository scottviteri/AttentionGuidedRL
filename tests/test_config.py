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


def test_gpt2_model_type():
    """Test explicitly setting MODEL_TYPE to gpt2."""
    with patch.dict(os.environ, {'MODEL_TYPE': 'gpt2'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.MODEL_TYPE == "gpt2"
        assert src.config.MODEL_NAME == "gpt2"
        assert src.config.TOKENIZER_NAME == "gpt2"


def test_llama_model_type():
    """Test setting MODEL_TYPE to llama."""
    with patch.dict(os.environ, {'MODEL_TYPE': 'llama'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.MODEL_TYPE == "llama"
        assert src.config.MODEL_NAME == "meta-llama/Llama-3.2-3B"
        assert src.config.TOKENIZER_NAME == "meta-llama/Llama-3.2-3B"


def test_case_insensitive_model_type():
    """Test that MODEL_TYPE is case insensitive."""
    with patch.dict(os.environ, {'MODEL_TYPE': 'GPT2'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.MODEL_TYPE == "gpt2"
        assert src.config.MODEL_NAME == "gpt2"
    
    with patch.dict(os.environ, {'MODEL_TYPE': 'LLAMA'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        assert src.config.MODEL_TYPE == "llama"
        assert src.config.MODEL_NAME == "meta-llama/Llama-3.2-3B"


def test_invalid_model_type():
    """Test that invalid MODEL_TYPE raises ValueError."""
    with patch.dict(os.environ, {'MODEL_TYPE': 'invalid_model'}):
        import importlib
        import src.config
        
        with pytest.raises(ValueError, match="Invalid MODEL_TYPE: invalid_model"):
            importlib.reload(src.config)


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
    """Test that device configuration works correctly."""
    with patch.dict(os.environ, {'MODEL_TYPE': 'gpt2'}):
        import importlib
        import src.config
        importlib.reload(src.config)
        
        # Device should be set based on CUDA availability
        import torch
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert src.config.DEVICE == expected_device
        assert src.config.device == expected_device 