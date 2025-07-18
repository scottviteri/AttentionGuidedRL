# tests/conftest.py
import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
from unittest.mock import MagicMock

# Use CPU for tests to avoid CUDA compatibility issues
device = torch.device("cpu")


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    
    # Set up the tokenizer to return realistic tokens
    # We'll use a fixed vocab size for consistency
    tokenizer.vocab_size = 50257  # GPT-2 vocab size
    
    # Mock the __call__ method to return input_ids
    def mock_call(*args, **kwargs):
        if isinstance(args[0], list):
            # Handle batch of strings
            batch_size = len(args[0])
            # Return a mock object with input_ids
            result = MagicMock()
            result.input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, 10))
            return result
        else:
            # Handle single string
            result = MagicMock()
            result.input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
            return result
    
    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    
    # Mock batch_decode to return list of strings
    tokenizer.batch_decode.return_value = ["mock decoded text"] * 10
    
    # Mock individual token properties
    tokenizer.eos_token_id = 50256
    tokenizer.pad_token_id = 50256
    tokenizer.unk_token_id = 50256
    
    return tokenizer


@pytest.fixture  
def mock_gpt2_model():
    """Create a minimal mock GPT-2 model for testing."""
    model = MagicMock()
    
    # Mock the config
    model.config = MagicMock()
    model.config.n_embd = 768  # GPT-2 embedding dimension
    model.config.vocab_size = 50257
    model.config.n_layer = 12
    model.config.n_head = 12
    
    # Mock device properties - assume CUDA is available
    device = torch.device("cpu")
    model.device = device
    
    def mock_parameters():
        # Return a list instead of generator to make it pickleable
        param = torch.randn(10, 10, device=device, requires_grad=True)
        return [param]
    
    model.parameters.return_value = mock_parameters()
    
    def mock_next_parameters():
        param = torch.randn(10, 10, device=device, requires_grad=True) 
        return param
        
    # Mock the __next__ method for next(model.parameters())
    model.__next__ = mock_next_parameters
    
    # Mock the forward pass
    def mock_forward(*args, **kwargs):
        batch_size = args[0].size(0) if len(args) > 0 else 1
        seq_len = args[0].size(1) if len(args) > 0 else 10
        
        # Create mock output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(batch_size, seq_len, model.config.vocab_size, device=device)
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, model.config.n_embd, device=device)
        
        return mock_output
    
    model.forward = mock_forward
    model.__call__ = mock_forward
    
    # Mock other methods
    model.eval.return_value = model
    model.train.return_value = model
    model.to.return_value = model
    model.cuda.return_value = model
    
    return model


# Add the expected fixture names that map to the mock fixtures
@pytest.fixture
def gpt2_model(mock_gpt2_model):
    """Fixture that provides a GPT-2 model for testing."""
    return mock_gpt2_model


@pytest.fixture 
def gpt2_tokenizer(mock_tokenizer):
    """Fixture that provides a GPT-2 tokenizer for testing."""
    return mock_tokenizer