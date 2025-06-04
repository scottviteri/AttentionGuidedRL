"""
Tests for vector query functionality.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.config import QUERY_VEC_TOKEN, USE_VECTOR_QUERIES
from src.model import setup_model_and_tokenizer
from src.data import get_tokenizer


def test_query_vec_token_added():
    """Test that the QUERY_VEC_TOKEN is properly added to the tokenizer."""
    tokenizer = get_tokenizer()
    
    # Check that the token is in the tokenizer
    assert QUERY_VEC_TOKEN in tokenizer.get_added_vocab()
    
    # Check that we can encode and decode the token
    token_ids = tokenizer.encode(QUERY_VEC_TOKEN, add_special_tokens=False)
    assert len(token_ids) == 1  # Should be a single token
    
    # Check that we can decode it back
    decoded = tokenizer.decode(token_ids)
    assert QUERY_VEC_TOKEN in decoded


def test_model_setup_with_query_vec_token(gpt2_model):
    """Test that model setup correctly handles the new token."""
    with patch('src.model.load_base_model') as mock_load:
        # Use the real GPT2 model from the fixture
        mock_load.return_value = gpt2_model
        
        # Setup model and tokenizer
        base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
        
        # Check that the token is in the tokenizer
        assert QUERY_VEC_TOKEN in tokenizer.get_added_vocab()
        
        # Check that model embeddings were resized
        # The embedding size should match the tokenizer vocab size
        assert base_model.get_input_embeddings().weight.shape[0] == len(tokenizer)
        assert adapter_model.get_input_embeddings().weight.shape[0] == len(tokenizer)


def test_backwards_compatibility():
    """Test that the USE_VECTOR_QUERIES flag defaults to False."""
    assert USE_VECTOR_QUERIES is False


def test_existing_functionality_unchanged(gpt2_model, gpt2_tokenizer):
    """Test that existing query generation still works with the new token added."""
    from src.training import generate_query
    
    # Mock the tokenizer to include our special token
    with patch('src.data.get_tokenizer') as mock_get_tokenizer:
        # Add the special token to the real tokenizer
        special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
        gpt2_tokenizer.add_special_tokens(special_tokens_dict)
        mock_get_tokenizer.return_value = gpt2_tokenizer
        
        # Test that generate_query still works
        context_text = ["This is a test context"]
        
        with patch('src.config.MODEL_TYPE', 'gpt2'):
            # Mock the model generate method
            mock_output = torch.randint(0, 1000, (1, 10))
            gpt2_model.generate = MagicMock(return_value=mock_output)
            
            # Generate query
            query_tokens = generate_query(gpt2_model, gpt2_tokenizer, context_text)
            
            # Verify it returns tokens
            assert query_tokens is not None
            assert isinstance(query_tokens, torch.Tensor) 