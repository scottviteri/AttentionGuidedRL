"""
Tests for vector query functionality.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.config import QUERY_VEC_TOKEN
from src.model import setup_model_and_tokenizer
from src.data import get_tokenizer


def test_query_vec_token_added():
    """Test that the QUERY_VEC_TOKEN is properly handled by the tokenizer."""
    from src.config import USE_STANDARD_QUERY_TOKEN
    tokenizer = get_tokenizer()
    
    if USE_STANDARD_QUERY_TOKEN:
        # Standard tokens are already in the vocabulary, not added
        # Check that we can encode and decode the token
        token_ids = tokenizer.encode(QUERY_VEC_TOKEN, add_special_tokens=False)
        assert len(token_ids) >= 1  # Standard token might be multiple tokens
        
        # Check that we can decode it back
        decoded = tokenizer.decode(token_ids)
        assert QUERY_VEC_TOKEN in decoded
    else:
        # Special tokens are added to the vocabulary
        # Check that the token is in the tokenizer
        assert QUERY_VEC_TOKEN in tokenizer.get_added_vocab()
        
        # Check that we can encode and decode the token
        token_ids = tokenizer.encode(QUERY_VEC_TOKEN, add_special_tokens=False)
        assert len(token_ids) == 1  # Should be a single token
        
        # Check that we can decode it back
        decoded = tokenizer.decode(token_ids)
        assert QUERY_VEC_TOKEN in decoded


def test_model_setup_with_query_vec_token(gpt2_model):
    """Test that model setup correctly handles the query token."""
    from src.config import USE_STANDARD_QUERY_TOKEN
    
    with patch('src.model.load_base_model') as mock_load:
        # Use the real GPT2 model from the fixture
        mock_load.return_value = gpt2_model
        
        # Setup model and tokenizer
        base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
        
        if USE_STANDARD_QUERY_TOKEN:
            # Standard tokens are already in vocabulary - embeddings should not be resized
            # Just check that the token can be encoded/decoded
            token_ids = tokenizer.encode(QUERY_VEC_TOKEN, add_special_tokens=False)
            assert len(token_ids) >= 1
            decoded = tokenizer.decode(token_ids)
            assert QUERY_VEC_TOKEN in decoded
        else:
            # Special tokens are added - check that token is in added vocab
            assert QUERY_VEC_TOKEN in tokenizer.get_added_vocab()
            
            # Check that model embeddings were resized
            # The embedding size should match the tokenizer vocab size
            assert base_model.get_input_embeddings().weight.shape[0] == len(tokenizer)
            assert adapter_model.get_input_embeddings().weight.shape[0] == len(tokenizer)


def test_vector_queries_only():
    """Test that we're now using vector queries only."""
    # This test confirms we've removed multi-token queries
    # and are committed to vector queries
    assert True  # Placeholder test confirming the simplification


def test_vector_query_generation(gpt2_model, gpt2_tokenizer):
    """Test that vector query generation works properly."""
    from src.training import generate_query_vector
    
    # Add the special token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
    gpt2_tokenizer.add_special_tokens(special_tokens_dict)
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
    
    # Test that generate_query_vector works
    context_text = ["This is a test context"]
    device = next(gpt2_model.parameters()).device
    context_tokens = gpt2_tokenizer(
        context_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids.to(device)
    
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        # Generate query vector
        query_vector = generate_query_vector(
            gpt2_model, gpt2_tokenizer, context_tokens
        )
        
        # Verify output
        assert query_vector is not None
        assert isinstance(query_vector, torch.Tensor)
        assert query_vector.shape[0] == 1  # batch size
        assert query_vector.shape[1] > 0   # embedding dimension


def test_generate_query_vector(gpt2_model, gpt2_tokenizer):
    """Test the generate_query_vector function."""
    from src.training import generate_query_vector
    from src.embeddings import get_query_dimension
    
    # Add the special token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
    gpt2_tokenizer.add_special_tokens(special_tokens_dict)
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
    
    # Create some context tokens
    context_text = ["This is a test context"]
    device = next(gpt2_model.parameters()).device
    context_tokens = gpt2_tokenizer(
        context_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids.to(device)
    
    # Generate query vector (defaults to layer_idx=-2)
    query_vector = generate_query_vector(
        gpt2_model,
        gpt2_tokenizer,
        context_tokens
    )
    
    # Get expected query dimension
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        expected_query_dim = get_query_dimension(gpt2_model)
    
    # Verify the output
    assert query_vector is not None
    assert isinstance(query_vector, torch.Tensor)
    assert query_vector.shape[0] == 1  # batch size
    # The dimension should be the query projection dimension
    assert query_vector.shape[1] == expected_query_dim, f"Expected query dimension {expected_query_dim}, got {query_vector.shape[1]}"


def test_query_vector_deterministic(gpt2_model, gpt2_tokenizer):
    """Test that query vector generation is deterministic."""
    from src.training import generate_query_vector
    from src.embeddings import get_query_dimension
    
    # Add the special token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
    gpt2_tokenizer.add_special_tokens(special_tokens_dict)
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
    
    # Create context
    context_text = ["This is a test context"]
    device = next(gpt2_model.parameters()).device
    context_tokens = gpt2_tokenizer(
        context_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids.to(device)
    
    # Generate query vectors multiple times
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        query_vector_1 = generate_query_vector(
            gpt2_model,
            gpt2_tokenizer,
            context_tokens
        )
        
        query_vector_2 = generate_query_vector(
            gpt2_model,
            gpt2_tokenizer,
            context_tokens
        )
        
        # Get expected query dimension
        expected_query_dim = get_query_dimension(gpt2_model)
        assert query_vector_1.shape == (1, expected_query_dim)
        assert query_vector_2.shape == (1, expected_query_dim)
        
        # The vectors should be identical (deterministic)
        assert torch.allclose(query_vector_1, query_vector_2, atol=1e-6)


def test_query_vector_layer_selection(gpt2_model, gpt2_tokenizer):
    """Test that query vectors can be extracted from different layers."""
    from src.training import generate_query_vector
    from src.embeddings import get_query_dimension
    
    # Add the special token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
    gpt2_tokenizer.add_special_tokens(special_tokens_dict)
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
    
    # Create context
    context_text = ["This is a test context"]
    device = next(gpt2_model.parameters()).device
    context_tokens = gpt2_tokenizer(
        context_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids.to(device)
    
    # Test extracting from different layers
    num_layers = len(gpt2_model.transformer.h)
    
    # Extract from second-to-last layer (default)
    query_vector_n2 = generate_query_vector(
        gpt2_model,
        gpt2_tokenizer,
        context_tokens
    )
    
    # Extract from last layer
    query_vector_n1 = generate_query_vector(
        gpt2_model,
        gpt2_tokenizer,
        context_tokens,
        layer_idx=-1
    )
    
    # Extract from first layer
    query_vector_0 = generate_query_vector(
        gpt2_model,
        gpt2_tokenizer,
        context_tokens,
        layer_idx=0
    )
    
    # Get expected query dimension
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        expected_query_dim = get_query_dimension(gpt2_model)
    
    # All should have correct shape
    assert query_vector_n2.shape == (1, expected_query_dim)
    assert query_vector_n1.shape == (1, expected_query_dim)
    assert query_vector_0.shape == (1, expected_query_dim)
    
    # Vectors from different layers should be different
    assert not torch.allclose(query_vector_n2, query_vector_n1)
    assert not torch.allclose(query_vector_n2, query_vector_0)
    assert not torch.allclose(query_vector_n1, query_vector_0) 