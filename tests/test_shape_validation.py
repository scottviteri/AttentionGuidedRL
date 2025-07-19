"""
Test shape validation in compute_similarity function.

This ensures that the GQA similarity computation fails fast when given
incorrectly shaped tensors instead of silently truncating or giving wrong results.
"""

import pytest
import torch


def test_compute_similarity_shape_validation(gpt2_model, gpt2_tokenizer, shape_validation_test):
    """Test that compute_similarity validates tensor shapes properly."""
    # Run the comprehensive shape validation test
    result = shape_validation_test(gpt2_model, gpt2_tokenizer)
    assert result is True, "Shape validation test failed"


def test_additional_edge_cases(gpt2_model):
    """Test additional edge cases for shape validation."""
    from src.embeddings import compute_similarity, get_attention_params
    
    # Get model parameters
    num_heads, num_groups, head_dim = get_attention_params(gpt2_model)
    correct_hidden_size = num_heads * head_dim
    
    batch_size = 1
    num_keys = 3
    device = torch.device("cpu")
    
    # Test with zero-sized tensors
    zero_query = torch.randn(0, correct_hidden_size, device=device)
    correct_keys = torch.randn(batch_size, num_keys, correct_hidden_size, device=device)
    
    try:
        compute_similarity(zero_query, correct_keys, num_heads, num_groups, head_dim)
        assert False, "Should fail with zero batch size"
    except (ValueError, RuntimeError):
        pass  # Expected to fail
    
    # Test with single dimension tensors
    flat_query = torch.randn(correct_hidden_size, device=device)
    
    try:
        compute_similarity(flat_query, correct_keys, num_heads, num_groups, head_dim)
        assert False, "Should fail with 1D query tensor"
    except ValueError as e:
        assert "query_embeddings must be 2D tensor" in str(e)
    
    # Test with 4D key embeddings
    fourd_keys = torch.randn(batch_size, num_keys, correct_hidden_size, 1, device=device)
    correct_query = torch.randn(batch_size, correct_hidden_size, device=device)
    
    try:
        compute_similarity(correct_query, fourd_keys, num_heads, num_groups, head_dim)
        assert False, "Should fail with 4D key tensor"
    except ValueError as e:
        assert "key_embeddings must be 3D tensor" in str(e)


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"]) 