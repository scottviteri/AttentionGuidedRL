"""
Tests for the embeddings module.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import numpy as np
from unittest.mock import MagicMock, patch

from src.embeddings import (
    register_embedding_hook,
    extract_embeddings,
    get_attention_params,
    compute_similarity,
    sample_key_value,
)
from src.config import MODEL_TYPE
from src.model import apply_lora_adapter


# Remove all the Mock classes - they're not needed since we have real model fixtures

def test_register_embedding_hook_llama(gpt2_model):
    """Test registering embedding hook for Llama model."""
    # Skip this test since we don't have a real Llama model in the test fixtures
    pytest.skip("This test requires a real Llama model, not GPT2")
    
    # Import here to avoid circular imports
    from src.embeddings import register_embedding_hook
    
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Use real model instead of mock
        embeddings_dict, hook_remover = register_embedding_hook(gpt2_model, embed_type="query")
    
    # Verify hook was registered
    assert "embeddings" in embeddings_dict
    assert embeddings_dict["embeddings"] is None  # Not populated until forward pass
    assert callable(hook_remover)
    
    # Clean up
    hook_remover()


def test_extract_embeddings_integration(gpt2_model):
    """Test extracting embeddings with a real model."""
    from src.embeddings import register_embedding_hook, extract_embeddings
    
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        # Register hook on real model
        embeddings_dict, hook_remover = register_embedding_hook(gpt2_model, embed_type="query")
        
        # Create token input
        batch_size = 2
        seq_len = 10
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Extract embeddings (this will run the model)
        embeddings = extract_embeddings(gpt2_model, tokens, embeddings_dict)
        
        # Verify shape
        assert embeddings.shape == (batch_size, gpt2_model.config.n_embd)
        
        # Clean up
        hook_remover()


def test_compute_similarity_with_real_model(gpt2_model):
    """Test similarity computation with real model."""
    from src.embeddings import compute_similarity
    
    batch_size = 2
    num_keys = 5
    hidden_dim = gpt2_model.config.n_embd
    
    # Create query and key embeddings
    query_embeddings = torch.randn(batch_size, hidden_dim)
    key_embeddings = torch.randn(batch_size, num_keys, hidden_dim)
    
    # Compute similarity
    similarity = compute_similarity(query_embeddings, key_embeddings, gpt2_model)
    
    # Verify shape and properties
    assert similarity.shape == (batch_size, num_keys)
    # Check they are probabilities (sum to 1)
    for b in range(batch_size):
        assert torch.isclose(similarity[b].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(similarity[b] >= 0) and torch.all(similarity[b] <= 1)


def test_sample_key_value_deterministic():
    """Test key value sampling without mocking the distribution."""
    from src.embeddings import sample_key_value
    
    batch_size = 3
    num_keys = 5
    
    # Create probability distribution
    probs = torch.softmax(torch.randn(batch_size, num_keys), dim=1)
    
    # Create available keys
    available_keys = [
        [0, 1, 2],      # Batch 0
        [1, 3, 4],      # Batch 1  
        [0, 2, 4]       # Batch 2
    ]
    
    # Sample multiple times to test randomness
    samples = []
    for _ in range(10):
        sampled_indices, sampled_probs = sample_key_value(probs, available_keys, batch_size)
        samples.append(sampled_indices)
        
        # Verify sampled indices are valid
        for b in range(batch_size):
            assert sampled_indices[b] in available_keys[b]
            assert torch.isclose(sampled_probs[b], probs[b, sampled_indices[b]])
    
    # Check that we get some variety in sampling (not always the same)
    # This might fail occasionally if we get unlucky, but should pass most of the time
    unique_samples = [len(set(s[b] for s in samples)) for b in range(batch_size)]
    assert any(u > 1 for u in unique_samples), "Sampling appears to be deterministic"


def test_compute_similarity_attention_mechanism():
    """Test that compute_similarity correctly implements attention mechanism.
    
    This test verifies:
    1. In MHA, each query head attends to its corresponding key head (1:1 mapping)
    2. In GQA, multiple query heads attend to the same key head (N:1 mapping)
    3. Softmax is applied per-head before averaging, as in real transformer models
    """
    pytest.skip("This test requires specific model architectures with GQA support")


def test_compute_similarity_with_high_temperature():
    """Test compute_similarity with different temperature values."""
    pytest.skip("This test requires specific model architectures with GQA support")


def test_compute_similarity_batch_behavior():
    """Test compute_similarity handles batch dimension correctly."""
    pytest.skip("This test requires specific model architectures with GQA support")


def test_compute_similarity_real_gpt2(gpt2_model, gpt2_tokenizer):
    """Test similarity computation using a real GPT-2 model"""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        # Get real model parameters
        batch_size = 2
        num_keys = 3
        
        # Get the real hidden size from the model
        hidden_size = gpt2_model.config.n_embd
        
        # Create query and key embeddings with correct dimensions for GPT-2
        query_embeddings = torch.randn(batch_size, hidden_size, device=gpt2_model.device)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size, device=gpt2_model.device)
        
        # Get real attention parameters
        num_heads, num_groups, head_dim = get_attention_params(gpt2_model)
        
        # Compute similarity
        similarity = compute_similarity(query_embeddings, key_embeddings, gpt2_model)
        
        # Check shape and properties
        assert similarity.shape == (batch_size, num_keys)
        assert torch.allclose(torch.sum(similarity, dim=1), torch.ones(batch_size, device=gpt2_model.device))
        
        # Test with different temperature
        similarity_high_temp = compute_similarity(
            query_embeddings, key_embeddings, gpt2_model, temperature=5.0
        )
        
        # Higher temperature should result in more uniform distribution
        assert torch.std(similarity_high_temp) < torch.std(similarity) 


def test_embedding_hook_registration_real_gpt2(gpt2_model):
    """Test embedding hook registration with a real GPT-2 model."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        embed_dict, remove_hook = register_embedding_hook(gpt2_model, embed_type="query")
        assert "embeddings" in embed_dict
        assert callable(remove_hook)
        
        # Clean up
        remove_hook()
        
        # Test key embeddings hook too
        embed_dict, remove_hook = register_embedding_hook(gpt2_model, embed_type="key")
        assert "embeddings" in embed_dict
        assert callable(remove_hook)
        
        # Clean up
        remove_hook()


def test_extract_embeddings_real_gpt2(gpt2_model, gpt2_tokenizer):
    """Test embedding extraction with a real GPT-2 model."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        # Register embedding hook
        embeddings_dict, hook_remover = register_embedding_hook(gpt2_model)
        
        try:
            # Create a short input
            batch_size = 2
            input_text = ["Hello world", "Testing GPT-2 embeddings"]
            
            # Tokenize input - explicitly set padding
            encoded_input = gpt2_tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=20
            )
            input_ids = encoded_input["input_ids"].to(gpt2_model.device)
            
            # Extract embeddings
            result = extract_embeddings(gpt2_model, input_ids, embeddings_dict)
            
            # Check shape
            hidden_size = gpt2_model.config.n_embd
            assert result.shape == (batch_size, hidden_size)
            
            # Verify embeddings are on the correct device
            assert result.device == gpt2_model.device
            
            # Verify embeddings have reasonable values
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
        finally:
            # Clean up
            hook_remover()


def test_get_attention_params_real_gpt2(gpt2_model):
    """Test getting attention parameters from a real GPT-2 model."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        num_heads, num_groups, head_dim = get_attention_params(gpt2_model)
        
        # Verify parameters match the model's config
        assert num_heads == gpt2_model.config.n_head
        assert num_groups == gpt2_model.config.n_head  # For GPT-2, num_groups == num_heads (no GQA)
        assert head_dim == gpt2_model.config.n_embd // gpt2_model.config.n_head


def test_batch_processing_real_gpt2(gpt2_model):
    """Test batch processing consistency with a real GPT-2 model."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        batch_size = 3
        num_keys = 4
        hidden_size = gpt2_model.config.n_embd
        device = gpt2_model.device
        
        # Create random query and key embeddings
        query_embeddings = torch.randn(batch_size, hidden_size, device=device)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size, device=device)
        
        # Process each batch item separately
        individual_results = []
        for b in range(batch_size):
            single_query = query_embeddings[b:b+1]  # Keep batch dim
            single_key = key_embeddings[b:b+1]      # Keep batch dim
            result = compute_similarity(single_query, single_key, gpt2_model)
            individual_results.append(result)
        
        # Process all batch items together
        batched_result = compute_similarity(query_embeddings, key_embeddings, gpt2_model)
        
        # Verify that processing individually or in batch gives same results
        for b in range(batch_size):
            assert torch.allclose(batched_result[b], individual_results[b].squeeze(0), rtol=1e-4)


def test_temperature_scaling_real_gpt2(gpt2_model):
    """Test temperature scaling effect on attention with a real GPT-2 model."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        batch_size = 2
        num_keys = 5
        hidden_size = gpt2_model.config.n_embd
        device = gpt2_model.device
        
        # Create query and key embeddings with controlled patterns
        # Make one key match the query much better than others
        query_embeddings = torch.randn(batch_size, hidden_size, device=device)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size, device=device)
        
        # Compute similarity with different temperatures
        similarity_low_temp = compute_similarity(
            query_embeddings, key_embeddings, gpt2_model, temperature=0.1
        )
        similarity_med_temp = compute_similarity(
            query_embeddings, key_embeddings, gpt2_model, temperature=1.0
        )
        similarity_high_temp = compute_similarity(
            query_embeddings, key_embeddings, gpt2_model, temperature=10.0
        )
        
        # Verify all outputs are valid probability distributions
        for similarity in [similarity_low_temp, similarity_med_temp, similarity_high_temp]:
            assert torch.allclose(torch.sum(similarity, dim=1), torch.ones(batch_size, device=device))
            assert torch.all(similarity >= 0) and torch.all(similarity <= 1)
        
        # Verify temperature effects: higher temp = more uniform
        # Calculate standard deviation of the distributions
        std_low = torch.std(similarity_low_temp, dim=1).mean()
        std_med = torch.std(similarity_med_temp, dim=1).mean()
        std_high = torch.std(similarity_high_temp, dim=1).mean()
        
        # Higher temperature should lead to lower standard deviation (more uniform)
        assert std_low > std_med > std_high


def test_sample_key_value_real_gpt2(gpt2_model):
    """Test key-value sampling with real GPT-2 similarity scores."""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        batch_size = 2
        num_keys = 6
        hidden_size = gpt2_model.config.n_embd
        device = gpt2_model.device
        
        # Create query and key embeddings
        query_embeddings = torch.randn(batch_size, hidden_size, device=device)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size, device=device)
        
        # Compute similarity
        similarity_scores = compute_similarity(query_embeddings, key_embeddings, gpt2_model)
        
        # Test sampling with all keys available
        all_available_keys = [list(range(num_keys))] * batch_size
        
        # To make test deterministic, patch the categorical sampling
        with patch('torch.distributions.Categorical') as mock_categorical:
            mock_dist = MagicMock()
            mock_dist.sample.return_value = torch.tensor([0, 1], device=device)
            mock_categorical.return_value = mock_dist
            
            sampled_indices, sampled_probs = sample_key_value(
                similarity_scores, all_available_keys, batch_size
            )
            
            # Verify output
            assert len(sampled_indices) == batch_size
            assert sampled_probs.shape == (batch_size,)
            
            # Verify sampled probabilities match input
            for b in range(batch_size):
                assert torch.isclose(
                    sampled_probs[b], 
                    similarity_scores[b, sampled_indices[b]]
                )
        
        # Test masking: make only some keys available
        limited_available_keys = [
            [1, 3, 5],  # Only keys 1, 3, 5 for batch 0
            [0, 2, 4],  # Only keys 0, 2, 4 for batch 1
        ]
        
        # Sample without mocking
        sampled_indices, sampled_probs = sample_key_value(
            similarity_scores, limited_available_keys, batch_size
        )
        
        # Verify sampled indices are in the available keys
        for b in range(batch_size):
            assert sampled_indices[b] in limited_available_keys[b] 


def test_extract_embeddings_difference_with_lora(gpt2_model, gpt2_tokenizer):
    """
    Test that extract_embeddings produces different results for base model vs LoRA adapter model.
    This verifies that the LoRA adapter's weights are making a difference in the model's behavior.
    """
    import torch
    from src.model import apply_lora_adapter
    from src.embeddings import register_embedding_hook, extract_embeddings
    
    # Get the device from the model
    device = next(gpt2_model.parameters()).device
    
    # Create input tokens
    input_text = ["Hello world", "Testing GPT-2"]
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    
    # Apply LoRA adapter with patch for GPT-2
    with patch("src.model.MODEL_TYPE", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Extract embeddings from base model
    base_embeddings_dict, remove_hook_base = register_embedding_hook(gpt2_model, embed_type="query")
    base_embeddings = extract_embeddings(gpt2_model, input_ids, base_embeddings_dict)
    remove_hook_base()
    
    # Extract embeddings from adapter model
    adapter_embeddings_dict, remove_hook_adapter = register_embedding_hook(adapter_model, embed_type="query")
    adapter_embeddings = extract_embeddings(adapter_model, input_ids, adapter_embeddings_dict)
    remove_hook_adapter()
    
    # Check shapes
    assert base_embeddings.shape == adapter_embeddings.shape, "Embeddings shape mismatch"
    
    # Calculate difference between embeddings
    diff = torch.abs(base_embeddings - adapter_embeddings).sum()
    
    print(f"Base model embeddings shape: {base_embeddings.shape}")
    print(f"Adapter model embeddings shape: {adapter_embeddings.shape}")
    print(f"Base embeddings sum: {base_embeddings.sum()}")
    print(f"Adapter embeddings sum: {adapter_embeddings.sum()}")
    print(f"Absolute difference between embeddings: {diff}")
    
    # The embeddings should be different due to LoRA weights
    assert diff > 0, "LoRA weights should produce different embeddings than the base model"
    
    # Test log probabilities are different
    test_inputs = ["What is the capital of France?", "How does a computer work?"]
    encoded = gpt2_tokenizer(test_inputs, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        base_outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        adapter_outputs = adapter_model(input_ids=input_ids, attention_mask=attention_mask)
    
    base_logits = base_outputs.logits
    adapter_logits = adapter_outputs.logits
    
    logit_diff = torch.abs(base_logits - adapter_logits).sum()
    print(f"Base model logits shape: {base_logits.shape}")
    print(f"Adapter model logits shape: {adapter_logits.shape}")
    print(f"Base model logits sum: {base_logits.sum()}")
    print(f"Adapter model logits sum: {adapter_logits.sum()}")
    print(f"Absolute difference between logits: {logit_diff}")
    
    assert logit_diff > 0, "LoRA should produce different logits than the base model" 


def test_gpt2_projection_structure():
    """Test that GPT-2's attention projection structure is correctly understood.
    
    This validates our understanding of how GPT-2 splits its c_attn layer into
    query, key, and value projections.
    """
    from transformers import GPT2Model
    import torch
    
    # Load GPT-2 model
    model = GPT2Model.from_pretrained('gpt2')
    hidden_size = model.config.hidden_size  # Should be 768 for GPT-2
    
    # Access the first transformer block's attention module
    first_block = model.h[0]
    attn = first_block.attn
    c_attn = attn.c_attn
    
    # Test weight shapes
    assert c_attn.weight.shape == (hidden_size, 3 * hidden_size), \
        f"Expected c_attn weight shape ({hidden_size}, {3 * hidden_size}), got {c_attn.weight.shape}"
    assert c_attn.bias.shape == (3 * hidden_size,), \
        f"Expected c_attn bias shape ({3 * hidden_size},), got {c_attn.bias.shape}"
    
    # Test that weights can be split into three equal parts after transpose
    weight_transposed = c_attn.weight.transpose(0, 1)
    assert weight_transposed.shape[0] % 3 == 0, \
        "The transposed c_attn weight's first dimension should be divisible by 3"
    
    # Split the weights and verify shapes
    weight_splits = torch.split(weight_transposed, hidden_size, dim=0)
    assert len(weight_splits) == 3, "Should have exactly 3 weight splits (query, key, value)"
    for idx, w in enumerate(weight_splits):
        assert w.shape == (hidden_size, hidden_size), \
            f"Weight split {idx} should have shape ({hidden_size}, {hidden_size}), got {w.shape}"
    
    # Test bias splits
    bias_splits = torch.split(c_attn.bias, hidden_size)
    assert len(bias_splits) == 3, "Should have exactly 3 bias splits (query, key, value)"
    for idx, b in enumerate(bias_splits):
        assert b.shape == (hidden_size,), \
            f"Bias split {idx} should have shape ({hidden_size},), got {b.shape}"
    
    # Test forward pass
    batch_size = 1
    seq_length = 10
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    output = c_attn(dummy_input)
    
    assert output.shape == (batch_size, seq_length, 3 * hidden_size), \
        f"Expected output shape ({batch_size}, {seq_length}, {3 * hidden_size}), got {output.shape}"
    
    # Test that output can be split into query, key, value
    output_splits = torch.split(output, hidden_size, dim=-1)
    assert len(output_splits) == 3, "Should have exactly 3 output splits"
    for idx, out in enumerate(output_splits):
        assert out.shape == (batch_size, seq_length, hidden_size), \
            f"Output split {idx} should have shape ({batch_size}, {seq_length}, {hidden_size}), got {out.shape}"
    
    # Test full attention module forward pass
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    # GPT2 expects attention_mask to be broadcastable to [batch_size, num_heads, seq_length, seq_length]
    # Create a proper causal mask
    attention_mask = torch.ones(batch_size, 1, 1, seq_length)
    
    # This should work without errors
    attn_outputs = attn(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    # Verify outputs
    assert len(attn_outputs) >= 2, "Attention should return at least output and attention weights"
    attn_output = attn_outputs[0]
    assert attn_output.shape == (batch_size, seq_length, hidden_size), \
        f"Attention output should have shape ({batch_size}, {seq_length}, {hidden_size}), got {attn_output.shape}"
    
    if len(attn_outputs) > 1 and attn_outputs[1] is not None:
        attention_weights = attn_outputs[1]
        num_heads = model.config.n_head
        assert attention_weights.shape == (batch_size, num_heads, seq_length, seq_length), \
            f"Attention weights should have shape ({batch_size}, {num_heads}, {seq_length}, {seq_length}), got {attention_weights.shape}"


if __name__ == "__main__":
    pytest.main([__file__]) 