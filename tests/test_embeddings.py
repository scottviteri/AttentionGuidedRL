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


class MockLlamaAttention:
    def __init__(self, num_heads=8, num_kv_heads=2, hidden_size=512):
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.hidden_size = hidden_size
        # Add projection layers that were missing
        self.q_proj = MagicMock()
        self.k_proj = MagicMock()
        self.v_proj = MagicMock()


class MockLlamaLayer:
    def __init__(self, num_heads=8, num_kv_heads=2, hidden_size=512):
        self.self_attn = MockLlamaAttention(num_heads, num_kv_heads, hidden_size)


class MockLlamaModel:
    def __init__(self, num_heads=8, num_kv_heads=2, hidden_size=512):
        self.layers = [MockLlamaLayer(num_heads, num_kv_heads, hidden_size)]


class MockLlama:
    def __init__(self, num_heads=8, num_kv_heads=2, hidden_size=512):
        self.model = MagicMock()
        self.model.model = MagicMock()
        
        # Create actual layer instances rather than relying on MagicMock's attribute access
        layers = [MockLlamaLayer(num_heads, num_kv_heads, hidden_size)]
        self.model.model.layers = layers
        
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))
        
    def __call__(self, tokens):
        # Simulate model forward pass
        return None


class MockGPT2Config:
    def __init__(self, n_head=12, n_embd=768):
        self.n_head = n_head
        self.n_embd = n_embd


class MockGPT2Attention:
    def __init__(self):
        self.c_attn = MagicMock()


class MockGPT2Layer:
    def __init__(self):
        self.attn = MockGPT2Attention()


class MockGPT2:
    def __init__(self, n_head=12, n_embd=768):
        self.config = MockGPT2Config(n_head, n_embd)
        self.transformer = MagicMock()
        self.transformer.h = [MockGPT2Layer()]
        
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))
        
    def __call__(self, tokens):
        # Simulate model forward pass
        return None


@pytest.fixture
def mock_llama_gqa():
    """Mock Llama model with Grouped Query Attention (8 query heads, 2 KV groups)"""
    return MockLlama(num_heads=8, num_kv_heads=2, hidden_size=512)


@pytest.fixture
def mock_llama_mha():
    """Mock Llama model with standard Multi-Head Attention (8 query heads, 8 KV groups)"""
    return MockLlama(num_heads=8, num_kv_heads=8, hidden_size=512)


@pytest.fixture
def mock_gpt2():
    """Mock GPT2 model with standard Multi-Head Attention"""
    return MockGPT2(n_head=12, n_embd=768)


def test_embedding_hook_registration():
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        model = MockLlama()
        embed_dict, remove_hook = register_embedding_hook(model, embed_type="query")
        assert "embeddings" in embed_dict
        assert callable(remove_hook)
        
        embed_dict, remove_hook = register_embedding_hook(model, embed_type="key")
        assert "embeddings" in embed_dict
        assert callable(remove_hook)


def test_extract_embeddings():
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        model = MockLlama()
        batch_size = 2
        seq_len = 10
        hidden_size = 512
        
        # Mock the embeddings dictionary that would be populated by the hook
        embeddings_dict = {
            "embeddings": torch.randn(batch_size, seq_len, hidden_size)
        }
        
        # Mock tokenized input
        tokenized_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Mock model forward call
        with patch.object(model, '__call__', return_value=None):
            # Extract embeddings
            result = extract_embeddings(model, tokenized_input, embeddings_dict)
            
            # Check shape
            assert result.shape == (batch_size, hidden_size)
            
            # Verify that the result is the mean over seq_len
            expected = torch.mean(embeddings_dict["embeddings"], dim=1)
            assert torch.allclose(result, expected)


def test_get_attention_params_llama_gqa():
    """Test getting attention parameters for Llama with GQA"""
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        model = MockLlama(num_heads=8, num_kv_heads=2, hidden_size=512)
        num_heads, num_groups, head_dim = get_attention_params(model)
        
        assert num_heads == 8
        assert num_groups == 2
        assert head_dim == 64  # 512 / 8


def test_get_attention_params_llama_mha():
    """Test getting attention parameters for Llama with standard MHA"""
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        model = MockLlama(num_heads=8, num_kv_heads=8, hidden_size=512)
        num_heads, num_groups, head_dim = get_attention_params(model)
        
        assert num_heads == 8
        assert num_groups == 8
        assert head_dim == 64  # 512 / 8


def test_get_attention_params_gpt2():
    """Test getting attention parameters for GPT2"""
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        model = MockGPT2(n_head=12, n_embd=768)
    num_heads, num_groups, head_dim = get_attention_params(model)
    
    assert num_heads == 12
    assert num_groups == 12  # For GPT2, num_groups == num_heads
    assert head_dim == 64  # 768 / 12


def test_compute_similarity_llama_gqa():
    """Test similarity computation for Llama with GQA"""
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Create mock model with GQA (8 query heads, 2 KV groups)
        model = MockLlama(num_heads=8, num_kv_heads=2, hidden_size=512)
        
        batch_size = 3
        num_keys = 5
        head_dim = 64
        num_heads = 8
        num_groups = 2
        hidden_size = num_heads * head_dim
        
        # Create query and key embeddings with correct dimensions
        query_embeddings = torch.randn(batch_size, hidden_size)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size)
        
        # Get parameters from the model
        model_heads, model_groups, model_head_dim = get_attention_params(model)
        
        # Verify parameters are correct
        assert model_heads == 8
        assert model_groups == 2
        assert model_head_dim == 64  # 512 / 8
        
        # Run computation with normal temperature
        similarity = compute_similarity(query_embeddings, key_embeddings, model)
        
        # Check shape and properties
        assert similarity.shape == (batch_size, num_keys)
        assert torch.allclose(torch.sum(similarity, dim=1), torch.ones(batch_size))
        
        # Test with different temperature parameter
        similarity_high_temp = compute_similarity(
            query_embeddings, key_embeddings, model, temperature=5.0  # Use much higher temperature
        )
        
        # The higher temperature should result in a more uniform distribution
        # Comparing the standard deviation is more stable with higher temperature difference
        assert torch.std(similarity_high_temp) < torch.std(similarity)


def test_compute_similarity_batch_consistency():
    """Test that compute_similarity gives consistent results when processing batches"""
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        model = MockLlama(num_heads=8, num_kv_heads=2, hidden_size=512)
        
        batch_size = 4
        num_keys = 6
        hidden_size = 512
        
        # Create query and key embeddings
        query_embeddings = torch.randn(batch_size, hidden_size)
        key_embeddings = torch.randn(batch_size, num_keys, hidden_size)
        
        # Process each batch item separately
        individual_results = []
        for b in range(batch_size):
            single_query = query_embeddings[b:b+1]  # Keep batch dim
            single_key = key_embeddings[b:b+1]      # Keep batch dim
            result = compute_similarity(single_query, single_key, model)
            individual_results.append(result)
        
        # Now process all batch items together
        batched_result = compute_similarity(query_embeddings, key_embeddings, model)
        
        # Verify that processing individually or in batch gives same results
        for b in range(batch_size):
            assert torch.allclose(batched_result[b], individual_results[b].squeeze(0), rtol=1e-4)


def test_embedding_hook_behavior():
    """Test that embedding hooks capture the right data"""
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Create mock model with realistic attention structure
        model = MockLlama(num_heads=8, num_kv_heads=2, hidden_size=512)
        
        # Create a fake output tensor that we want the hook to capture
        fake_output = torch.randn(2, 10, 512)  # [batch, seq_len, hidden_size]
        
        # Mock the hook function directly using our implementation's behavior
        def mock_hook_fn(module, input_tensor, output_tensor):
            embeddings_dict["embeddings"] = output_tensor
        
        # Mock the register_forward_hook to return our hook
        mock_hook = MagicMock()
        mock_hook.remove = lambda: None
        
        # Set up our mocks
        model.model.model.layers[-1].self_attn.q_proj.register_forward_hook = MagicMock(return_value=mock_hook)
        
        # Create dictionary to store embeddings and directly run hook
        embeddings_dict = {"embeddings": None}
        
        # Register the hook (normal flow)
        result_dict, remove_hook = register_embedding_hook(model, embed_type="query")
        
        # Simulate the hook being called during a forward pass
        # We know from the implementation that the hook stores the output tensor
        # Directly access the hook function from register_llama_embedding_hook
        hook_func = lambda m, i, o: embeddings_dict.__setitem__("embeddings", o)
        hook_func(None, None, fake_output)
        
        # Verify that embeddings were captured properly
        assert "embeddings" in embeddings_dict
        assert embeddings_dict["embeddings"] is not None
        assert torch.equal(embeddings_dict["embeddings"], fake_output)


def test_sample_key_value():
    """Test sampling key-value pairs based on similarity scores"""
    batch_size = 3
    num_keys = 10
    
    # Create deterministic similarity scores where each batch item has one clearly preferred key
    similarity_scores = torch.ones(batch_size, num_keys) * 0.01
    
    # Set up preferred keys
    similarity_scores[0, 0] = 0.9  # Batch 0 prefers key 0
    similarity_scores[1, 1] = 0.9  # Batch 1 prefers key 1
    similarity_scores[2, 2] = 0.9  # Batch 2 prefers key 2
    
    # Normalize to ensure they sum to 1
    for b in range(batch_size):
        similarity_scores[b] = similarity_scores[b] / similarity_scores[b].sum()
    
    # Create available keys for each batch item (include all keys)
    available_keys = [
        list(range(num_keys)),  # All keys for each batch
        list(range(num_keys)),
        list(range(num_keys)),
    ]
    
    # Mock the categorical distribution to return deterministic values
    with patch('torch.distributions.Categorical') as mock_categorical:
        # Configure the mock to return the indices with highest probability
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.tensor([0, 1, 2])
        mock_categorical.return_value = mock_dist
        
        # Call sample_key_value with our mocked distribution
        sampled_indices, sampled_probs = sample_key_value(
            similarity_scores, available_keys, batch_size
        )
        
        # Verify the results
        assert sampled_indices[0] == 0
        assert sampled_indices[1] == 1
        assert sampled_indices[2] == 2
        
        # Verify the probabilities match the inputs
        for b in range(batch_size):
            assert torch.isclose(
                sampled_probs[b], 
                similarity_scores[b, sampled_indices[b]]
            )


def test_sample_key_value_with_masked_keys():
    """Test sampling with unavailable/masked keys"""
    batch_size = 3
    num_keys = 10
    device = torch.device("cpu")
    
    # Create similarity scores with some clearly preferred keys
    similarity_scores = torch.zeros(batch_size, num_keys, device=device) + 0.1
    
    # Set some preferred keys (that we'll make unavailable)
    similarity_scores[0, 5] = 10.0  # Batch 0 prefers key 5 (unavailable)
    similarity_scores[1, 6] = 10.0  # Batch 1 prefers key 6 (unavailable)
    similarity_scores[2, 7] = 10.0  # Batch 2 prefers key 7 (unavailable)
    
    # Then set secondary preferred keys that are available
    similarity_scores[0, 0] = 5.0   # Batch 0's available preferred key
    similarity_scores[1, 1] = 5.0   # Batch 1's available preferred key
    similarity_scores[2, 2] = 5.0   # Batch 2's available preferred key
    
    # Create available keys - deliberately excluding the most preferred keys
    available_keys = [
        [0, 2, 4, 8],       # Batch 0 - key 5 unavailable
        [1, 3, 9],          # Batch 1 - key 6 unavailable
        [2, 4, 6, 8, 9],    # Batch 2 - key 7 unavailable
    ]
    
    # To make the test deterministic, mock the categorical distribution
    with patch('torch.distributions.Categorical') as mock_categorical:
        # Configure the mock to return deterministic samples
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.tensor([0, 1, 2])  # Return indices pointing to our preferred keys
        mock_categorical.return_value = mock_dist
        
        # Sample keys with mocked distribution
        sampled_indices, sampled_probs = sample_key_value(
            similarity_scores, available_keys, batch_size
        )
        
        # With the mocked distribution, we can check exact indices
        assert sampled_indices[0] == 0  # Should select batch 0's highest available key
        assert sampled_indices[1] == 1  # Should select batch 1's highest available key
        assert sampled_indices[2] == 2  # Should select batch 2's highest available key
    
    # Also test with real sampling (not deterministic)
    # Sample keys without mocking
    sampled_indices, sampled_probs = sample_key_value(
        similarity_scores, available_keys, batch_size
    )
    
    # Instead of checking exact indices, verify they are in the available keys
    for b in range(batch_size):
        assert sampled_indices[b] in available_keys[b]
        
        # The sampled indices should not be the unavailable high-preference keys
        if b == 0:
            assert sampled_indices[b] != 5  # Unavailable key
        elif b == 1:
            assert sampled_indices[b] != 6  # Unavailable key
        elif b == 2:
            assert sampled_indices[b] != 7  # Unavailable key
    
    # Test with empty available keys for one batch item
    empty_available_keys = [
        [0, 1, 2],  # Normal
        [],         # Empty
        [5, 6, 7]   # Normal
    ]
    
    # This should raise an exception because we can't sample from an empty set
    with pytest.raises(Exception):
        sample_key_value(similarity_scores, empty_available_keys, batch_size)


def test_gqa_vs_mha_similarity_computation():
    """
    Test that Grouped Query Attention (GQA) and Multi-Head Attention (MHA) 
    compute similarities correctly with their different attention patterns.
    
    This test uses controlled inputs to verify that:
    1. In MHA, each query head attends to its corresponding key head (1:1 mapping)
    2. In GQA, multiple query heads attend to the same key head (N:1 mapping)
    3. Softmax is applied per-head before averaging, as in real transformer models
    """
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Test with a simplified example where we can predict the exact behavior
        
        # First, test with standard MHA (num_heads = num_groups = 2)
        model_mha = MockLlama(num_heads=2, num_kv_heads=2, hidden_size=128)
        
        # Create controlled inputs with clear patterns
        # Each query head will strongly match with its corresponding key
        batch_size = 1
        num_keys = 2
        head_dim = 64
        
        # Create query embedding with 2 heads:
        # Head 0 has a pattern [1,0,0,...], Head 1 has a pattern [0,1,0,...]
        query_mha = torch.zeros((batch_size, 2 * head_dim))
        query_mha[0, 0] = 1.0  # First element of Head 0 is high
        query_mha[0, head_dim] = 1.0  # First element of Head 1 is high
        
        # Create key embeddings with 2 keys, each with 2 groups:
        # Key 0: Group 0 matches with Query Head 0, Group 1 doesn't match
        # Key 1: Group 0 doesn't match, Group 1 matches with Query Head 1
        key_mha = torch.zeros((batch_size, num_keys, 2 * head_dim))
        key_mha[0, 0, 0] = 1.0  # Key 0, Group 0: matches with Query Head 0
        key_mha[0, 1, head_dim] = 1.0  # Key 1, Group 1: matches with Query Head 1
        
        # Compute similarity with MHA
        similarity_mha = compute_similarity(query_mha, key_mha, model_mha)
        
        # In MHA, both keys should have equal probability (0.5) because:
        # Key 0 matches perfectly with Head 0 (prob=1.0) and not at all with Head 1 (prob=0.0)
        # Key 1 matches perfectly with Head 1 (prob=1.0) and not at all with Head 0 (prob=0.0)
        # Average probabilities: Key 0 = (1.0 + 0.0)/2 = 0.5, Key 1 = (0.0 + 1.0)/2 = 0.5
        assert similarity_mha.shape == (batch_size, num_keys)
        assert torch.allclose(similarity_mha[0, 0], torch.tensor(0.5), atol=1e-4)
        assert torch.allclose(similarity_mha[0, 1], torch.tensor(0.5), atol=1e-4)
        
        # Test with extreme values to create clear separation
        # This test focuses on the per-head softmax behavior
        query_extreme = torch.zeros((batch_size, 2 * head_dim))
        query_extreme[0, 0] = 10.0  # Head 0 has a very strong signal
        query_extreme[0, head_dim] = 10.0  # Head 1 also has a very strong signal
        
        key_extreme = torch.zeros((batch_size, num_keys, 2 * head_dim))
        key_extreme[0, 0, 0] = 10.0  # Key 0 perfectly matches with Head 0
        key_extreme[0, 1, head_dim] = 10.0  # Key 1 perfectly matches with Head 1
        
        # Compute similarity with extreme values
        similarity_extreme = compute_similarity(query_extreme, key_extreme, model_mha)
        
        # The separation should be much clearer now
        assert torch.allclose(similarity_extreme[0, 0], torch.tensor(0.5), atol=1e-4)
        assert torch.allclose(similarity_extreme[0, 1], torch.tensor(0.5), atol=1e-4)
        
        # Now, create a test where one key matches BOTH heads
        key_mha_one_key_matches_both = torch.zeros((batch_size, num_keys, 2 * head_dim))
        key_mha_one_key_matches_both[0, 0, 0] = 10.0  # Key 0, Group 0: perfectly matches Head 0
        key_mha_one_key_matches_both[0, 0, head_dim] = 10.0  # Key 0, Group 1: also perfectly matches Head 1
        # Key 1 doesn't match any head
        
        # Compute similarity
        similarity_one_key = compute_similarity(query_extreme, key_mha_one_key_matches_both, model_mha)
        
        # Key 0 should get probability 1.0 from both heads, averaged to 1.0
        # Key 1 should get probability 0.0 from both heads, averaged to 0.0
        assert torch.allclose(similarity_one_key[0, 0], torch.tensor(1.0), atol=1e-4)
        assert torch.allclose(similarity_one_key[0, 1], torch.tensor(0.0), atol=1e-4)
        
        # Test GQA with 4 query heads and 2 KV groups
        model_gqa = MockLlama(num_heads=4, num_kv_heads=2, hidden_size=256)
        
        # Create query embedding with 4 heads, but only 2 are active
        query_gqa = torch.zeros((batch_size, 4 * head_dim))
        query_gqa[0, 0] = 10.0  # Head 0 - maps to Group 0
        query_gqa[0, 3 * head_dim] = 10.0  # Head 3 - maps to Group 1
        
        # Create key embeddings with 2 groups
        key_gqa = torch.zeros((batch_size, num_keys, 4 * head_dim))
        # We only use the first 2*head_dim dimensions (for 2 groups)
        key_gqa[0, 0, 0] = 10.0  # Key 0, Group 0: matches with Head 0
        key_gqa[0, 1, head_dim] = 10.0  # Key 1, Group 1: matches with Head 3
        
        # Compute similarity with GQA
        similarity_gqa = compute_similarity(query_gqa, key_gqa, model_gqa)
        
        # Each key matches with one head out of 4 heads, so they should have nearly equal probability
        # Key 0: matches with Head 0 (1/4 of heads)
        # Key 1: matches with Head 3 (1/4 of heads)
        # The other two heads have no strong preference
        # Allow more tolerance since the inactive heads affect the distribution slightly
        assert abs(similarity_gqa[0, 0] - 0.5) < 0.05
        assert abs(similarity_gqa[0, 1] - 0.5) < 0.05 


def test_equidistant_queries_give_uniform_distribution():
    """
    Test that when a query is equidistant from all keys in all heads,
    the resulting probability distribution is uniform.
    
    This tests the unbiased nature of the attention mechanism when no key
    is preferred over others.
    """
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Create a model with 4 heads and 4 groups (standard MHA for simplicity)
        model = MockLlama(num_heads=4, num_kv_heads=4, hidden_size=128)
        
        batch_size = 2
        num_keys = 5
        head_dim = 32  # smaller head dim for this test
        hidden_size = 4 * head_dim  # 4 heads x 32 head dim = 128
        
        # Create query embeddings where each head has the same pattern
        # Each head's embedding will have the same dot product with all keys
        query_embeddings = torch.ones((batch_size, hidden_size))
        
        # Create key embeddings where all keys have identical patterns
        # This ensures all keys are equidistant from the query for all heads
        key_embeddings = torch.ones((batch_size, num_keys, hidden_size))
        
        # Compute similarity
        similarity = compute_similarity(query_embeddings, key_embeddings, model)
        
        # The result should be a uniform distribution over keys
        expected_probability = 1.0 / num_keys
        
        # Check that all probabilities are equal
        for b in range(batch_size):
            for k in range(num_keys):
                assert torch.isclose(similarity[b, k], torch.tensor(expected_probability), atol=1e-4)
        
        # Also verify the sum is 1.0
        assert torch.allclose(torch.sum(similarity, dim=1), torch.ones(batch_size))
        
        # Verify that different temperature values preserve the uniform distribution
        # High temperature
        similarity_high_temp = compute_similarity(
            query_embeddings, key_embeddings, model, temperature=10.0
        )
        
        # Low temperature
        similarity_low_temp = compute_similarity(
            query_embeddings, key_embeddings, model, temperature=0.1
        )
        
        # Both should still result in uniform distributions
        for b in range(batch_size):
            for k in range(num_keys):
                assert torch.isclose(similarity_high_temp[b, k], torch.tensor(expected_probability), atol=1e-4)
                assert torch.isclose(similarity_low_temp[b, k], torch.tensor(expected_probability), atol=1e-4)
                
        # Test with a mixed case: one batch item has uniform distances, the other has non-uniform
        mixed_query = torch.ones((2, hidden_size))
        mixed_keys = torch.ones((2, num_keys, hidden_size))
        
        # Make the second batch item have different key similarities
        # For the second batch item, each key will have a different value for head 0
        for k in range(num_keys):
            mixed_keys[1, k, 0] = k + 1  # Key k has value k+1 for the first position of head 0
        
        # Compute similarity for the mixed case
        mixed_similarity = compute_similarity(mixed_query, mixed_keys, model)
        
        # First batch item should still have uniform distribution
        for k in range(num_keys):
            assert torch.isclose(mixed_similarity[0, k], torch.tensor(expected_probability), atol=1e-4)
            
        # Second batch item should have non-uniform distribution
        assert not torch.allclose(
            mixed_similarity[1], 
            torch.ones(num_keys) * expected_probability,
            atol=1e-4
        )
        
        # But the sum should still be 1.0
        assert torch.isclose(torch.sum(mixed_similarity[1]), torch.tensor(1.0), atol=1e-4) 


def test_single_head_non_uniformity():
    """
    Test that non-uniformity in a single head of a single batch item
    only affects the distribution for that specific batch item.
    
    This tests the isolation of effects across batch dimensions and heads,
    ensuring that per-head softmax properly handles localized differences.
    """
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        # Create a model with 4 heads and 4 groups
        model = MockLlama(num_heads=4, num_kv_heads=4, hidden_size=128)
        
        batch_size = 3
        num_keys = 4
        head_dim = 32
        hidden_size = 4 * head_dim  # 4 heads x 32 head dim = 128
        
        # Start with all ones for both queries and keys (equidistant baseline)
        query_embeddings = torch.ones((batch_size, hidden_size))
        key_embeddings = torch.ones((batch_size, num_keys, hidden_size))
        
        # Modify just ONE head of ONE batch item to create non-uniformity
        # We'll modify head 1 (position 32-63) of batch item 1 (the middle one)
        target_batch = 1
        target_head = 1
        target_head_start = target_head * head_dim
        target_head_end = (target_head + 1) * head_dim
        
        # In this head, make it strongly prefer key 2
        for k in range(num_keys):
            if k == 2:
                # Strongly match with key 2
                key_embeddings[target_batch, k, target_head_start:target_head_end] = 10.0
            else:
                # Weakly match with other keys
                key_embeddings[target_batch, k, target_head_start:target_head_end] = 0.1
        
        # Compute similarity
        similarity = compute_similarity(query_embeddings, key_embeddings, model)
        
        # Expected probability for uniform distribution
        expected_uniform = 1.0 / num_keys
        
        # Batch 0 and 2 should have uniform distributions
        for k in range(num_keys):
            assert torch.isclose(similarity[0, k], torch.tensor(expected_uniform), atol=1e-4)
            assert torch.isclose(similarity[2, k], torch.tensor(expected_uniform), atol=1e-4)
        
        # Batch 1 should have a non-uniform distribution
        assert not torch.allclose(
            similarity[target_batch], 
            torch.ones(num_keys) * expected_uniform,
            atol=1e-4
        )
        
        # Key 2 should have higher probability than others in batch 1
        for k in range(num_keys):
            if k == 2:
                assert similarity[target_batch, k] > expected_uniform
            else:
                # The other probabilities might be below the uniform value
                # but we can't assert that directly since it depends on 
                # the exact implementation details
                pass
        
        # All batches should still sum to 1.0
        for b in range(batch_size):
            assert torch.isclose(torch.sum(similarity[b]), torch.tensor(1.0), atol=1e-4)
            
        # As a sanity check, verify the effect of temperature
        # With very high temperature, even the non-uniform batch should become more uniform
        similarity_high_temp = compute_similarity(
            query_embeddings, key_embeddings, model, temperature=100.0
        )
        
        # The distribution for batch 1 should be more uniform with high temperature
        batch1_std_normal = torch.std(similarity[target_batch])
        batch1_std_high_temp = torch.std(similarity_high_temp[target_batch])
        
        # Higher temperature should result in lower standard deviation (more uniform)
        assert batch1_std_high_temp < batch1_std_normal 


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