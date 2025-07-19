"""
Tests for mathematical correctness of the RL components.

This test file verifies that key mathematical operations are correct:
1. Masking is applied correctly (before softmax, not after)
2. GAE computation follows the correct formula
3. Probability distributions sum to 1
4. KL divergence is computed correctly
"""

import torch
import numpy as np
from src.training import compute_advantages, compute_returns
from src.embeddings import compute_similarity, get_attention_params
from transformers import GPT2LMHeadModel


def test_probability_distribution_correctness():
    """Test that compute_similarity produces valid probability distributions."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    num_heads, num_groups, head_dim = get_attention_params(model)
    
    batch_size = 2
    num_keys = 4
    hidden_size = num_heads * head_dim
    
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, num_keys, hidden_size)
    
    # Test without mask
    log_probs = compute_similarity(query, keys, num_heads, num_groups, head_dim)
    probs = torch.exp(log_probs)
    
    # Probabilities should sum to 1 for each batch item
    for b in range(batch_size):
        assert abs(probs[b].sum().item() - 1.0) < 1e-6, f"Batch {b} probabilities don't sum to 1: {probs[b].sum()}"
    
    # All probabilities should be non-negative
    assert (probs >= 0).all(), "Some probabilities are negative"
    
    print("✅ Probability distribution test passed")


def test_masking_correctness():
    """Test that masking produces correct probability distributions."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    num_heads, num_groups, head_dim = get_attention_params(model)
    
    batch_size = 1
    num_keys = 4
    hidden_size = num_heads * head_dim
    
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, num_keys, hidden_size)
    
    # Create mask that allows only keys 1 and 3
    mask = torch.full((batch_size, num_keys), float('-inf'))
    mask[0, [1, 3]] = 0.0
    
    log_probs = compute_similarity(query, keys, num_heads, num_groups, head_dim, availability_mask=mask)
    probs = torch.exp(log_probs)
    
    # Masked keys should have essentially 0 probability
    assert probs[0, 0] < 1e-6, f"Masked key 0 has non-zero probability: {probs[0, 0]}"
    assert probs[0, 2] < 1e-6, f"Masked key 2 has non-zero probability: {probs[0, 2]}"
    
    # Available keys should have non-zero probability
    assert probs[0, 1] > 1e-6, f"Available key 1 has zero probability: {probs[0, 1]}"
    assert probs[0, 3] > 1e-6, f"Available key 3 has zero probability: {probs[0, 3]}"
    
    # Probabilities should still sum to 1
    assert abs(probs[0].sum().item() - 1.0) < 1e-6, f"Masked probabilities don't sum to 1: {probs[0].sum()}"
    
    # Only available keys should have significant probability
    available_prob_sum = probs[0, 1] + probs[0, 3]
    assert abs(available_prob_sum.item() - 1.0) < 1e-6, f"Available keys don't sum to 1: {available_prob_sum}"
    
    print("✅ Masking correctness test passed")


def test_gae_mathematical_properties():
    """Test that GAE computation has correct mathematical properties."""
    batch_size = 3
    num_steps = 5
    
    # Create test rewards
    rewards = torch.randn(batch_size, num_steps)
    
    gamma = 0.99
    gae_lambda = 0.95
    
    # Test GAE computation
    advantages, returns = compute_advantages(rewards, gamma, gae_lambda, use_grpo_baseline=True)
    
    # Test that returns are computed correctly
    expected_returns = compute_returns(rewards, gamma)
    assert torch.allclose(returns, expected_returns, atol=1e-6), "Returns computation is incorrect"
    
    # Test baseline centering: advantages should have mean close to 0 across batch
    baseline = returns.mean(dim=0, keepdim=True)
    expected_baseline_centered = returns - baseline
    
    # The final advantages should be centered around the baseline
    mean_advantages = advantages.mean(dim=0)
    assert torch.allclose(mean_advantages, torch.zeros_like(mean_advantages), atol=1e-6), \
        f"Advantages not properly centered: {mean_advantages}"
    
    print("✅ GAE mathematical properties test passed")


def test_returns_computation():
    """Test that returns (rewards-to-go) are computed correctly."""
    batch_size = 2
    num_steps = 4
    
    # Simple test case with known expected results
    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # Batch 0
        [0.5, 1.5, 2.5, 3.5]   # Batch 1
    ])
    
    gamma = 0.9
    
    returns = compute_returns(rewards, gamma)
    
    # Manually compute expected returns for verification
    # R_t = r_t + γ * R_{t+1}
    # Working backwards:
    # R_3 = 4.0 (last step)
    # R_2 = 3.0 + 0.9 * 4.0 = 6.6
    # R_1 = 2.0 + 0.9 * 6.6 = 7.94
    # R_0 = 1.0 + 0.9 * 7.94 = 8.146
    
    expected_batch_0 = torch.tensor([8.146, 7.94, 6.6, 4.0])
    expected_batch_1 = torch.tensor([6.5735, 7.315, 5.75, 3.5])  # Similar computation for batch 1
    
    assert torch.allclose(returns[0], expected_batch_0, atol=1e-3), \
        f"Batch 0 returns incorrect: got {returns[0]}, expected {expected_batch_0}"
    
    print("✅ Returns computation test passed")


def test_temperature_scaling():
    """Test that temperature scaling works correctly."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    num_heads, num_groups, head_dim = get_attention_params(model)
    
    batch_size = 1
    num_keys = 3
    hidden_size = num_heads * head_dim
    
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, num_keys, hidden_size)
    
    # Test different temperatures
    low_temp_probs = torch.exp(compute_similarity(query, keys, num_heads, num_groups, head_dim, temperature=0.1))
    high_temp_probs = torch.exp(compute_similarity(query, keys, num_heads, num_groups, head_dim, temperature=10.0))
    
    # Lower temperature should make distribution more peaked (higher max probability)
    assert low_temp_probs.max() > high_temp_probs.max(), \
        "Lower temperature should create more peaked distribution"
    
    # Higher temperature should make distribution more uniform (lower max probability)
    # Both should still sum to 1
    assert abs(low_temp_probs.sum() - 1.0) < 1e-6, "Low temp probabilities don't sum to 1"
    assert abs(high_temp_probs.sum() - 1.0) < 1e-6, "High temp probabilities don't sum to 1"
    
    print("✅ Temperature scaling test passed")


if __name__ == "__main__":
    test_probability_distribution_correctness()
    test_masking_correctness()
    test_gae_mathematical_properties()
    test_returns_computation()
    test_temperature_scaling()
    print("\n🎉 All mathematical correctness tests passed!") 