"""
Tests for the training module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import copy
import torch.nn.functional as F
import logging

from src.config import WARMUP_EPISODES, GENERATION_BATCH_SIZE, KL_PENALTY_COEFFICIENT, TOKENS_PER_KEY, TOKENS_PER_VALUE, QUERY_PREFIX
from src.data import KeyValuePair, QKVStep


@pytest.fixture
def mock_kv_pair():
    """Create a mock key-value pair for testing."""
    batch_size = 2
    embedding_dim = 768
    
    return KeyValuePair(
        key_tokens=torch.randint(0, 1000, (batch_size, 10)),
        value_tokens=torch.randint(0, 1000, (batch_size, 10)),
        key_embedding=torch.randn(batch_size, embedding_dim),
        key_text=["key1", "key2"],
        value_text=["value1", "value2"],
    )


@pytest.fixture
def mock_trajectory(mock_kv_pair):
    """Create a mock trajectory with KV pairs."""
    # Import here to avoid circular imports
    from src.training import Trajectory
    
    # Create two KV pairs
    qkv_steps = [mock_kv_pair, mock_kv_pair]
    
    # Create trajectory
    trajectory = Trajectory(qkv_steps=qkv_steps)
    
    # Add rewards
    batch_size = qkv_steps[0].key_tokens.shape[0]
    trajectory.rewards = torch.tensor([[0.5, 0.6], [0.7, 0.8]])  # [batch_size, num_pairs]
    trajectory.avg_reward = torch.tensor([0.55, 0.75])  # [batch_size]
    
    return trajectory


@pytest.fixture
def mock_models():
    """Create mock models for training."""
    base_model = MagicMock()
    adapter_model = MagicMock()
    previous_model = MagicMock()
    return base_model, adapter_model, previous_model


def test_calculate_conditional_log_prob():
    """Test calculating conditional log probability."""
    # Import here to avoid circular imports
    from src.training import calculate_conditional_log_prob
    
    # Create mock inputs
    model = MagicMock()
    
    # Input shapes
    batch_size = 2
    seq_length = 10
    vocab_size = 1000
    context_length = 20
    
    # Create mock tokens and context
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    context = torch.randint(0, vocab_size, (batch_size, context_length))
    
    # Mock model output with logits that have a large enough sequence length
    mock_logits = torch.rand(batch_size, context_length, vocab_size)
    model.return_value = MagicMock(logits=mock_logits)
    
    # Call function
    log_probs = calculate_conditional_log_prob(model, tokens, context)
    
    # Check output
    assert log_probs.shape == (batch_size,)
    # All log probabilities should be non-positive
    assert torch.all(log_probs <= 0)


def test_generate_query():
    """Test generating a query."""
    # Import the function
    from src.training import generate_query
    from src.config import QUERY_PREFIX
    
    # Mock tokenizer, model, and context
    tokenizer = MagicMock()
    model = MagicMock()
    context = ["Context 1", "Context 2"]
    
    # Setup model.generate to return a valid output
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    
    # Setup tokenizer to return a valid encoding object with a 'to' method
    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.tensor([[1, 2], [3, 4]])
    tokenizer_output.attention_mask = torch.tensor([[1, 1], [1, 1]])
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output
    tokenizer.eos_token_id = 50001
    
    # Call function
    with patch("src.training.DEVICE", torch.device("cpu")):
        result = generate_query(model, tokenizer, context)
    
    # Check tokenizer was called with the right parameters
    tokenizer.assert_called_once()
    assert all(f"{ctx}{QUERY_PREFIX}" in tokenizer.call_args[0][0] for ctx in context)
    
    # Check model.generate was called with the right parameters
    model.generate.assert_called_once()
    # Don't assert specific token counts as they may change
    assert "min_new_tokens" in model.generate.call_args[1]
    
    # Check result
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == len(context)


def test_compute_trajectory_rewards(mock_trajectory, mock_models):
    """Test computing trajectory rewards."""
    # Import here to avoid circular imports
    from src.training import compute_trajectory_rewards
    
    # Unpack models
    base_model, adapter_model, _ = mock_models
    
    # Batch size and dimensions
    batch_size = mock_trajectory.qkv_steps[0].key_tokens.shape[0]
    
    # Mock model behaviors
    adapter_model.generate.return_value = torch.randint(0, 1000, (batch_size, 20))
    
    # Mock conditional log probs
    def mock_log_prob(model, *args, **kwargs):
        return torch.tensor([0.1, 0.2])
        
    with patch('src.training.calculate_conditional_log_prob', side_effect=mock_log_prob):
        # Create context tokens
        context_tokens = torch.randint(0, 1000, (batch_size, 5))
        
        # Compute rewards
        rewards = compute_trajectory_rewards(
            mock_trajectory,
            adapter_model,
            base_model,
            context_tokens,
        )
        
        # Verify shapes and rewards computation
        assert rewards is not None
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (batch_size, len(mock_trajectory.qkv_steps))


def test_update_reward_stats():
    """Test updating reward statistics."""
    # Import here to avoid circular imports
    from src.training import update_reward_stats
    
    # Create initial stats
    stats = {"mean": 0.0, "std": 1.0, "count": 0}
    
    # New rewards
    rewards = torch.tensor([1.0, 3.0])
    
    # Call function
    updated_stats = update_reward_stats(stats, rewards)
    
    # Check output
    assert updated_stats["count"] == 2
    assert updated_stats["mean"] == 2.0
    # With initial count=0, std should be computed directly from rewards
    assert updated_stats["std"] == 1.0


def test_filter_trajectories():
    """Test filtering batch elements within a trajectory based on reward."""
    # Import here to avoid circular imports
    from src.training import filter_trajectories
    from src.training import Trajectory
    from src.data import KeyValuePair, QKVStep
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    
    # Enable debug logging temporarily
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a trajectory with batch dimensions
    batch_size = 3
    kv_pair = KeyValuePair(
        key_tokens=torch.randint(0, 1000, (batch_size, 10)),
        value_tokens=torch.randint(0, 1000, (batch_size, 10)),
        key_embedding=torch.zeros(batch_size, 10),
        key_text=["key1", "key2", "key3"],
        value_text=["value1", "value2", "value3"]
    )
    
    trajectory = Trajectory(qkv_steps=[kv_pair])
    
    # Set rewards with different values for each batch element
    # (first element below threshold, other two above)
    trajectory.avg_reward = torch.tensor([0.5, 1.5, 2.0])
    trajectory.rewards = torch.tensor([[0.5], [1.5], [2.0]])
    
    # Set reward stats above warmup threshold
    reward_stats = {"mean": 1.0, "std": 1.0, "count": 10}
    
    # Use a very low percentile (33.3%) to ensure we filter out the first element
    # With 33.3%, the threshold index would be int(3 * (1 - 33.3/100)) - 1 = int(3 * 0.667) - 1 = 2 - 1 = 1
    # So we should use the value at index 1 (1.5) as the threshold, keeping only elements >= 1.5
    filtered = filter_trajectories(trajectory, reward_stats, percentile=33.3)
    
    # Check output - should keep batch elements with avg_reward >= 1.5
    assert filtered is not None
    assert filtered.avg_reward.shape[0] == 2, f"Expected 2 elements, got {filtered.avg_reward.shape[0]}: {filtered.avg_reward}"
    
    # Reset log level
    logging.basicConfig(level=logging.INFO)


def test_compute_policy_loss(mock_trajectory, mock_models):
    """Test computing policy loss with KL regularization."""
    # Import here to avoid circular imports
    from src.training import compute_policy_loss
    
    # Unpack models
    _, adapter_model, previous_model = mock_models
    
    # Extract batch size
    batch_size = mock_trajectory.qkv_steps[0].key_tokens.shape[0]
    
    # Ensure mock_trajectory has rewards
    assert mock_trajectory.rewards is not None
    assert mock_trajectory.avg_reward is not None
    
    # Mock parameter().device
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    adapter_model.parameters = MagicMock(return_value=iter([mock_param]))
    
    # Mock the model outputs
    vocab_size = 1000
    seq_length = TOKENS_PER_KEY
    
    current_logits = torch.randn(batch_size, seq_length, vocab_size)
    previous_logits = torch.randn(batch_size, seq_length, vocab_size)
    
    adapter_model.return_value = MagicMock(logits=current_logits)
    previous_model.return_value = MagicMock(logits=previous_logits)
    
    # Call function
    total_loss, policy_loss, kl_loss = compute_policy_loss(
        mock_trajectory,
        adapter_model,
        previous_model,
        KL_PENALTY_COEFFICIENT
    )
    
    # Check output is a scalar tensor
    assert total_loss.dim() == 0
    assert policy_loss.dim() == 0
    assert kl_loss.dim() == 0


def test_train_step(mock_models, mock_trajectory):
    """Test a complete training step."""
    # Import here to avoid circular imports
    from src.training import train_step
    
    # Unpack mock models
    base_model, adapter_model, previous_model = mock_models
    
    # Create mock optimizer
    optimizer = MagicMock()
    
    # Mock compute_policy_loss
    with patch("src.training.compute_policy_loss", return_value=(torch.tensor(1.0, requires_grad=True), torch.tensor(0.7, requires_grad=True), torch.tensor(0.3, requires_grad=True))):
        # Mock filter_trajectories to return the trajectory with filtered batch elements
        with patch("src.training.filter_trajectories", return_value=mock_trajectory):
            # Call function
            total_loss, num_filtered, policy_loss, kl_loss = train_step(
                mock_trajectory, 
                adapter_model, 
                base_model,
                previous_model,
                optimizer, 
                {"mean": 0.0, "std": 1.0, "count": 10},
                KL_PENALTY_COEFFICIENT,
                verbose=False
            )
    
    # Check outputs
    assert isinstance(total_loss, float)
    assert isinstance(num_filtered, int)
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(kl_loss, torch.Tensor)
    assert optimizer.zero_grad.called
    assert optimizer.step.called


def test_model_behavior_during_training():
    """Test that the base model stays the same while adapter model changes during training."""
    import torch.nn as nn
    from src.model import apply_lora_adapter
    from src.training import generate_query, compute_trajectory_rewards, train_step
    from src.data import KeyValuePair
    from src.training import Trajectory
    
    # Create simple test model that can be used with LoRA
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)
            self.output = nn.Linear(64, 100)
            
        def forward(self, input_ids):
            embeds = self.embedding(input_ids)
            hidden = self.q_proj(embeds) + self.k_proj(embeds) + self.v_proj(embeds)
            hidden = self.o_proj(hidden)
            logits = self.output(hidden)
            return MagicMock(logits=logits)
            
        def generate(self, input_ids, **kwargs):
            # Simple mock generation: just append some tokens
            batch_size = input_ids.shape[0]
            new_tokens = torch.randint(0, 100, (batch_size, kwargs.get('max_new_tokens', 1)))
            return torch.cat([input_ids, new_tokens], dim=1)
    
    # Create tokenizer mock
    tokenizer = MagicMock()
    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.randint(0, 100, (2, 5))
    tokenizer_output.attention_mask = torch.ones(2, 5)
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output
    tokenizer.eos_token_id = 0
    
    # Create models
    base_model = SimpleModel()
    base_model.num_heads = 4
    base_model.num_key_value_groups = 2
    base_model.hidden_size = 64
    
    # Apply adapter to create adapter model
    with patch("src.model.LoraConfig"):
        with patch("src.model.get_peft_model", lambda model, config: model):
            adapter_model = apply_lora_adapter(SimpleModel())
            adapter_model.num_heads = 4
            adapter_model.num_key_value_groups = 2
            adapter_model.hidden_size = 64
    
    # Create an input for testing
    test_input = torch.randint(0, 100, (2, 10))
    
    # 1. Verify both models give different outputs before training
    with torch.no_grad():
        base_output_before = base_model(test_input).logits
        adapter_output_before = adapter_model(test_input).logits
        
    # Models should start with different outputs due to initialization
    assert not torch.allclose(base_output_before, adapter_output_before)
    
    # 2. Create trajectories and train adapter model
    # Create KV pair
    kv_pair = KeyValuePair(
        key_tokens=torch.randint(0, 100, (2, TOKENS_PER_KEY)),
        value_tokens=torch.randint(0, 100, (2, TOKENS_PER_VALUE)),
        key_embedding=torch.randn(2, 64),
        key_text=["key1", "key2"],
        value_text=["value1", "value2"],
    )
    
    # Create trajectory
    trajectory = Trajectory(qkv_steps=[kv_pair])
    
    # Setup for training
    optimizer = torch.optim.Adam(adapter_model.parameters(), lr=0.001)
    
    # Compute rewards for the trajectory
    with patch("src.training.calculate_conditional_log_prob", side_effect=[
        torch.tensor([-1.0, -2.0]),  # Adapter log probs
        torch.tensor([-3.0, -4.0]),  # Base log probs
    ]):
        # Compute rewards
        with patch("torch.cat", return_value=test_input):
            compute_trajectory_rewards(trajectory, adapter_model, base_model, test_input)
        
    # Keep a copy of previous adapter model
    previous_model = SimpleModel()
    previous_model.load_state_dict(adapter_model.state_dict())
    
    # Perform training step
    with patch("src.training.filter_trajectories", return_value=trajectory):
        train_step(
            trajectory,
            adapter_model,
            base_model,
            previous_model,
            optimizer,
            {"mean": 0.0, "std": 1.0, "count": WARMUP_EPISODES + 1},
            KL_PENALTY_COEFFICIENT,
            verbose=False
        )
    
    # 3. Verify base model output hasn't changed, but adapter model has
    with torch.no_grad():
        base_output_after = base_model(test_input).logits
        adapter_output_after = adapter_model(test_input).logits
    
    # Base model should remain unchanged
    assert torch.allclose(base_output_before, base_output_after)
    
    # Adapter model should change after training
    assert not torch.allclose(adapter_output_before, adapter_output_after)
    
    # Adapter and base models should have different outputs after training
    assert not torch.allclose(base_output_after, adapter_output_after)


def test_generate_query_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test generating queries with an actual GPT-2 model."""
    from src.training import generate_query
    from src.config import TOKENS_PER_KEY
    
    # Test with a real model and tokenizer
    context = ["This is a test context."]
    
    # Move model to CPU for testing to avoid device mismatch
    gpt2_model = gpt2_model.to("cpu")
    
    with patch("src.training.DEVICE", torch.device("cpu")):
        result = generate_query(gpt2_model, gpt2_tokenizer, context)
    
    # Check the result has the expected shape
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 1  # Batch size 1
    # Don't assert specific token counts
    assert result.shape[1] > 0  # At least some tokens generated
    
    # Check we can decode the result back to text
    decoded = gpt2_tokenizer.decode(result[0])
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_compute_trajectory_rewards_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test computing trajectory rewards with real models."""
    from src.training import compute_trajectory_rewards, Trajectory, calculate_conditional_log_prob
    from src.data import KeyValuePair
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    
    # Create a simple trajectory with some key-value pairs
    batch_size = 1
    kv_pairs = []
    
    # Create a few key-value pairs
    for i in range(3):
        kv_pair = KeyValuePair(
            key_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_KEY), device=gpt2_model.device),
            value_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_VALUE), device=gpt2_model.device),
            key_embedding=torch.randn(batch_size, gpt2_model.config.n_embd, device=gpt2_model.device),
            key_text=[f"Key {i}"],
            value_text=[f"Value {i}"]
        )
        kv_pairs.append(kv_pair)
    
    # Create trajectory
    trajectory = Trajectory(qkv_steps=kv_pairs)
    
    # Create initial context
    context_tokens = torch.randint(0, 1000, (batch_size, 5), device=gpt2_model.device)
    
    # With some patching to avoid full model runs
    with patch('src.training.calculate_conditional_log_prob', return_value=torch.tensor([0.5], device=gpt2_model.device)):
        # Compute rewards
        compute_trajectory_rewards(trajectory, gpt2_model, gpt2_model, context_tokens)
        
        # Verify rewards were computed
        assert trajectory.rewards is not None
        assert trajectory.avg_reward is not None
        assert trajectory.rewards.shape[0] == batch_size


def test_train_step_with_real_model(gpt2_model):
    """Test training step with a real GPT-2 model."""
    from src.training import train_step, Trajectory
    from src.model import apply_lora_adapter
    from src.data import KeyValuePair
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    import copy
    
    # Set up adapter model with LoRA
    adapter_model = apply_lora_adapter(gpt2_model)
    previous_model = copy.deepcopy(adapter_model)
    
    # Create optimizer
    optimizer = torch.optim.Adam(adapter_model.parameters(), lr=0.001)
    
    # Spy on optimizer methods
    original_zero_grad = optimizer.zero_grad
    original_step = optimizer.step
    zero_grad_called = [False]
    step_called = [False]
    
    def spy_zero_grad(*args, **kwargs):
        zero_grad_called[0] = True
        return original_zero_grad(*args, **kwargs)
        
    def spy_step(*args, **kwargs):
        step_called[0] = True
        return original_step(*args, **kwargs)
        
    optimizer.zero_grad = spy_zero_grad
    optimizer.step = spy_step
    
    # Create a batched trajectory
    batch_size = 2
    
    # Create a proper KeyValuePair with batch dimension
    kv_pair = KeyValuePair(
        key_tokens=torch.randint(0, 100, (batch_size, 10), device=gpt2_model.device),
        value_tokens=torch.randint(0, 100, (batch_size, 10), device=gpt2_model.device),
        key_embedding=torch.randn(batch_size, gpt2_model.config.n_embd, device=gpt2_model.device),
        key_text=[f"Test key {i}" for i in range(batch_size)],
        value_text=[f"Test value {i}" for i in range(batch_size)]
    )
    
    # Use real Trajectory object with batch dimension
    trajectory = Trajectory(qkv_steps=[kv_pair])
    trajectory.rewards = torch.tensor([[0.5], [1.5]], device=gpt2_model.device)
    trajectory.avg_reward = torch.tensor([0.5, 1.5], device=gpt2_model.device)
    
    # Setup reward stats
    reward_stats = {"mean": 0.0, "std": 1.0, "count": 10}
    
    # Patch compute_policy_loss to return a tuple (total_loss, policy_loss, kl_loss)
    with patch('src.training.compute_policy_loss') as mock_compute_policy_loss:
        # Create tensors that require grad for the backward pass
        mock_total_loss = torch.tensor([0.1], device=gpt2_model.device, requires_grad=True)
        mock_policy_loss = torch.tensor([0.07], device=gpt2_model.device, requires_grad=True)
        mock_kl_loss = torch.tensor([0.03], device=gpt2_model.device, requires_grad=True)
        mock_compute_policy_loss.return_value = (mock_total_loss, mock_policy_loss, mock_kl_loss)
        
        # Run train step
        total_loss, num_filtered, policy_loss, kl_loss = train_step(
            trajectory,
            adapter_model,
            gpt2_model,
            previous_model,
            optimizer,
            reward_stats,
            kl_penalty_coef=0.1,
            verbose=False
        )
    
    # Verify output
    assert isinstance(total_loss, float)
    assert isinstance(num_filtered, int)
    assert isinstance(policy_loss, torch.Tensor)
    assert isinstance(kl_loss, torch.Tensor)
    assert zero_grad_called[0]  # Check that zero_grad was called
    assert step_called[0]       # Check that step was called


def test_conditional_log_prob_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test calculating conditional log probability with a real GPT-2 model."""
    # Import here to avoid circular imports
    from src.training import calculate_conditional_log_prob
    
    # Create real token sequences
    batch_size = 2
    
    # Create context text and continuation text
    context_text = ["Hello world", "Testing the model"]
    continuation_text = ["how are you", "with real tokens"]
    
    # Tokenize context
    context_encoded = gpt2_tokenizer(context_text, return_tensors="pt", padding=True)
    context_tokens = context_encoded.input_ids.to(gpt2_model.device)
    
    # Tokenize continuation
    continuation_encoded = gpt2_tokenizer(continuation_text, return_tensors="pt", padding=True)
    continuation_tokens = continuation_encoded.input_ids.to(gpt2_model.device)
    
    # Call the function with the real model
    log_probs = calculate_conditional_log_prob(gpt2_model, continuation_tokens, context_tokens)
    
    # Verify the output
    assert log_probs.shape == (batch_size,)
    assert torch.all(log_probs <= 0)  # Log probabilities should be non-positive
    
    # Try with different contexts to ensure variance in probabilities
    new_context_text = ["Once upon a time", "In a galaxy far"]
    new_context_encoded = gpt2_tokenizer(new_context_text, return_tensors="pt", padding=True)
    new_context_tokens = new_context_encoded.input_ids.to(gpt2_model.device)
    
    new_log_probs = calculate_conditional_log_prob(gpt2_model, continuation_tokens, new_context_tokens)
    
    # The log probabilities should be different with different contexts
    assert not torch.allclose(log_probs, new_log_probs, atol=1e-3)
    
    # Also verify that different continuations produce different probabilities
    # Use a fixed context for this test
    fixed_context_tokens = context_tokens
    
    new_continuation_text = ["this is different", "completely new"]
    new_continuation_encoded = gpt2_tokenizer(new_continuation_text, return_tensors="pt", padding=True)
    new_continuation_tokens = new_continuation_encoded.input_ids.to(gpt2_model.device)
    
    different_log_probs = calculate_conditional_log_prob(gpt2_model, new_continuation_tokens, fixed_context_tokens)
    
    # The log probabilities should be different with different continuations
    assert not torch.allclose(log_probs, different_log_probs, atol=1e-3) 