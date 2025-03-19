"""
Tests for the training module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import copy

from src.config import WARMUP_EPISODES, GENERATION_BATCH_SIZE, KL_PENALTY_COEFFICIENT, TOKENS_PER_KEY, TOKENS_PER_VALUE
from src.data import KeyValuePair


@pytest.fixture
def mock_kv_pair():
    """Create a mock key-value pair for testing."""
    batch_size = 2
    embedding_dim = 768
    
    return KeyValuePair(
        key_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_KEY)),
        value_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_VALUE)),
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
    kv_pairs = [mock_kv_pair, mock_kv_pair]
    
    # Create trajectory
    trajectory = Trajectory(kv_pairs=kv_pairs)
    
    # Add rewards
    batch_size = kv_pairs[0].key_tokens.shape[0]
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
    # Import here to avoid circular imports
    from src.training import generate_query
    
    # Create mock inputs
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 50001
    
    # Create a proper MagicMock for the tokenizer return value
    tokenizer_output = MagicMock()
    tokenizer_output.input_ids = torch.tensor([[101, 102], [101, 102]])
    tokenizer_output.attention_mask = torch.tensor([[1, 1], [1, 1]])
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output
    
    # Mock generate method to return exactly the expected length + context length
    expected_length = 10
    context_length = 2
    model.generate.return_value = torch.cat([
        torch.tensor([[101, 102], [101, 102]]),  # Context
        torch.randint(0, 1000, (2, expected_length))  # Generated tokens
    ], dim=1)
    
    # Call function with ensure_exact_length=True
    result = generate_query(
        model, 
        tokenizer, 
        ["context1", "context2"], 
        max_length=expected_length,
        ensure_exact_length=True
    )
    
    # Check output
    assert model.generate.called
    assert result.shape == (2, expected_length)  # Should have batch_size=2 outputs with exact length
    
    # Check that min_new_tokens parameter was passed correctly
    called_args = model.generate.call_args[1]
    assert called_args["min_new_tokens"] == expected_length
    assert called_args["max_new_tokens"] == expected_length


def test_compute_trajectory_rewards(mock_trajectory, mock_models):
    """Test computing rewards for a trajectory."""
    # Import here to avoid circular imports
    from src.training import compute_trajectory_rewards
    
    # Unpack mock models
    base_model, adapter_model, _ = mock_models
    
    # Create mock context
    batch_size = mock_trajectory.kv_pairs[0].key_tokens.shape[0]
    context_length = 30
    context_tokens = torch.randint(0, 1000, (batch_size, context_length))
    
    # Mock calculate_conditional_log_prob
    with patch("src.training.calculate_conditional_log_prob") as mock_calc:
        # Set up side effects to return different values for each call
        mock_calc.side_effect = [
            torch.tensor([-1.0, -2.0]),  # First adapter log probs
            torch.tensor([-3.0, -4.0]),  # First base log probs
            torch.tensor([-0.5, -1.5]),  # Second adapter log probs
            torch.tensor([-2.5, -3.5]),  # Second base log probs
        ]
        
        # Mock torch.cat to avoid device issues
        with patch("torch.cat", return_value=context_tokens):
            # Call function
            rewards = compute_trajectory_rewards(
                mock_trajectory,
                adapter_model,
                base_model,
                context_tokens
            )
        
        # Check output
        assert rewards.shape == (batch_size, len(mock_trajectory.kv_pairs))
        # Rewards should be calculated as adapter_log_prob - base_log_prob
        expected_rewards = torch.tensor([
            [2.0, 2.0],  # First pair rewards
            [2.0, 2.0]   # Second pair rewards
        ])
        assert torch.allclose(rewards, expected_rewards)
        
        # Check trajectory was updated with rewards
        assert mock_trajectory.rewards is not None
        assert mock_trajectory.avg_reward is not None
        assert torch.allclose(mock_trajectory.avg_reward, torch.tensor([2.0, 2.0]))


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
    """Test filtering trajectories based on reward."""
    # Import here to avoid circular imports
    from src.training import filter_trajectories
    from src.training import Trajectory
    
    # Create three separate trajectories with different rewards
    # We need separate instances to avoid shared references
    trajectory1 = Trajectory(kv_pairs=[])
    trajectory1.avg_reward = torch.tensor([0.5, 0.6])
    
    trajectory2 = Trajectory(kv_pairs=[])
    trajectory2.avg_reward = torch.tensor([1.5, 1.6])
    
    trajectory3 = Trajectory(kv_pairs=[])
    trajectory3.avg_reward = torch.tensor([2.5, 2.6])
    
    trajectories = [trajectory1, trajectory2, trajectory3]
    
    # Set reward stats above warmup threshold
    reward_stats = {"mean": 1.0, "std": 1.0, "count": WARMUP_EPISODES + 1}
    
    # Call function
    filtered = filter_trajectories(trajectories, reward_stats)
    
    # Check output - should keep trajectories with avg_reward > mean + std (2.0)
    assert len(filtered) == 1
    assert filtered[0].avg_reward[0].item() == 2.5


def test_compute_policy_loss(mock_trajectory, mock_models):
    """Test computing the policy gradient loss."""
    # Import here to avoid circular imports
    from src.training import compute_policy_loss
    
    # Unpack mock models
    _, adapter_model, previous_model = mock_models
    
    # Ensure mock_trajectory has rewards
    assert mock_trajectory.rewards is not None
    assert mock_trajectory.avg_reward is not None
    
    # Mock parameter().device
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    adapter_model.parameters = MagicMock(return_value=iter([mock_param]))
    
    # Mock the model outputs
    batch_size = mock_trajectory.kv_pairs[0].key_tokens.shape[0]
    vocab_size = 1000
    seq_length = TOKENS_PER_KEY
    
    current_logits = torch.randn(batch_size, seq_length, vocab_size)
    previous_logits = torch.randn(batch_size, seq_length, vocab_size)
    
    adapter_model.return_value = MagicMock(logits=current_logits)
    previous_model.return_value = MagicMock(logits=previous_logits)
    
    # Call function
    loss = compute_policy_loss(
        [mock_trajectory], 
        adapter_model, 
        previous_model, 
        KL_PENALTY_COEFFICIENT
    )
    
    # Check output is a scalar tensor
    assert loss.dim() == 0
    assert loss.dtype == torch.float32


def test_train_step(mock_models, mock_trajectory):
    """Test a complete training step."""
    # Import here to avoid circular imports
    from src.training import train_step
    
    # Unpack mock models
    base_model, adapter_model, previous_model = mock_models
    
    # Create mock optimizer
    optimizer = MagicMock()
    
    # Mock compute_policy_loss
    with patch("src.training.compute_policy_loss", return_value=torch.tensor(1.0, requires_grad=True)):
        # Mock filter_trajectories
        with patch("src.training.filter_trajectories", return_value=[mock_trajectory]):
            # Call function
            loss, num_filtered = train_step(
                [mock_trajectory], 
                adapter_model, 
                base_model,
                previous_model,
                optimizer, 
                {"mean": 0.0, "std": 1.0, "count": 10},
                KL_PENALTY_COEFFICIENT
            )
    
    # Check outputs
    assert isinstance(loss, float)
    assert num_filtered == 1
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
    trajectory = Trajectory(kv_pairs=[kv_pair])
    
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
    with patch("src.training.filter_trajectories", return_value=[trajectory]):
        train_step(
            [trajectory],
            adapter_model,
            base_model,
            previous_model,
            optimizer,
            {"mean": 0.0, "std": 1.0, "count": WARMUP_EPISODES + 1},
            KL_PENALTY_COEFFICIENT
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
    """Test generating a query with a real GPT-2 model."""
    from src.training import generate_query
    
    # Setup inputs
    context_texts = ["This is a test context.", "Another test context."]
    max_length = 10
    
    # Patch the model type
    with patch('src.config.MODEL_TYPE', 'gpt2'):
        # Generate queries
        query_tokens = generate_query(
            gpt2_model,
            gpt2_tokenizer,
            context_texts,
            max_length=max_length,
            ensure_exact_length=True
        )
        
        # Verify output
        assert query_tokens is not None
        assert query_tokens.shape[1] == max_length  # Should be exactly max_length
        assert query_tokens.shape[0] == len(context_texts)  # Should match batch size
        assert query_tokens.device == gpt2_model.device


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
    trajectory = Trajectory(kv_pairs=kv_pairs)
    
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
    
    # Create trajectories with real Trajectory objects
    trajectories = []
    batch_size = 1
    
    for i in range(2):
        # Create a proper KeyValuePair
        kv_pair = KeyValuePair(
            key_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_KEY), device=gpt2_model.device),
            value_tokens=torch.randint(0, 1000, (batch_size, TOKENS_PER_VALUE), device=gpt2_model.device),
            key_embedding=torch.randn(batch_size, gpt2_model.config.n_embd, device=gpt2_model.device),
            key_text=[f"Test key {i}"],
            value_text=[f"Test value {i}"]
        )
        
        # Use real Trajectory objects
        trajectory = Trajectory(kv_pairs=[kv_pair])
        trajectory.rewards = torch.tensor([[0.5]], device=gpt2_model.device)
        trajectory.avg_reward = torch.tensor([0.5], device=gpt2_model.device)
        
        trajectories.append(trajectory)
    
    # Setup reward stats
    reward_stats = {"mean": 0.0, "std": 1.0, "count": 10}
    
    # Patch compute_policy_loss to return a tensor with requires_grad=True
    with patch('src.training.compute_policy_loss') as mock_compute_policy_loss:
        # Create a tensor that requires grad for the backward pass
        mock_loss = torch.tensor([0.1], device=gpt2_model.device, requires_grad=True)
        mock_compute_policy_loss.return_value = mock_loss
        
        # Run train step
        loss, num_filtered = train_step(
            trajectories,
            adapter_model,
            gpt2_model,
            previous_model,
            optimizer,
            reward_stats,
            kl_penalty_coef=0.1
        )
    
    # Verify output
    assert isinstance(loss, float)
    assert num_filtered >= 0


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