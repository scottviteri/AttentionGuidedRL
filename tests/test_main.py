"""
Tests for the main module.
"""

import pytest
import torch
import argparse
from unittest.mock import MagicMock, patch
import copy

from src.data import KeyValuePair
from src.training import Trajectory


class MockArgs:
    """Mock command-line arguments."""
    def __init__(self):
        self.batch_size = 2
        self.resume = False
        self.episodes = 10
        self.trajectory_length = 3
        self.log_interval = 5


@pytest.fixture
def mock_kv_pair():
    """Create a mock key-value pair for testing."""
    batch_size = 2
    embedding_dim = 768
    
    return KeyValuePair(
        key_tokens=torch.randint(0, 1000, (batch_size, 20)),
        value_tokens=torch.randint(0, 1000, (batch_size, 20)),
        key_embedding=torch.randn(batch_size, embedding_dim),
        key_text=["key1", "key2"],
        value_text=["value1", "value2"],
    )


def test_generate_trajectory(mock_kv_pair):
    """Test generating a trajectory."""
    from src.main import generate_trajectory
    
    # Create mock models and tokenizer
    adapter_model = MagicMock()
    base_model = MagicMock()
    tokenizer = MagicMock()
    embeddings_dict = {"embeddings": None}
    hook_remover = MagicMock()
    
    # Create list of available KV pairs
    batch_size = 2
    available_kv_pairs = [mock_kv_pair] * 10
    
    # Create initial context
    context_tokens = torch.zeros((batch_size, 1), dtype=torch.long)
    
    # Set trajectory length for test
    trajectory_length = 3
    
    # Setup basic mocks
    tokenizer.batch_decode.return_value = ["context1", "context2"]
    
    # Mock the key functions in the generate_trajectory flow
    with patch("src.main.generate_query") as mock_generate_query:
        # Create a proper tensor for the query tokens
        mock_generate_query.return_value = torch.randint(0, 1000, (batch_size, 20))
        
        with patch("src.main.extract_embeddings") as mock_extract_embeddings:
            # Return embeddings with correct shape
            mock_extract_embeddings.return_value = torch.randn(batch_size, 768)
            
            with patch("src.main.compute_similarity") as mock_compute_similarity:
                # Return valid probability distribution
                mock_compute_similarity.return_value = torch.softmax(torch.randn(batch_size, 10), dim=1)
                
                with patch("src.main.sample_key_value") as mock_sample_key_value:
                    # Return valid sampled indices and probs
                    mock_sample_key_value.return_value = ([0, 1], torch.randn(batch_size))
                    
                    with patch("src.main.compute_trajectory_rewards") as mock_compute_rewards:
                        # Call generate_trajectory
                        trajectory = generate_trajectory(
                            context_tokens,
                            adapter_model,
                            base_model,
                            tokenizer,
                            embeddings_dict,
                            hook_remover,
                            available_kv_pairs,
                            batch_size,
                            trajectory_length=trajectory_length,
                        )
                        
                        # Verify the trajectory
                        assert trajectory is not None
                        assert len(trajectory.kv_pairs) == trajectory_length
                        
                        # Verify the key function calls
                        assert mock_generate_query.call_count == trajectory_length
                        assert mock_extract_embeddings.call_count == trajectory_length
                        assert mock_compute_similarity.call_count == trajectory_length
                        assert mock_sample_key_value.call_count == trajectory_length
                        assert mock_compute_rewards.call_count == 1


def test_parse_args():
    """Test parsing command-line arguments."""
    from src.main import parse_args
    
    # Mock sys.argv
    with patch("sys.argv", ["main.py"]):
        # Call function
        args = parse_args()
        
        # Check defaults
        assert args.batch_size == 2
        assert not args.resume
        assert args.trajectory_length == 3
        assert args.log_interval == 10
    
    # Mock sys.argv with arguments
    with patch("sys.argv", [
        "main.py",
        "--batch-size=4",
        "--resume",
        "--episodes=20",
        "--trajectory-length=5",
        "--log-interval=5"
    ]):
        # Call function
        args = parse_args()
        
        # Check parsed arguments
        assert args.batch_size == 4
        assert args.resume
        assert args.episodes == 20
        assert args.trajectory_length == 5
        assert args.log_interval == 5


class MockGenerator:
    """Mock generator that returns a fixed number of items before StopIteration."""
    def __init__(self, items, repeat_count=20):
        """Initialize the mock generator.
        
        Args:
            items: The items to yield
            repeat_count: How many times to yield the items before StopIteration
        """
        self.items = items
        self.index = 0
        self.repeat_count = repeat_count
        
    def __iter__(self):
        """Return self as iterator."""
        return self
        
    def __next__(self):
        """Return the next item or raise StopIteration."""
        if self.index < self.repeat_count:
            self.index += 1
            return self.items
        raise StopIteration


@patch("src.main.setup_logging")
@patch("src.main.setup_model_and_tokenizer")
@patch("src.main.register_embedding_hook")
@patch("src.main.iter_key_value_pairs")
@patch("src.main.generate_trajectory")
@patch("src.main.train_step")
@patch("src.main.save_checkpoint")
@patch("torch.optim.Adam")
@patch("src.main.logging")  # Mock logging to prevent actual logs
def test_main(
    mock_logging,
    mock_adam,
    mock_save_checkpoint,
    mock_train_step,
    mock_generate_trajectory,
    mock_iter_kv_pairs,
    mock_register_hook,
    mock_setup_models,
    mock_setup_logging
):
    """Test the main function (sanity check only)."""
    from src.main import main
    
    # Create a proper mock for tqdm
    mock_progress_bar = MagicMock()
    # Make the progress_bar iterable and return only a single episode to avoid many iterations
    mock_progress_bar.__iter__.return_value = iter([0])
    mock_progress_bar.set_description = MagicMock()
    
    # Mock optimizer
    mock_optimizer = MagicMock()
    mock_adam.return_value = mock_optimizer
    
    # Mock parse_args
    with patch("src.main.parse_args") as mock_parse_args:
        mock_args = MockArgs()
        mock_args.episodes = 1  # Set to 1 to minimize iterations
        mock_parse_args.return_value = mock_args
        
        # Create mock models
        base_model = MagicMock()
        adapter_model = MagicMock()
        tokenizer = MagicMock()
        
        # Setup adapter_model.parameters to return a valid parameter list
        mock_param = MagicMock()
        adapter_model.parameters.return_value = [mock_param]
        
        # Mock necessary return values
        mock_setup_logging.return_value = "logs/test"
        mock_setup_models.return_value = (base_model, adapter_model, tokenizer)
        mock_register_hook.return_value = ({"embeddings": None}, MagicMock())
        
        # Create a mock key-value pair
        mock_kv_pair = MagicMock()
        
        # Set up the mock generator that won't exhaust
        mock_kv_generator = MockGenerator(mock_kv_pair)
        mock_iter_kv_pairs.return_value = mock_kv_generator
        
        # Mock generate_trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.avg_reward = torch.tensor([1.0, 1.0])
        mock_generate_trajectory.return_value = mock_trajectory
        
        # Mock train_step
        mock_train_step.return_value = (0.5, 1)
        
        # Mock torch.zeros
        with patch("torch.zeros", return_value=torch.zeros((2, 1), dtype=torch.long)):
            # Mock load_checkpoint
            with patch("src.main.load_checkpoint", return_value=False):
                # Mock tqdm properly
                with patch("src.main.tqdm", return_value=mock_progress_bar):
                    # Mock next to handle batch key-value pair creation
                    with patch("builtins.next", side_effect=lambda gen: mock_kv_pair):
                        # Call function
                        main()
        
        # Check that setup functions were called
        mock_setup_logging.assert_called_once()
        mock_setup_models.assert_called_once()
        mock_register_hook.assert_called_once()
        
        # Check that train_step was called
        assert mock_train_step.call_count > 0
        
        # Check that save_checkpoint was called at least once
        assert mock_save_checkpoint.call_count > 0


def test_weights_update_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test that model weights are actually updated during training using a real model."""
    from src.training import train_step
    from src.model import apply_lora_adapter
    
    # Create a copy of the model with LoRA adapter
    adapter_model = apply_lora_adapter(gpt2_model)
    base_model = gpt2_model
    previous_model = copy.deepcopy(adapter_model)  # Copy for KL divergence
    
    # Create optimizer
    optimizer = torch.optim.Adam(adapter_model.parameters(), lr=0.01)
    
    # Create trajectory with rewards
    mock_trajectory = MagicMock()
    mock_trajectory.avg_reward = torch.tensor([1.0], device=adapter_model.device)
    mock_trajectory.rewards = torch.tensor([[1.0]], device=adapter_model.device)
    
    # Create a KV pair with tokens
    mock_kv_pair = MagicMock()
    mock_kv_pair.key_tokens = torch.randint(0, 100, (1, 5), device=adapter_model.device)
    
    # Set the trajectory's KV pairs
    mock_trajectory.kv_pairs = [mock_kv_pair]
    
    # Store initial weights
    initial_weights = {}
    for name, param in adapter_model.named_parameters():
        if param.requires_grad:  # Only check trainable params
            initial_weights[name] = param.data.clone()
    
    # Set up reward stats
    reward_stats = {"mean": 0.0, "std": 1.0, "count": 1}
    
    # Run a training step
    loss, num_filtered = train_step(
        [mock_trajectory],
        adapter_model,
        base_model,
        previous_model,
        optimizer,
        reward_stats,
        kl_penalty_coef=0.01
    )
    
    # Verify weights changed
    weights_changed = False
    for name, param in adapter_model.named_parameters():
        if param.requires_grad and name in initial_weights:
            if not torch.allclose(initial_weights[name], param.data, rtol=1e-4, atol=1e-4):
                weights_changed = True
                break
    
    assert weights_changed, "Model weights did not change after training step"
    assert num_filtered == 1, "Should have 1 filtered trajectory"


def test_embedding_pipeline():
    """Test the entire embedding extraction and similarity computation pipeline with real tensors.
    
    This tests the actual interfaces between embeddings.py and main.py to ensure
    tensor shapes are compatible and the embeddings flow correctly through the system.
    """
    import torch
    from src.embeddings import register_embedding_hook, extract_embeddings, compute_similarity, sample_key_value, get_attention_params
    
    # Create a small mock model with attention parameters that mimics real transformer architecture
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add a real parameter so next(model.parameters()) works
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
            
            self.model = MagicMock()
            self.model.model = MagicMock()
            
            # Create a mock attention layer with proper parameters
            class MockAttention:
                def __init__(self):
                    self.num_heads = 4
                    self.num_key_value_heads = 2  # GQA setup with 2 key/value heads
                    self.hidden_size = 128  # 4 heads * 32 head_dim
                    self.q_proj = torch.nn.Linear(128, 128)
                    self.k_proj = torch.nn.Linear(128, 64)  # Only 2 heads for keys in GQA
            
            # Create a mock layer
            class MockLayer:
                def __init__(self):
                    self.self_attn = MockAttention()
            
            # Set up the model structure
            self.model.model.layers = [MockLayer()]
        
        def __call__(self, tokens):
            # Forward pass that properly activates the hooks
            batch_size, seq_len = tokens.shape
            # Return embeddings with correct shape to be captured by the hook
            hidden_size = self.model.model.layers[0].self_attn.hidden_size
            # Simulate that the q_proj module gets called and outputs embeddings
            self.model.model.layers[0].self_attn.q_proj(torch.zeros(batch_size, seq_len, hidden_size))
            self.model.model.layers[0].self_attn.k_proj(torch.zeros(batch_size, seq_len, hidden_size))
            return None
    
    # Create the model
    model = MockModel()
    
    # 1. Test register_embedding_hook
    with patch('src.embeddings.MODEL_TYPE', 'llama'):
        embeddings_dict, hook_remover = register_embedding_hook(model, embed_type="query")
        
        # Verify the hook was registered and has the right structure
        assert "embeddings" in embeddings_dict
        assert callable(hook_remover)
        
        # 2. Test extract_embeddings
        batch_size = 2
        seq_len = 5
        
        # Create tokens input
        token_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Mock the hook capturing by directly setting the embeddings
        # This simulates what would happen in a real forward pass
        fake_embeddings = torch.randn(batch_size, seq_len, 128)  # [batch, seq, hidden]
        embeddings_dict["embeddings"] = fake_embeddings
        
        # Extract the embeddings
        query_embeddings = extract_embeddings(model, token_input, embeddings_dict)
        
        # Verify shape is correct (should be [batch, hidden])
        assert query_embeddings.shape == (batch_size, 128)
        
        # 3. Test compute_similarity with real tensors
        # Create some key embeddings
        num_keys = 3
        key_embeddings = torch.randn(batch_size, num_keys, 128)
        
        # Compute similarity scores
        similarity = compute_similarity(query_embeddings, key_embeddings, model)
        
        # Verify shape is correct (should be [batch, num_keys])
        assert similarity.shape == (batch_size, num_keys)
        
        # Verify they are probabilities (sum to 1, all between 0 and 1)
        for b in range(batch_size):
            assert torch.isclose(torch.sum(similarity[b]), torch.tensor(1.0), atol=1e-5)
            assert torch.all(similarity[b] >= 0) and torch.all(similarity[b] <= 1)
        
        # 4. Test sample_key_value
        # Create available keys for each batch
        available_keys = [
            [0, 1],      # Batch 0 has keys 0 and 1 available
            [0, 1, 2]    # Batch 1 has all keys available
        ]
        
        # Sample keys
        sampled_indices, sampled_probs = sample_key_value(similarity, available_keys, batch_size)
        
        # Verify the returned sampled indices are in the available keys
        assert sampled_indices[0] in available_keys[0]
        assert sampled_indices[1] in available_keys[1]
        
        # Verify the returned probabilities match the corresponding similarity scores
        assert torch.isclose(sampled_probs[0], similarity[0, sampled_indices[0]])
        assert torch.isclose(sampled_probs[1], similarity[1, sampled_indices[1]])
        
        # Finally, test get_attention_params
        heads, groups, head_dim = get_attention_params(model)
        assert heads == 4
        assert groups == 2
        assert head_dim == 32  # 128 / 4
        
        # Clean up the hook
        hook_remover()


def test_generate_trajectory_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test generating a trajectory with a real GPT-2 model."""
    from src.main import generate_trajectory
    from src.embeddings import compute_similarity, extract_embeddings, register_embedding_hook
    from src.data import KeyValuePair
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    import torch
    
    # We need to patch most of the external functions to make this test work
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        # This makes sure the get_attention_params returns correct heads/dimensions
        embeddings_dict, hook_remover = register_embedding_hook(gpt2_model)
        
        try:
            # Use simple test values that won't exercise the real functionality
            # but will verify the flow works correctly
            batch_size = 1
            device = gpt2_model.device
            
            # Create initial context
            context_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            
            # Mock necessary functions
            with patch("src.main.generate_query") as mock_generate_query:
                # Return query tokens with valid shape
                mock_generate_query.return_value = torch.randint(0, 1000, (batch_size, TOKENS_PER_KEY), device=device)
                
                with patch("src.main.extract_embeddings") as mock_extract_embeddings:
                    # Return embeddings with valid shape for the model
                    mock_extract_embeddings.return_value = torch.randn(batch_size, gpt2_model.config.n_embd, device=device)
                    
                    with patch("src.main.compute_similarity") as mock_compute_similarity:
                        # Return valid probability distribution
                        mock_compute_similarity.return_value = torch.softmax(torch.randn(batch_size, 2), dim=1)
                        
                        with patch("src.main.sample_key_value") as mock_sample_key_value:
                            # Return valid sampled indices and probabilities
                            mock_sample_key_value.return_value = ([0], torch.tensor([0.5], device=device))
                            
                            with patch("src.main.compute_trajectory_rewards"):
                                # Create simple test KV pairs
                                kv_pairs = []
                                for i in range(2):  # 2 test pairs
                                    key_tokens = torch.randint(0, 1000, (batch_size, TOKENS_PER_KEY), device=device)
                                    value_tokens = torch.randint(0, 1000, (batch_size, TOKENS_PER_VALUE), device=device)
                                    key_embedding = torch.randn(batch_size, gpt2_model.config.n_embd, device=device)
                                    
                                    kv_pair = KeyValuePair(
                                        key_tokens=key_tokens,
                                        value_tokens=value_tokens,
                                        key_embedding=key_embedding,
                                        key_text=["Test key " + str(i)],
                                        value_text=["Test value " + str(i)],
                                    )
                                    kv_pairs.append(kv_pair)
                                
                                # Call the function with the real model but minimal actual functionality
                                trajectory = generate_trajectory(
                                    context_tokens,
                                    gpt2_model,
                                    gpt2_model,
                                    gpt2_tokenizer,
                                    embeddings_dict,
                                    hook_remover,
                                    kv_pairs,
                                    batch_size,
                                    trajectory_length=1,  # Just get one pair to simplify test
                                )
                                
                                # Verify the basic structure is correct
                                assert trajectory is not None
                                assert hasattr(trajectory, 'kv_pairs')
                                assert len(trajectory.kv_pairs) == 1
        finally:
            # Clean up hook
            hook_remover() 