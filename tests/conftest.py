# tests/conftest.py
import pytest  # type: ignore
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Real model and tokenizer  # type: ignore
import torch
from src.config import TrainingConfig, CONFIG

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def gpt2_model():
    """Fixture that provides a real GPT-2 model for testing."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    return model

@pytest.fixture
def gpt2_tokenizer():
    """Fixture that provides a real GPT-2 tokenizer for testing."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer

@pytest.fixture
def tiny_llama_model():
    """Fixture that provides a tiny Llama model for testing (mock)."""
    # For tests, we'll create a mock Llama-like model structure
    class MockLlamaModel:
        def __init__(self):
            # Create a proper config structure
            self.config = type('Config', (), {
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_key_value_heads': 4,  # GQA configuration
            })()
            self.device = device
            
            # Create mock projection layers that embedding hooks expect
            class MockProjection:
                def __init__(self):
                    self.hook_fn = None
                    
                def register_forward_hook(self, hook_fn):
                    self.hook_fn = hook_fn
                    # Return a mock hook that can be removed
                    class MockHook:
                        def remove(self):
                            pass
                    return MockHook()
                
                def trigger_hook(self, input_tensor):
                    # Simulate forward pass and trigger hook
                    batch_size, seq_len = input_tensor.shape[:2]
                    hidden_size = 768
                    # Create mock output tensor
                    output_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
                    if self.hook_fn:
                        self.hook_fn(self, input_tensor, output_tensor)
                    return output_tensor
            
            # Create projection instances
            self.q_proj = MockProjection()
            self.k_proj = MockProjection()
            self.v_proj = MockProjection()
            
            # Create proper model structure that get_llama_attention_params expects
            # Create a mock self_attn with config attribute and projection layers
            mock_self_attn = type('MockSelfAttn', (), {
                'config': self.config,
                'q_proj': self.q_proj,
                'k_proj': self.k_proj,
                'v_proj': self.v_proj,
            })()
            
            # Create a mock layer with self_attn
            mock_layer = type('MockLayer', (), {
                'self_attn': mock_self_attn
            })()
            
            # Create model structure: model.model.layers[0]
            mock_model_inner = type('MockModelInner', (), {
                'layers': [mock_layer]
            })()
            
            self.model = type('MockModel', (), {
                'layers': [mock_layer]  # Also add direct access for fallback
            })()
            
        def parameters(self):
            # Return a dummy parameter to satisfy device checks
            yield torch.nn.Parameter(torch.zeros(1, device=device))
        
        def __call__(self, input_tokens):
            # Mock forward pass - trigger the registered hooks
            if hasattr(self, 'q_proj') and self.q_proj.hook_fn:
                self.q_proj.trigger_hook(input_tokens)
            return type('MockOutput', (), {})()
            
    return MockLlamaModel()

@pytest.fixture
def test_config_factory():
    """Factory for creating test configurations."""
    def create_config(**overrides):
        # Create a basic config with test-appropriate defaults
        config_dict = {
            'model_name': 'gpt2',
            'model_type': 'gpt2',
            'learning_rate': 1e-4,
            'num_episodes': 10,
            'batch_size': 2,
            'num_kv_pairs': 3,
            'tokens_per_key': 5,
            'tokens_per_value': 5,
            'device': str(device),  # Ensure device is set correctly
        }
        config_dict.update(overrides)
        return TrainingConfig(**config_dict)
    return create_config

@pytest.fixture(autouse=True)
def setup_config_for_tests():
    """Automatically set up a proper config for all tests."""
    # Create a test config with appropriate defaults
    test_config = TrainingConfig(
        model_name='gpt2',
        model_type='gpt2',
        device=str(device),
        batch_size=2,
        num_kv_pairs=3,
        tokens_per_key=10,
        tokens_per_value=10,
        learning_rate=1e-4,
    )
    
    # Set the config for all tests
    CONFIG.set_config(test_config)
    
    yield  # Run the test
    
    # Reset to default after test
    CONFIG.reset_to_default()

@pytest.fixture
def shape_validation_test():
    """Test fixture to validate shape checking in compute_similarity."""
    def test_shape_validation(gpt2_model, gpt2_tokenizer):
        """Test that compute_similarity properly validates tensor shapes."""
        import torch
        from src.embeddings import compute_similarity
        
        # Test with wrong query embedding dimensions
        batch_size = 2
        num_keys = 5
        
        # Get correct dimensions from model
        from src.embeddings import get_attention_params
        num_heads, num_groups, head_dim = get_attention_params(gpt2_model)
        correct_hidden_size = num_heads * head_dim
        correct_key_group_dim = num_groups * head_dim
        
        # Create correctly shaped key embeddings
        correct_key_embeddings = torch.randn(batch_size, num_keys, correct_hidden_size, device=device)
        
        # Test 1: Wrong query embedding dimensions (too small)
        wrong_query_embeddings = torch.randn(batch_size, correct_hidden_size - 10, device=device)
        
        try:
            compute_similarity(wrong_query_embeddings, correct_key_embeddings, num_heads, num_groups, head_dim)
            assert False, "Should have raised ValueError for wrong query embedding size"
        except ValueError as e:
            assert "query_embeddings hidden_size mismatch" in str(e)
            assert f"expected {correct_hidden_size}" in str(e)
            assert f"got {correct_hidden_size - 10}" in str(e)
        
        # Test 2: Wrong key embedding dimensions (insufficient for GQA)
        correct_query_embeddings = torch.randn(batch_size, correct_hidden_size, device=device)
        wrong_key_embeddings = torch.randn(batch_size, num_keys, correct_key_group_dim - 10, device=device)
        
        try:
            compute_similarity(correct_query_embeddings, wrong_key_embeddings, num_heads, num_groups, head_dim)
            assert False, "Should have raised ValueError for insufficient key embedding size"
        except ValueError as e:
            assert "key_embeddings hidden_size insufficient for GQA" in str(e)
            assert f"need at least {correct_key_group_dim}" in str(e)
            assert f"got {correct_key_group_dim - 10}" in str(e)
        
        # Test 3: Wrong tensor dimensions
        wrong_query_shape = torch.randn(batch_size, num_keys, correct_hidden_size, device=device)  # 3D instead of 2D
        
        try:
            compute_similarity(wrong_query_shape, correct_key_embeddings, num_heads, num_groups, head_dim)
            assert False, "Should have raised ValueError for wrong query tensor dimensions"
        except ValueError as e:
            assert "query_embeddings must be 2D tensor" in str(e)
            assert "got 3D tensor" in str(e)
        
        # Test 4: Batch size mismatch
        wrong_batch_key_embeddings = torch.randn(batch_size + 1, num_keys, correct_hidden_size, device=device)
        
        try:
            compute_similarity(correct_query_embeddings, wrong_batch_key_embeddings, num_heads, num_groups, head_dim)
            assert False, "Should have raised ValueError for batch size mismatch"
        except ValueError as e:
            assert "key_embeddings batch size mismatch" in str(e)
            assert f"expected {batch_size}" in str(e)
            assert f"got {batch_size + 1}" in str(e)
        
        # Test 5: Valid shapes should work
        try:
            result = compute_similarity(correct_query_embeddings, correct_key_embeddings, num_heads, num_groups, head_dim)
            assert result.shape == (batch_size, num_keys), f"Expected shape ({batch_size}, {num_keys}), got {result.shape}"
        except Exception as e:
            assert False, f"Valid shapes should not raise exception, but got: {e}"
        
        return True
    
    return test_shape_validation