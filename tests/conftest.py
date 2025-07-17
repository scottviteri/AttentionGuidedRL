# tests/conftest.py
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

@pytest.fixture(scope="session")
def gpt2_model():
    """Load a GPT-2 model for testing."""
    # Use the standard GPT-2 (small) model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Always use GPU for tests to match project preference
    device = torch.device("cuda")
    model = model.to(device)
    # Enable evaluation mode
    model.eval()
    return model

@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Load the GPT-2 tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set padding token to be the same as the EOS token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture(scope="session")
def tiny_llama_model():
    """Create a tiny Llama model for testing GQA functionality.
    
    This creates a minimal Llama model with:
    - 2 layers (instead of 32+)
    - 128 hidden size (instead of 4096+)
    - 4 attention heads
    - 2 KV heads (for GQA testing)
    - Small vocabulary
    
    This allows us to test Llama-specific GQA functionality without
    the computational burden of a full model.
    """
    config = LlamaConfig(
        vocab_size=1000,        # Tiny vocabulary
        hidden_size=128,        # Small hidden size
        intermediate_size=256,  # Small intermediate size
        num_hidden_layers=2,    # Just 2 layers
        num_attention_heads=4,  # 4 query heads
        num_key_value_heads=2,  # 2 KV heads (GQA with 2:1 ratio)
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # Disable caching for testing
        use_cache=False,
    )
    
    # Create model from config
    model = LlamaForCausalLM(config)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize weights to small random values for stable testing
    for param in model.parameters():
        param.data.normal_(mean=0.0, std=0.02)
    
    return model

@pytest.fixture(scope="session")
def tiny_llama_tokenizer():
    """Create a simple tokenizer for the tiny Llama model."""
    # We can reuse GPT2 tokenizer for simplicity, just need to adjust vocab size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Note: In real use, the tokenizer vocab should match model vocab,
    # but for testing embedding extraction this mismatch is acceptable
    return tokenizer