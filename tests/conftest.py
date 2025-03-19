# tests/conftest.py
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.fixture(scope="session")
def gpt2_model():
    """Load a GPT-2 model for testing."""
    # Use the standard GPT-2 (small) model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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