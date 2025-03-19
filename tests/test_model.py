"""
Tests for the model module.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.config import MODEL_TYPE


@pytest.fixture
def mock_pretrained_model_for_llama():
    """Create a mock pre-trained model for Llama."""
    model = MagicMock()
    model.model.model.layers = []
    
    # Mock layers
    for i in range(2):
        layer = MagicMock()
        model.model.model.layers.append(layer)
    
    return model


@pytest.fixture
def mock_pretrained_model_for_gpt2():
    """Create a mock pre-trained model for GPT-2."""
    model = MagicMock()
    model.transformer.h = []
    
    # Mock layers
    for i in range(2):
        layer = MagicMock()
        model.transformer.h.append(layer)
    
    return model


@patch("src.model.MODEL_TYPE", "llama")
@patch("src.model.AutoModelForCausalLM.from_pretrained")
def test_load_base_model_llama(mock_from_pretrained, mock_pretrained_model_for_llama):
    """Test loading the base Llama model."""
    # Import here to use the patched MODEL_TYPE
    from src.model import load_base_model
    
    # Setup mocks
    mock_from_pretrained.return_value = mock_pretrained_model_for_llama
    
    # Call the function
    model = load_base_model()
    
    # Check that from_pretrained was called
    mock_from_pretrained.assert_called_once()
    
    # Check that we got the model
    assert model == mock_pretrained_model_for_llama


@patch("src.model.MODEL_TYPE", "gpt2")
@patch("src.model.AutoModelForCausalLM.from_pretrained")
def test_load_base_model_gpt2(mock_from_pretrained, mock_pretrained_model_for_gpt2):
    """Test loading the base GPT-2 model."""
    # Import here to use the patched MODEL_TYPE
    from src.model import load_base_model
    
    # Setup mocks
    mock_from_pretrained.return_value = mock_pretrained_model_for_gpt2
    
    # Call the function
    model = load_base_model()
    
    # Check that from_pretrained was called
    mock_from_pretrained.assert_called_once()
    
    # Check that we got the model
    assert model == mock_pretrained_model_for_gpt2


@patch("src.model.LoraConfig")
@patch("src.model.get_peft_model")
def test_apply_lora_adapter(mock_get_peft_model, mock_lora_config):
    """Test applying LoRA adapter to a model."""
    # Import here to avoid circular imports
    from src.model import apply_lora_adapter
    
    # Setup mocks
    mock_model = MagicMock()
    mock_lora_config.return_value = "lora_config"
    mock_get_peft_model.return_value = "peft_model"
    
    # Call the function
    result = apply_lora_adapter(mock_model)
    
    # Check that LoraConfig was created
    mock_lora_config.assert_called_once()
    
    # Check that get_peft_model was called with the right parameters
    mock_get_peft_model.assert_called_with(mock_model, "lora_config")
    
    # Check that we got the peft model
    assert result == "peft_model"


@patch("src.model.MODEL_TYPE", "llama")
def test_get_target_modules_llama():
    """Test getting target modules for LoRA in Llama models."""
    # Import here to use the patched MODEL_TYPE
    from src.model import get_target_modules
    
    # Call the function
    target_modules = get_target_modules()
    
    # Check that we got the right modules for Llama
    assert "q_proj" in target_modules
    assert "k_proj" in target_modules
    assert "v_proj" in target_modules
    assert "o_proj" in target_modules


@patch("src.model.MODEL_TYPE", "gpt2")
def test_get_target_modules_gpt2():
    """Test getting target modules for LoRA in GPT-2 models."""
    # Import here to use the patched MODEL_TYPE
    from src.model import get_target_modules
    
    # Call the function
    target_modules = get_target_modules()
    
    # Check that we got the right modules for GPT-2
    assert "c_attn" in target_modules
    assert "c_proj" in target_modules


@patch("src.model.torch.save")
def test_save_model_adapter(mock_save):
    """Test saving the model adapter."""
    # Import here to avoid circular imports
    from src.model import save_model_adapter
    
    # Setup mocks
    mock_model = MagicMock()
    mock_model.state_dict.return_value = {"weights": "values"}
    
    # Call the function
    save_model_adapter(mock_model, "path/to/save")
    
    # Check that state_dict was called
    mock_model.state_dict.assert_called_once()
    
    # Check that save was called with the right parameters
    mock_save.assert_called_with({"weights": "values"}, "path/to/save")


@patch("src.model.torch.load")
def test_load_model_adapter(mock_load):
    """Test loading the model adapter."""
    # Import here to avoid circular imports
    from src.model import load_model_adapter
    
    # Setup mocks
    mock_model = MagicMock()
    mock_load.return_value = {"weights": "values"}
    
    # Call the function
    load_model_adapter(mock_model, "path/to/load")
    
    # Check that load was called with the right path
    mock_load.assert_called_with("path/to/load")
    
    # Check that load_state_dict was called with the right parameters
    mock_model.load_state_dict.assert_called_with({"weights": "values"})


def test_real_gpt2_model_setup(gpt2_model, gpt2_tokenizer):
    """Test the complete flow with a real GPT-2 model."""
    from src.model import apply_lora_adapter
    from src.embeddings import get_attention_params
    import torch
    
    # Make sure we're working with a real GPT-2 model
    assert hasattr(gpt2_model, 'transformer')
    assert hasattr(gpt2_model, 'config')
    
    # In the real implementation, base model params are frozen before LoRA is applied
    # For testing purposes, make sure all base model params are already frozen
    for param in gpt2_model.parameters():
        param.requires_grad = False
    
    # Store trainable parameter count (should be 0 after freezing all params)
    original_trainable_count = sum(p.numel() for p in gpt2_model.parameters() if p.requires_grad)
    
    # Apply LoRA adapter (with MODEL_TYPE patch since we're using GPT-2)
    with patch("src.model.MODEL_TYPE", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Verify LoRA adapter added trainable parameters
    lora_trainable_count = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    
    # Now we should have trainable parameters (the LoRA weights)
    assert lora_trainable_count > 0, "LoRA should add trainable parameters"
    assert lora_trainable_count > original_trainable_count, "LoRA should increase trainable parameter count"
    
    # Test attention parameters extraction
    with patch("src.embeddings.MODEL_TYPE", "gpt2"):
        num_heads, num_kv_groups, head_dim = get_attention_params(adapter_model)
    
    # Verify parameters match GPT-2 config
    assert num_heads == adapter_model.config.n_head
    assert num_kv_groups == adapter_model.config.n_head  # In GPT-2, num_groups == num_heads (no GQA)
    assert head_dim == adapter_model.config.n_embd // adapter_model.config.n_head
    
    # Test forward pass through the model
    batch_size = 2
    seq_length = 10
    
    # Create input tokens
    input_text = ["Hello world", "Testing GPT-2"]
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(adapter_model.device)
    attention_mask = inputs["attention_mask"].to(adapter_model.device)
    
    # Run a forward pass
    with torch.no_grad():
        outputs = adapter_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Verify output shape (should be [batch_size, seq_length, vocab_size])
    assert outputs.logits.shape[0] == batch_size
    assert outputs.logits.shape[1] == input_ids.shape[1]
    assert outputs.logits.shape[2] == adapter_model.config.vocab_size
    
    # Check generate capability
    generated = adapter_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False
    )
    
    # Verify output (should be original input plus generated tokens)
    assert generated.shape[0] == batch_size
    assert generated.shape[1] > input_ids.shape[1]  # Should be longer
    
    # Decode the generated text to ensure it's reasonable
    decoded = gpt2_tokenizer.batch_decode(generated, skip_special_tokens=True)
    assert len(decoded) == batch_size
    assert all(isinstance(text, str) for text in decoded)
    assert all(len(text) > 0 for text in decoded) 