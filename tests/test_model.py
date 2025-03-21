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
@patch("src.model.copy.deepcopy")
@patch("torch.no_grad")
def test_apply_lora_adapter(mock_no_grad, mock_deepcopy, mock_get_peft_model, mock_lora_config):
    """Test applying LoRA adapter to a model."""
    # Import here to avoid circular imports
    from src.model import apply_lora_adapter
    
    # Setup mocks
    mock_model = MagicMock()
    mock_model_copy = MagicMock()
    mock_peft_model = MagicMock()
    
    # Setup return values
    mock_deepcopy.return_value = mock_model_copy
    mock_lora_config.return_value = "lora_config"
    mock_get_peft_model.return_value = mock_peft_model
    
    # Setup mock modules for lora_B initialization
    mock_lora_module = MagicMock()
    mock_lora_module.lora_B = {"default": MagicMock()}
    mock_lora_module.lora_B["default"].weight = MagicMock()
    
    # Mock the named_modules method to return some modules with lora_B
    mock_peft_model.named_modules.return_value = [
        ("module1", mock_lora_module),
        ("module2", MagicMock())  # module without lora_B
    ]
    
    # Call the function
    result = apply_lora_adapter(mock_model)
    
    # Check that the model was deep-copied
    mock_deepcopy.assert_called_once_with(mock_model)
    
    # Check that LoraConfig was created with the expected parameters
    mock_lora_config.assert_called_once()
    
    # Check that get_peft_model was called with the right parameters
    mock_get_peft_model.assert_called_with(mock_model_copy, "lora_config")
    
    # Check that named_modules was called for the LoRA initialization
    mock_peft_model.named_modules.assert_called()
    
    # Check that we normalized weights for modules with lora_B
    mock_lora_module.lora_B["default"].weight.normal_.assert_called_once()
    
    # Check that we got the peft model
    assert result == mock_peft_model


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


def test_base_model_unchanged_after_lora(gpt2_model):
    """
    Test that the base model remains unchanged after applying LoRA adapter.
    
    This test verifies that applying LoRA adapter to a model does not modify the 
    original model in place, which could cause unexpected behavior in training.
    """
    import copy
    from src.model import apply_lora_adapter
    
    # Create a deep copy of the model architecture for comparison
    # We can't directly compare model parameters because the model includes non-leaf tensors
    base_model_architecture = str(gpt2_model)
    
    # Store the original structure of c_attn layers (before LoRA)
    original_c_attn_layers = []
    for block_idx, block in enumerate(gpt2_model.transformer.h):
        original_c_attn_layers.append(str(block.attn.c_attn))
    
    # Verify that the base model doesn't have LoRA modules already
    for block_idx, block in enumerate(gpt2_model.transformer.h):
        assert "lora" not in str(block.attn.c_attn).lower(), f"Base model already has LoRA modules: {block.attn.c_attn}"
    
    # Apply LoRA adapter with patch for GPT-2
    with patch("src.model.MODEL_TYPE", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Verify that adapter model now has LoRA modules
    for block_idx, block in enumerate(adapter_model.transformer.h):
        assert "lora" in str(block.attn.c_attn).lower(), f"Adapter model missing LoRA modules: {block.attn.c_attn}"
    
    # If the base model is also changed, this means apply_lora_adapter is modifying it in-place
    for block_idx, block in enumerate(gpt2_model.transformer.h):
        current_c_attn = str(block.attn.c_attn)
        assert "lora" not in current_c_attn.lower(), (
            f"Base model was modified after applying LoRA adapter. Block {block_idx}, "
            f"Original: {original_c_attn_layers[block_idx]}, Current: {current_c_attn}"
        )
    
    # Verify overall model architecture hasn't changed
    assert base_model_architecture == str(gpt2_model), "Base model architecture changed after applying LoRA"


def test_lora_weight_initialization(gpt2_model):
    """
    Test that both lora_A and lora_B weights are properly initialized with non-zero values.
    """
    import torch
    from src.model import apply_lora_adapter
    
    # Apply LoRA adapter with patch for GPT-2
    with patch("src.model.MODEL_TYPE", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Check that lora_A and lora_B weights exist and are not zero
    lora_a_nonzero = False
    lora_b_nonzero = False
    
    # Iterate through all modules to find LoRA layers
    for name, module in adapter_model.named_modules():
        # Check lora_A weights
        if hasattr(module, 'lora_A'):
            for key in module.lora_A.keys():
                # Check if weights are non-zero (at least some of them)
                weights = module.lora_A[key].weight
                if torch.abs(weights).sum() > 0:
                    lora_a_nonzero = True
                    
        # Check lora_B weights
        if hasattr(module, 'lora_B'):
            for key in module.lora_B.keys():
                # Check if weights are non-zero (at least some of them)
                weights = module.lora_B[key].weight
                if torch.abs(weights).sum() > 0:
                    lora_b_nonzero = True
    
    # Both lora_A and lora_B should have non-zero weights
    assert lora_a_nonzero, "LoRA A weights are all zeros"
    assert lora_b_nonzero, "LoRA B weights are all zeros"
    
    # Print statistics about LoRA weights for debugging
    lora_a_stats = []
    lora_b_stats = []
    
    for name, module in adapter_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for key in module.lora_A.keys():
                a_weights = module.lora_A[key].weight
                b_weights = module.lora_B[key].weight
                
                lora_a_stats.append({
                    'mean': torch.mean(a_weights).item(),
                    'std': torch.std(a_weights).item(),
                    'min': torch.min(a_weights).item(),
                    'max': torch.max(a_weights).item(),
                    'nonzero': (torch.abs(a_weights) > 1e-10).float().mean().item() * 100
                })
                
                lora_b_stats.append({
                    'mean': torch.mean(b_weights).item(),
                    'std': torch.std(b_weights).item(),
                    'min': torch.min(b_weights).item(),
                    'max': torch.max(b_weights).item(),
                    'nonzero': (torch.abs(b_weights) > 1e-10).float().mean().item() * 100
                })
                
    # Average statistics
    a_mean_stats = {k: sum(stat[k] for stat in lora_a_stats) / len(lora_a_stats) for k in lora_a_stats[0]}
    b_mean_stats = {k: sum(stat[k] for stat in lora_b_stats) / len(lora_b_stats) for k in lora_b_stats[0]}
    
    print(f"LoRA A stats: {a_mean_stats}")
    print(f"LoRA B stats: {b_mean_stats}")
    
    # Check that the statistics are reasonable
    assert abs(a_mean_stats['mean']) < 0.1, "LoRA A weights have unexpectedly large mean"
    assert a_mean_stats['std'] > 0, "LoRA A weights have zero standard deviation"
    assert b_mean_stats['std'] > 0, "LoRA B weights have zero standard deviation" 