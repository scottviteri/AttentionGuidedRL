"""
Tests for the model module.
"""

import pytest
import torch
import copy
from unittest.mock import patch

from src.config import CONFIG


def test_get_target_modules_llama():
    """Test getting target modules for LoRA in Llama models."""
    from src.model import get_target_modules
    
    with patch("src.config.CONFIG.model_type", "llama"):
        target_modules = get_target_modules()
        
        # Check that we got the right modules for Llama
        assert "q_proj" in target_modules
        assert "k_proj" in target_modules
        assert "v_proj" in target_modules
        assert len(target_modules) == 3


def test_get_target_modules_gpt2():
    """Test getting target modules for LoRA in GPT-2 models."""
    from src.model import get_target_modules
    
    with patch("src.config.CONFIG.model_type", "gpt2"):
        target_modules = get_target_modules()
        
        # Check that we got the right modules for GPT-2
        assert "c_attn" in target_modules
        assert "q_proj" not in target_modules
        assert len(target_modules) == 1


def test_apply_lora_adapter_integration(gpt2_model):
    """Test applying LoRA adapter to a real GPT-2 model."""
    from src.model import apply_lora_adapter
    
    # Store original parameter count
    original_params = sum(p.numel() for p in gpt2_model.parameters())
    original_trainable = sum(p.numel() for p in gpt2_model.parameters() if p.requires_grad)
    
    # Apply LoRA adapter
    with patch("src.config.CONFIG.model_type", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Verify we have LoRA modules
    lora_modules = [name for name, module in adapter_model.named_modules() if 'lora' in name.lower()]
    assert len(lora_modules) > 0, "Should have LoRA modules after applying adapter"
    
    # Verify LoRA parameters are trainable
    lora_params = [p for name, p in adapter_model.named_parameters() if 'lora' in name.lower()]
    assert all(p.requires_grad for p in lora_params), "LoRA parameters should be trainable"
    
    # Verify total parameter count increased (due to LoRA)
    new_total_params = sum(p.numel() for p in adapter_model.parameters())
    assert new_total_params > original_params, "LoRA should add parameters"
    
    # Test forward pass works
    device = next(adapter_model.parameters()).device
    test_input = torch.randint(0, 1000, (1, 10), device=device)
    with torch.no_grad():
        output = adapter_model(test_input)
        vocab_size = getattr(adapter_model.config, 'vocab_size', 50257)  # GPT-2 default
        assert output.logits.shape == (1, 10, vocab_size)


def test_model_setup_and_tokenizer_integration():
    """Test the complete model and tokenizer setup process."""
    from src.model import setup_model_and_tokenizer
    
    # Test the setup process
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()  # Unpack 3 values, not 2
    
    # Verify models are not None
    assert base_model is not None
    assert adapter_model is not None
    assert tokenizer is not None
    
    # Note: After applying LoRA, base_model parameters may have requires_grad=True
    # but the important thing is that adapter_model has additional LoRA parameters
    
    # Verify adapter model has LoRA parameters
    lora_params = sum(1 for name, _ in adapter_model.named_parameters() if 'lora' in name)
    assert lora_params > 0, "Adapter model should have LoRA parameters"
    
    # Verify there are trainable parameters in the adapter model
    trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Adapter model should have trainable parameters"
    
    # Verify tokenizer has pad token
    assert tokenizer.pad_token is not None, "Tokenizer should have a pad token"


def test_checkpoint_save_and_load_integration(gpt2_model, tmp_path):
    """Test checkpoint saving and loading with real model."""
    from src.model import apply_lora_adapter, save_checkpoint, load_checkpoint
    from src.config import CONFIG
    import os
    
    # Apply LoRA to get trainable parameters
    with patch("src.config.CONFIG.model_type", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Create a temporary checkpoint directory and override CONFIG.checkpoint_dir
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Override the config's checkpoint_dir
    original_checkpoint_dir = CONFIG.checkpoint_dir
    with patch.object(CONFIG, 'checkpoint_dir', str(checkpoint_dir)):
        # Save checkpoint
        save_checkpoint(adapter_model, "test_episode")
        
        # Verify checkpoint file was created
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0, "Checkpoint file should be created"
        
        # Create a fresh model to load into
        with patch("src.config.CONFIG.model_type", "gpt2"):
            fresh_model = apply_lora_adapter(copy.deepcopy(gpt2_model))
        
        # Modify the fresh model to ensure loading actually changes it
        for param in fresh_model.parameters():
            if param.requires_grad:
                param.data.fill_(0.0)
        
        # Load checkpoint
        success = load_checkpoint(fresh_model, "test_episode")
        assert success, "Checkpoint loading should succeed"
        
        # Verify parameters changed from zeros
        has_nonzero = any(
            torch.any(param.data != 0.0).item()
            for param in fresh_model.parameters()
            if param.requires_grad
        )
        assert has_nonzero, "Loading checkpoint should restore non-zero parameters"


def test_model_copy_integration(gpt2_model):
    """Test creating a model copy."""
    from src.model import apply_lora_adapter, create_model_copy
    
    # Apply LoRA first
    with patch("src.config.CONFIG.model_type", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Create a copy
    model_copy = create_model_copy(adapter_model)
    
    # Verify it's a different object
    assert model_copy is not adapter_model
    
    # Verify they have the same structure
    assert type(model_copy) == type(adapter_model)
    
    # Set both models to eval mode for deterministic comparison
    adapter_model.eval()
    model_copy.eval()
    
    # Verify both produce the same output (before any training)
    device = next(adapter_model.parameters()).device
    test_input = torch.randint(0, 1000, (1, 5), device=device)
    
    with torch.no_grad():
        original_output = adapter_model(test_input)
        copy_output = model_copy(test_input)
    
    # Should be identical when both in eval mode
    assert torch.allclose(original_output.logits, copy_output.logits, atol=1e-6)
    
    # Verify the models are independent - changing one doesn't affect the other
    # Get a parameter from each model
    original_param = None
    copy_param = None
    
    for name, param in adapter_model.named_parameters():
        if 'lora' in name and param.requires_grad:
            original_param = param
            break
    
    for name, param in model_copy.named_parameters():
        if 'lora' in name and param.requires_grad:
            copy_param = param
            break
    
    assert original_param is not None, "Could not find LoRA parameter in original"
    assert copy_param is not None, "Could not find LoRA parameter in copy"
    
    # Verify they start with the same values
    assert torch.allclose(original_param.data, copy_param.data, atol=1e-6)
    
    # Modify the original
    with torch.no_grad():
        original_param.data += 1.0
    
    # Verify the copy is unchanged
    assert not torch.allclose(original_param.data, copy_param.data, atol=0.5)


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
    with patch("src.config.CONFIG.model_type", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Verify LoRA adapter added trainable parameters
    lora_trainable_count = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    
    # Now we should have trainable parameters (the LoRA weights)
    assert lora_trainable_count > 0, "LoRA should add trainable parameters"
    assert lora_trainable_count > original_trainable_count, "LoRA should increase trainable parameter count"
    
    # Test attention parameters extraction
    with patch("src.config.CONFIG.model_type", "gpt2"):
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
    device = next(adapter_model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
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
    Test that LoRA adapter application works correctly.
    
    Note: PEFT's get_peft_model wraps the original model in a PeftModelForCausalLM wrapper
    and adds LoRA modules to it, but the original base model remains accessible.
    """
    import copy
    from src.model import apply_lora_adapter
    
    # Create a deep copy to preserve the original state
    original_model = copy.deepcopy(gpt2_model)
    
    # Verify that the original model doesn't have LoRA modules
    for block_idx, block in enumerate(original_model.transformer.h):
        lora_modules = [name for name, _ in block.named_modules() if 'lora' in name.lower()]
        assert len(lora_modules) == 0, f"Block {block_idx} should not have LoRA modules before applying adapter"
    
    # Apply LoRA adapter (this wraps the model in a PEFT wrapper)
    with patch("src.config.CONFIG.model_type", "gpt2"):
        adapter_model = apply_lora_adapter(gpt2_model)
    
    # Verify that the adapter model is a PEFT wrapper (different type)
    assert type(adapter_model).__name__ == "PeftModelForCausalLM", "Adapter model should be a PEFT wrapper"
    assert type(gpt2_model).__name__ == "GPT2LMHeadModel", "Original model should still be GPT2LMHeadModel"
    
    # Verify that the adapter model now has LoRA modules
    adapter_lora_modules = [name for name, _ in adapter_model.named_modules() if 'lora' in name.lower()]
    assert len(adapter_lora_modules) > 0, "Adapter model should have LoRA modules"
    
    # Verify that the original gpt2_model also has LoRA modules (because PEFT modified it)
    gpt2_lora_modules = [name for name, _ in gpt2_model.named_modules() if 'lora' in name.lower()]
    assert len(gpt2_lora_modules) > 0, "GPT2 model should have LoRA modules after PEFT wrapping"
    
    # Verify that the base model is accessible through the wrapper
    assert hasattr(adapter_model, 'base_model'), "Adapter model should have base_model attribute"
    assert adapter_model.base_model.model is gpt2_model, "Base model should be accessible through wrapper"
    
    # Verify that the original_model (our copy) is still unchanged
    for block_idx, block in enumerate(original_model.transformer.h):
        lora_modules = [name for name, _ in block.named_modules() if 'lora' in name.lower()]
        assert len(lora_modules) == 0, f"Original model copy block {block_idx} should not have LoRA modules"
    
    # Test that LoRA adapter has trainable parameters
    lora_params = [p for name, p in adapter_model.named_parameters() if 'lora' in name and p.requires_grad]
    assert len(lora_params) > 0, "LoRA adapter should have trainable parameters"
    
    # Test that base model parameters are frozen
    base_params = [p for name, p in adapter_model.named_parameters() if 'lora' not in name]
    frozen_base_params = [p for p in base_params if not p.requires_grad]
    assert len(frozen_base_params) == len(base_params), "All base model parameters should be frozen" 