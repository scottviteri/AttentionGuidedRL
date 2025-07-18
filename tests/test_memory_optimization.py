"""
Test cases for memory-efficient LoRA state management.

These tests verify that we can manage LoRA adapter states without
storing multiple full model copies, achieving significant memory savings.
"""

import pytest
import torch
import copy
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import (
    load_base_model, 
    apply_lora_adapter,
    save_lora_state,
    load_lora_state,
    update_lora_ema,
    create_model_copy
)
from src.config import CONFIG


class TestLoRAStateOperations:
    """Test basic LoRA state save/load operations."""
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        # Mock a simple model with both LoRA and non-LoRA parameters
        model = MagicMock()
        
        # Create realistic parameter structure
        lora_params = {
            'base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight': torch.randn(8, 512),
            'base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight': torch.randn(512, 8),
            'base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight': torch.randn(8, 512),
            'base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight': torch.randn(512, 8),
        }
        
        base_params = {
            'base_model.model.layers.0.self_attn.q_proj.base_layer.weight': torch.randn(512, 512),
            'base_model.model.layers.0.self_attn.v_proj.base_layer.weight': torch.randn(512, 512),
            'base_model.model.embed_tokens.weight': torch.randn(50257, 512),
        }
        
        all_params = {**lora_params, **base_params}
        
        # Mock named_parameters to return our test parameters
        def named_parameters():
            for name, param in all_params.items():
                mock_param = MagicMock()
                mock_param.data = param
                yield name, mock_param
        
        model.named_parameters = named_parameters
        return model, lora_params, base_params
    
    def test_save_lora_state_extracts_only_lora_params(self, mock_models):
        """Test that save_lora_state extracts only LoRA parameters."""
        model, expected_lora, base_params = mock_models
        
        lora_state = save_lora_state(model)
        
        # Should contain all LoRA parameters
        assert len(lora_state) == len(expected_lora)
        
        # Should only contain LoRA parameters
        for name in lora_state.keys():
            assert 'lora_' in name
            
        # Should not contain base model parameters  
        for name in base_params.keys():
            assert name not in lora_state
            
        # Values should match
        for name, expected_tensor in expected_lora.items():
            assert torch.equal(lora_state[name], expected_tensor)
    
    def test_load_lora_state_updates_only_lora_params(self, mock_models):
        """Test that load_lora_state updates only LoRA parameters."""
        model, original_lora, base_params = mock_models
        
        # Create new LoRA state with different values
        new_lora_state = {}
        for name, tensor in original_lora.items():
            new_lora_state[name] = torch.randn_like(tensor)
        
        # Track which parameters get updated
        updated_params = []
        
        def mock_copy(self, tensor):
            updated_params.append((self._name, tensor))
            self._original_data = tensor
        
        # Mock parameter data copying
        for name, param in model.named_parameters():
            param._name = name
            param.data.copy_ = lambda tensor, self=param: mock_copy(self, tensor)
        
        # Load the new LoRA state
        load_lora_state(model, new_lora_state)
        
        # Verify only LoRA parameters were updated
        updated_names = {name for name, _ in updated_params}
        expected_names = set(new_lora_state.keys())
        assert updated_names == expected_names
        
        # Verify no base model parameters were updated
        for name, _ in updated_params:
            assert 'lora_' in name


class TestLoRAStateSwapping:
    """Test LoRA state swapping functionality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model with LoRA-like structure for testing."""
        class SimpleLoRAModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate base layer (frozen)
                self.base_layer = torch.nn.Linear(10, 10)
                self.base_layer.requires_grad_(False)
                
                # Simulate LoRA adapters (trainable)
                self.lora_A = torch.nn.Parameter(torch.randn(4, 10))
                self.lora_B = torch.nn.Parameter(torch.randn(10, 4))
                
            def forward(self, x):
                # Base transformation
                base_out = self.base_layer(x)
                # LoRA adaptation: x @ lora_A.T @ lora_B.T
                lora_out = x @ self.lora_A.T @ self.lora_B.T
                return base_out + lora_out
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                """Override to include lora_ prefix in names."""
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if 'lora_' not in name and (name == 'lora_A' or name == 'lora_B'):
                        name = f'lora_{name}'
                    yield name, param
        
        return SimpleLoRAModel()
    
    def test_state_swapping_changes_model_behavior(self, simple_model):
        """Test that swapping LoRA states changes model behavior."""
        # Generate test input
        test_input = torch.randn(5, 10)
        
        # Get initial output
        initial_output = simple_model(test_input)
        
        # Save current LoRA state
        current_state = save_lora_state(simple_model)
        
        # Create and load different LoRA state
        new_state = {}
        for name, param in simple_model.named_parameters():
            if 'lora_' in name:
                new_state[name] = torch.randn_like(param.data)
        
        load_lora_state(simple_model, new_state)
        
        # Get output with new state
        new_output = simple_model(test_input)
        
        # Outputs should be different
        assert not torch.allclose(initial_output, new_output, atol=1e-6)
        
        # Load original state back
        load_lora_state(simple_model, current_state)
        
        # Output should match initial
        restored_output = simple_model(test_input)
        assert torch.allclose(initial_output, restored_output, atol=1e-8)


class TestLoRAEMAUpdates:
    """Test LoRA-only EMA updates."""
    
    @pytest.fixture
    def twin_models(self):
        """Create two identical models for EMA testing."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_weight = torch.nn.Parameter(torch.randn(5, 5))
                self.base_weight.requires_grad_(False)  # Simulate frozen base
                self.lora_A_weight = torch.nn.Parameter(torch.randn(2, 5))
                self.lora_B_weight = torch.nn.Parameter(torch.randn(5, 2))
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                """Add lora_ prefix to LoRA parameters."""
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if name in ['lora_A_weight', 'lora_B_weight']:
                        name = f'lora_{name}'
                    yield name, param
        
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Make them identical initially
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
                p2.data.copy_(p1.data)
        
        return model1, model2
    
    def test_lora_ema_updates_only_lora_params(self, twin_models):
        """Test that LoRA EMA only updates LoRA parameters."""
        target_model, source_model = twin_models
        
        # Store initial base model parameters
        initial_base_params = {}
        for name, param in target_model.named_parameters():
            if 'lora_' not in name:
                initial_base_params[name] = param.data.clone()
        
        # Modify source model's LoRA parameters
        with torch.no_grad():
            for name, param in source_model.named_parameters():
                if 'lora_' in name:
                    param.data.add_(torch.randn_like(param.data))
        
        # Perform EMA update
        decay = 0.9
        update_lora_ema(target_model, source_model, decay)
        
        # Base model parameters should be unchanged
        for name, param in target_model.named_parameters():
            if 'lora_' not in name:
                assert torch.equal(param.data, initial_base_params[name])
        
        # LoRA parameters should be updated
        for (target_name, target_param), (source_name, source_param) in zip(
            target_model.named_parameters(), source_model.named_parameters()
        ):
            if 'lora_' in target_name:
                # Should not be identical to source (due to EMA)
                assert not torch.equal(target_param.data, source_param.data)
                # Should not be identical to original (should have changed)
                # This is harder to test without storing original values


class TestMemoryUsage:
    """Test memory usage comparisons."""
    
    def test_lora_state_dict_much_smaller_than_full_model(self):
        """Test that LoRA state dict uses much less memory than full model copy."""
        # Create a mock model with realistic parameter sizes
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate large base model (e.g., 1B parameters)
                self.large_base = torch.nn.Parameter(torch.randn(10000, 10000))
                self.large_base.requires_grad_(False)
                
                # Simulate small LoRA adapters
                self.small_lora_A = torch.nn.Parameter(torch.randn(8, 100))
                self.small_lora_B = torch.nn.Parameter(torch.randn(100, 8))
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if 'small_lora' in name:
                        name = f'lora_{name}'
                    yield name, param
        
        model = MockModel()
        
        # Calculate memory usage of full model copy
        full_copy = copy.deepcopy(model)
        full_copy_params = sum(p.numel() for p in full_copy.parameters())
        
        # Calculate memory usage of LoRA state dict
        lora_state = save_lora_state(model)
        lora_params = sum(tensor.numel() for tensor in lora_state.values())
        
        # LoRA state should be much smaller
        memory_ratio = lora_params / full_copy_params
        assert memory_ratio < 0.01  # LoRA should be < 1% of full model
        
        print(f"Memory usage - Full model: {full_copy_params:,} parameters")
        print(f"Memory usage - LoRA state: {lora_params:,} parameters")
        print(f"Memory ratio: {memory_ratio:.4f} ({memory_ratio*100:.2f}%)")


class TestIntegration:
    """Integration tests for complete LoRA state management workflow."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for realistic memory testing")
    def test_memory_efficient_training_simulation(self):
        """Test a simplified training loop using LoRA state management."""
        # This test simulates the training loop optimization
        # Skip actual model loading, use simple mock
        
        class TrainingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(100, 100)
                self.base.requires_grad_(False)
                self.lora_A = torch.nn.Parameter(torch.randn(8, 100))
                self.lora_B = torch.nn.Parameter(torch.randn(100, 8))
                
            def forward(self, x):
                # Base transformation + LoRA adaptation
                base_out = self.base(x)
                lora_out = x @ self.lora_A.T @ self.lora_B.T
                return base_out + lora_out
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if name in ['lora_A', 'lora_B']:
                        name = f'lora_{name}'
                    yield name, param
        
        # Setup
        current_model = TrainingModel()
        old_lora_state = save_lora_state(current_model)
        
        # Simulate training step that modifies current model
        with torch.no_grad():
            for name, param in current_model.named_parameters():
                if 'lora_' in name:
                    param.data.add_(0.01 * torch.randn_like(param.data))
        
        # Simulate needing old model for PPO ratios
        current_lora_state = save_lora_state(current_model)  # Save current
        load_lora_state(current_model, old_lora_state)       # Switch to old
        
        # ... Here we would compute PPO ratios using "old" model ...
        test_input = torch.randn(10, 100)
        old_output = current_model(test_input)
        
        # Switch back to current
        load_lora_state(current_model, current_lora_state)
        new_output = current_model(test_input)
        
        # Outputs should be different (proving state swapping works)
        assert not torch.allclose(old_output, new_output, atol=1e-6)
        
        # Update old state using EMA
        update_lora_ema(current_model, current_model, decay=0.95)  # Self-update for demo
        
        # Test passes if no errors occur
        assert True


if __name__ == "__main__":
    # Run specific tests for development
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr) 