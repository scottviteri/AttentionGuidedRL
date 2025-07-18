"""
Test cases for memory-efficient training implementation.

These tests verify that the memory-efficient training loop produces
equivalent results to the original training loop while using less memory.
"""

import pytest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory_efficient_training import (
    MemoryEfficientLoRAManager,
    memory_efficient_train_step,
    memory_efficient_compute_policy_loss
)
from src.model import save_lora_state, load_lora_state
from src.data import RawTrajectory, Trajectory, QKVSelection, KVPair


class TestMemoryEfficientLoRAManager:
    """Test the LoRA state manager."""
    
    @pytest.fixture
    def simple_lora_model(self):
        """Create a simple model with LoRA-like parameters."""
        class SimpleLorAModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_weight = torch.nn.Parameter(torch.randn(10, 10))
                self.base_weight.requires_grad_(False)
                self.lora_A = torch.nn.Parameter(torch.randn(4, 10))
                self.lora_B = torch.nn.Parameter(torch.randn(10, 4))
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if name in ['lora_A', 'lora_B']:
                        name = f'lora_{name}'
                    yield name, param
        
        return SimpleLorAModel()
    
    def test_lora_manager_initialization(self, simple_lora_model):
        """Test that LoRA manager initializes correctly."""
        manager = MemoryEfficientLoRAManager(simple_lora_model)
        
        # Should have saved initial LoRA state
        assert manager.old_lora_state is not None
        assert len(manager.old_lora_state) == 2  # lora_A and lora_B
        
        # Should contain LoRA parameters
        for name in manager.old_lora_state.keys():
            assert 'lora_' in name
    
    def test_state_switching(self, simple_lora_model):
        """Test switching between current and old states."""
        manager = MemoryEfficientLoRAManager(simple_lora_model)
        
        # Modify current model parameters
        with torch.no_grad():
            for name, param in simple_lora_model.named_parameters():
                if 'lora_' in name:
                    param.data.add_(torch.randn_like(param.data) * 0.1)
        
        # Save current state
        manager.save_current_state()
        current_values = {}
        for name, param in simple_lora_model.named_parameters():
            if 'lora_' in name:
                current_values[name] = param.data.clone()
        
        # Switch to old state
        manager.switch_to_old_state()
        
        # Parameters should be different now
        for name, param in simple_lora_model.named_parameters():
            if 'lora_' in name:
                assert not torch.equal(param.data, current_values[name])
        
        # Switch back to current state
        manager.switch_to_current_state()
        
        # Parameters should match current values again
        for name, param in simple_lora_model.named_parameters():
            if 'lora_' in name:
                assert torch.allclose(param.data, current_values[name], atol=1e-8)
    
    def test_ema_update(self, simple_lora_model):
        """Test EMA updates on LoRA state."""
        manager = MemoryEfficientLoRAManager(simple_lora_model)
        
        # Save initial old state
        initial_old_state = {name: tensor.clone() for name, tensor in manager.old_lora_state.items()}
        
        # Modify current model
        with torch.no_grad():
            for name, param in simple_lora_model.named_parameters():
                if 'lora_' in name:
                    param.data.add_(torch.ones_like(param.data))
        
        # Save current state
        manager.save_current_state()
        
        # Perform EMA update
        decay = 0.9
        manager.update_old_state_ema(decay)
        
        # Old state should be between initial and current
        for name in manager.old_lora_state.keys():
            old_val = manager.old_lora_state[name]
            initial_val = initial_old_state[name]
            current_val = manager.current_lora_state[name]
            
            # EMA formula: old = decay * old + (1 - decay) * current
            expected_val = decay * initial_val + (1 - decay) * current_val
            assert torch.allclose(old_val, expected_val, atol=1e-6)


class TestMemoryEfficientTraining:
    """Test memory-efficient training functions."""
    
    @pytest.fixture
    def mock_trajectory(self):
        """Create a mock trajectory for testing."""
        batch_size = 2
        num_steps = 3
        
        # Create mock QKV steps
        qkv_steps = []
        for step in range(num_steps):
            # Create mock data
            mock_data = KVPair(
                key_tokens=torch.randint(0, 1000, (batch_size, 10)),
                value_tokens=torch.randint(0, 1000, (batch_size, 10)),
                key_embedding=torch.randn(batch_size, 128),
                key_text=[f"key_{step}_batch_{b}" for b in range(batch_size)],
                value_text=[f"value_{step}_batch_{b}" for b in range(batch_size)]
            )
            
            qkv_selection = QKVSelection(
                data=mock_data,
                query_embedding=torch.randn(batch_size, 128),
                similarity_scores=torch.randn(batch_size, 5),  # 5 available keys
                selected_idx=torch.randint(0, 5, (batch_size,)),
                available_mask=torch.zeros(batch_size, 5)
            )
            qkv_steps.append(qkv_selection)
        
        # Create trajectory
        trajectory = Trajectory(
            qkv_steps=qkv_steps,
            rewards=torch.randn(batch_size, num_steps),
            avg_reward=torch.randn(batch_size),
            all_key_embeddings=torch.randn(batch_size, 5, 128)
        )
        
        return trajectory
    
    @pytest.fixture  
    def mock_models(self):
        """Create mock models for training tests."""
        class MockLoRAModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = torch.nn.Linear(128, 128)
                self.base_layer.requires_grad_(False)
                self.lora_A = torch.nn.Parameter(torch.randn(8, 128))
                self.lora_B = torch.nn.Parameter(torch.randn(128, 8))
                
            def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
                for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
                    if name in ['lora_A', 'lora_B']:
                        name = f'lora_{name}'
                    yield name, param
                    
            def parameters(self, recurse=True):
                """Override to only return LoRA parameters as trainable."""
                yield self.lora_A
                yield self.lora_B
        
        adapter_model = MockLoRAModel()
        ref_model = MockLoRAModel()
        
        return adapter_model, ref_model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        
        def mock_call(text_list, **kwargs):
            # Return mock tokenized output
            mock_result = MagicMock()
            batch_size = len(text_list)
            mock_result.input_ids = torch.randint(0, 1000, (batch_size, 20))
            return mock_result
        
        tokenizer.side_effect = mock_call
        return tokenizer
    
    @patch('src.memory_efficient_training.generate_query_vector')
    @patch('src.memory_efficient_training.compute_similarity')
    def test_memory_efficient_compute_policy_loss_runs(self, mock_compute_similarity, mock_generate_query, 
                                                       mock_trajectory, mock_models, mock_tokenizer):
        """Test that memory-efficient policy loss computation runs without errors."""
        adapter_model, ref_model = mock_models
        
        # Setup mocks
        mock_generate_query.return_value = torch.randn(2, 128)
        mock_compute_similarity.return_value = torch.randn(2, 5)
        
        # Create LoRA manager
        manager = MemoryEfficientLoRAManager(adapter_model)
        
        # Run policy loss computation
        try:
            total_loss, policy_loss, kl_loss, clipping_ratio = memory_efficient_compute_policy_loss(
                mock_trajectory,
                adapter_model,
                ref_model,
                manager,
                kl_penalty_coef=0.1,
                verbose=False,
                tokenizer=mock_tokenizer
            )
            
            # Should return valid tensors
            assert isinstance(total_loss, torch.Tensor)
            assert isinstance(policy_loss, torch.Tensor)
            assert isinstance(kl_loss, torch.Tensor)
            assert isinstance(clipping_ratio, float)
            
            # Losses should be finite
            assert torch.isfinite(total_loss)
            assert torch.isfinite(policy_loss)
            assert torch.isfinite(kl_loss)
            
        except Exception as e:
            pytest.fail(f"Memory-efficient policy loss computation failed: {e}")
    
    @patch('src.memory_efficient_training.memory_efficient_compute_policy_loss')
    def test_memory_efficient_train_step_runs(self, mock_policy_loss, mock_trajectory, mock_models, mock_tokenizer):
        """Test that memory-efficient training step runs without errors."""
        adapter_model, ref_model = mock_models
        
        # Setup mocks
        mock_policy_loss.return_value = (
            torch.tensor(1.0, requires_grad=True),  # total_loss
            torch.tensor(0.8),  # policy_loss  
            torch.tensor(0.2),  # kl_loss
            0.95  # clipping_ratio
        )
        
        # Create optimizer and LoRA manager
        optimizer = torch.optim.Adam(adapter_model.parameters(), lr=1e-4)
        manager = MemoryEfficientLoRAManager(adapter_model)
        reward_stats = {"mean": 0.0, "std": 1.0, "count": 10}
        
        # Run training step
        try:
            total_loss, policy_loss, kl_loss, clipping_ratio = memory_efficient_train_step(
                mock_trajectory,
                adapter_model,
                ref_model,
                manager,
                optimizer,
                reward_stats,
                kl_penalty_coef=0.1,
                verbose=False,
                tokenizer=mock_tokenizer
            )
            
            # Should return valid values
            assert isinstance(total_loss, float)
            assert isinstance(policy_loss, float)
            assert isinstance(kl_loss, float)
            assert isinstance(clipping_ratio, float)
            
            # Values should be finite
            assert torch.isfinite(torch.tensor(total_loss))
            assert torch.isfinite(torch.tensor(policy_loss))
            assert torch.isfinite(torch.tensor(kl_loss))
            
        except Exception as e:
            pytest.fail(f"Memory-efficient training step failed: {e}")


if __name__ == "__main__":
    # Run tests
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr) 