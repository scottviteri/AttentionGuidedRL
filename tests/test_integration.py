"""
Integration tests for the complete Attention-Guided RL pipeline.

These tests verify that all components work together correctly:
- Data pipeline (functional toolz-based processing)
- Embedding extraction and similarity computation  
- Trajectory generation with vector queries
- PPO training with GRPO batching
- Model checkpointing and loading
"""

import pytest
import torch
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data import KVPair, QKVSelection, create_kv_stream
from src.training import RawTrajectory, build_trajectory_from_raw, train_step
from src.embeddings import register_embedding_hook, compute_similarity, extract_embeddings
from src.model import apply_lora_adapter, save_checkpoint, load_checkpoint
from src.main import generate_trajectory, compute_trajectory_rewards
from src.config import CONFIG


class TestDataPipelineIntegration:
    """Test the complete data pipeline integration."""
    
    def test_wikipedia_to_kvpair_pipeline(self, gpt2_model, gpt2_tokenizer, test_config_factory):
        """Test the complete Wikipedia data pipeline."""
        from src.data import wikipedia_kv_stream
        
        # Create and set test configuration
        test_config = test_config_factory()
        CONFIG.set_config(test_config)
        
        # Mock embedding function
        def mock_embedding_fn(tokens):
            batch_size = tokens.shape[0]
            device = tokens.device  # Use same device as input tokens
            return torch.randn(batch_size, gpt2_model.config.n_embd, device=device)
        
        # Mock Wikipedia articles - need to be long enough to pass filters
        mock_articles = [
            {"text": "This is a test article with enough content. " * 200, "title": "Test", "id": "1"},
            {"text": "Another test article with sufficient length. " * 200, "title": "Test2", "id": "2"}
        ]
        
        # Patch the wikipedia_articles function to return our mock articles
        def mock_wikipedia_articles():
            return iter(mock_articles)
        
        with patch('src.data.wikipedia_articles', mock_wikipedia_articles):
            # Create KV stream (this takes a while due to processing)
            kv_stream = wikipedia_kv_stream(
                batch_size=1,
                tokenizer=gpt2_tokenizer,
                embedding_fn=mock_embedding_fn
            )
            
            # Get first KV pair
            first_kv_pair = next(kv_stream)
            
            # Verify structure using CONFIG values
            assert isinstance(first_kv_pair, KVPair)
            assert first_kv_pair.key_tokens.shape == (1, CONFIG.tokens_per_key)
            assert first_kv_pair.value_tokens.shape == (1, CONFIG.tokens_per_value)
            assert first_kv_pair.key_embedding.shape == (1, gpt2_model.config.n_embd)
            assert len(first_kv_pair.key_text) == 1
            assert len(first_kv_pair.value_text) == 1
    
    def test_twenty_questions_integration(self, gpt2_tokenizer):
        """Test twenty questions dataset integration."""
        # Mock the twenty questions dataset
        mock_dataset = {
            'questions': [
                'Is it larger than a breadbox?',
                'Is it a living thing?',
                'Can you hold it in your hand?',
                'Is it man-made?',
                'Is it electronic?',
                'Is it used for entertainment?',
                'Is it found indoors?',
                'Is it expensive?',
                'Is it colorful?',
                'Is it edible?'
            ],
            'data': [
                {'answers': ['YES', 'NO', 'YES', 'YES', 'NO', 'YES', 'YES', 'NO', 'YES', 'NO']},
                {'answers': ['NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'YES', 'YES', 'NO', 'YES']}
            ]
        }
        
        def mock_embedding_fn(tokens):
            device = tokens.device
            return torch.randn(tokens.shape[0], 768, device=device)
        
        with patch('src.data.load_twenty_questions', return_value=mock_dataset):
            from src.data import twenty_questions_kv_stream
            
            kv_stream = twenty_questions_kv_stream(
                batch_size=1,
                tokenizer=gpt2_tokenizer,
                embedding_fn=mock_embedding_fn
            )
            
            # Get first KV pair
            first_kv_pair = next(kv_stream)
            
            # Verify structure
            assert isinstance(first_kv_pair, KVPair)
            assert first_kv_pair.key_text[0] in mock_dataset['questions']
            assert first_kv_pair.value_text[0] in ['YES', 'NO']


class TestEmbeddingIntegration:
    """Test embedding extraction and similarity computation integration."""
    
    def test_embedding_hook_and_extraction_flow(self, gpt2_model, gpt2_tokenizer):
        """Test the complete embedding extraction flow."""
        with patch("src.config.CONFIG.model_type", 'gpt2'):
            # Register hooks for both query and key embeddings
            query_dict, query_hook_remover = register_embedding_hook(gpt2_model, embed_type="query")
            key_dict, key_hook_remover = register_embedding_hook(gpt2_model, embed_type="key")
            
            try:
                # Test data
                test_texts = ["Hello world", "How are you?"]
                tokens = gpt2_tokenizer(test_texts, return_tensors="pt", padding=True)
                
                # Extract embeddings
                query_embeddings = extract_embeddings(gpt2_model, tokens.input_ids, query_dict)
                key_embeddings = extract_embeddings(gpt2_model, tokens.input_ids, key_dict, requires_grad=False)
                
                # Verify shapes
                assert query_embeddings.shape == (2, gpt2_model.config.n_embd)
                assert key_embeddings.shape == (2, gpt2_model.config.n_embd)
                
                # Test similarity computation
                key_embeddings_3d = key_embeddings.unsqueeze(1)  # [batch, 1, hidden]
                
                # Get attention parameters for the model
                from src.embeddings import get_attention_params
                num_heads, num_groups, head_dim = get_attention_params(gpt2_model)
                
                similarities = compute_similarity(query_embeddings, key_embeddings_3d, num_heads, num_groups, head_dim)
                
                # Verify similarity output
                assert similarities.shape == (2, 1)
                
                # Verify it's a proper log probability distribution (LogSumExp should be ≈ 0)
                log_sum_exp = torch.logsumexp(similarities, dim=1)
                assert torch.allclose(log_sum_exp, torch.zeros(2, device=similarities.device), atol=1e-5)
                
            finally:
                query_hook_remover()
                key_hook_remover()


class TestTrajectoryIntegration:
    """Test trajectory generation and training integration."""
    
    def test_trajectory_generation_flow(self, gpt2_model, gpt2_tokenizer, test_config_factory):
        """Test trajectory generation with real models."""
        # Note: MODEL_TYPE was replaced with CONFIG.model_type
        with patch('src.config.CONFIG.model_type', 'gpt2'):
            # Create adapter model
            adapter_model = apply_lora_adapter(gpt2_model)
            
            # Create and set test configuration
            test_config = test_config_factory()
            CONFIG.set_config(test_config)
            
            # Create test KV pairs
            device = next(adapter_model.parameters()).device
            available_qkv_steps = []
            
            for i in range(CONFIG.num_kv_pairs):
                kv_pair = KVPair(
                    key_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_key), device=device),
                    value_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_value), device=device), 
                    key_embedding=torch.randn(1, gpt2_model.config.n_embd, device=device),
                    key_text=[f"test key {i}"],
                    value_text=[f"test value {i}"]
                )
                available_qkv_steps.append(kv_pair)
            
            # Generate trajectory
            context_tokens = torch.zeros((1, 5), dtype=torch.long, device=device)
            
            raw_trajectory = generate_trajectory(
                context_tokens=context_tokens,
                adapter_model=adapter_model,
                tokenizer=gpt2_tokenizer,
                available_qkv_steps=available_qkv_steps,
                batch_size=1,
                config=test_config,
                verbose=False
            )
            
            # Verify trajectory structure
            assert isinstance(raw_trajectory, RawTrajectory)
            assert len(raw_trajectory.qkv_steps) == CONFIG.num_kv_pairs
            assert raw_trajectory.all_key_embeddings.shape == (1, CONFIG.num_kv_pairs, gpt2_model.config.n_embd)
            
            # Test trajectory reward computation
            trajectory, adapter_logprobs, ref_logprobs = compute_trajectory_rewards(
                raw_trajectory,
                adapter_model,
                gpt2_model,  # Use base model as reference
                context_tokens,
                tokenizer=gpt2_tokenizer,
                verbose=False
            )
            
            # Verify computed trajectory
            assert trajectory.rewards.shape == (1, CONFIG.num_kv_pairs)
            assert trajectory.avg_reward.shape == (1,)
            assert adapter_logprobs.shape == (1, CONFIG.num_kv_pairs)
            assert ref_logprobs.shape == (1, CONFIG.num_kv_pairs)


class TestTrainingIntegration:
    """Test the complete training pipeline integration."""
    
    def test_complete_training_step_flow(self, gpt2_model, gpt2_tokenizer, test_config_factory):
        """Test a complete training step with real models."""
        with patch("src.config.CONFIG.model_type", 'gpt2'):
            # Create models
            adapter_model = apply_lora_adapter(gpt2_model)
            ref_model = gpt2_model
            old_model = apply_lora_adapter(gpt2_model)  # Separate old model
            
            # Create and set test configuration
            test_config = test_config_factory()
            CONFIG.set_config(test_config)
            
            # Create optimizer
            optimizer = torch.optim.Adam(adapter_model.parameters(), lr=1e-4)
            
            # Create a simple trajectory
            device = next(adapter_model.parameters()).device
            
            # Build trajectory components
            qkv_data = KVPair(
                key_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_key), device=device),
                value_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_value), device=device),
                key_embedding=torch.randn(1, gpt2_model.config.n_embd, device=device),
                key_text=["test key"],
                value_text=["test value"]
            )
                
            qkv_step = QKVSelection(
                data=qkv_data,
                query_embedding=torch.randn(1, gpt2_model.config.n_embd, device=device),
                similarity_scores=torch.randn(1, 5, device=device),
                selected_idx=torch.tensor([0], device=device),
                available_mask=torch.zeros(1, 5, device=device)
            )
                
            # Create raw trajectory
            all_key_embeddings = torch.randn(1, 5, gpt2_model.config.n_embd, device=device)
            raw_traj = RawTrajectory(qkv_steps=[qkv_step], all_key_embeddings=all_key_embeddings)
                
            # Build complete trajectory with rewards
            rewards = torch.tensor([[0.5]], device=device)
            avg_reward = rewards.mean(dim=1)
            trajectory = build_trajectory_from_raw(raw_traj, rewards, avg_reward)
                
            # Reward stats
            reward_stats = {"mean": 0.0, "std": 1.0, "count": 1}
                
            # Perform training step
            total_loss, policy_loss, kl_loss, avg_clipping_ratio = train_step(
                trajectory=trajectory,
                adapter_model=adapter_model,
                ref_model=ref_model,
                old_model=old_model,
                optimizer=optimizer,
                reward_stats=reward_stats,
                kl_penalty_coef=0.01,
                verbose=False,
                tokenizer=gpt2_tokenizer
            )
                
            # Verify training step results
            assert isinstance(total_loss, float)
            assert isinstance(policy_loss, float)
            assert isinstance(kl_loss, float)
            assert isinstance(avg_clipping_ratio, float)
            assert not torch.isnan(torch.tensor(total_loss))


class TestCheckpointIntegration:
    """Test checkpoint saving and loading integration."""
    
    def test_checkpoint_persistence_flow(self, gpt2_model):
        """Test the complete checkpoint save/load flow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            
            with patch('src.config.CONFIG.checkpoint_dir', str(checkpoint_dir)):
                with patch('src.config.CONFIG.model_type', 'gpt2'):
                    # Create adapter model
                    adapter_model = apply_lora_adapter(gpt2_model)
                    
                    # Store initial parameter state
                    initial_params = {}
                    for name, param in adapter_model.named_parameters():
                        if param.requires_grad:
                            initial_params[name] = param.data.clone()
                    
                    # Modify parameters to simulate training
                    for name, param in adapter_model.named_parameters():
                        if param.requires_grad:
                            param.data.add_(torch.randn_like(param.data) * 0.01)
                    
                    # Save checkpoint
                    save_checkpoint(adapter_model, "test_checkpoint")
                    
                    # Verify checkpoint file exists
                    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
                    assert len(checkpoint_files) > 0
                    
                    # Create fresh model and load checkpoint
                    fresh_model = apply_lora_adapter(gpt2_model)
                    success = load_checkpoint(fresh_model, "test_checkpoint")
                    assert success
                    
                    # Verify parameters match (not initial state)
                    for name, param in fresh_model.named_parameters():
                        if param.requires_grad and name in initial_params:
                            # Should NOT match initial state (was modified before saving)
                            assert not torch.allclose(param.data, initial_params[name], atol=1e-6)


class TestEndToEndIntegration:
    """Complete end-to-end integration tests."""
    
    def test_minimal_training_pipeline(self, gpt2_model, gpt2_tokenizer, test_config_factory):
        """Test a minimal but complete training pipeline."""
        with patch("src.config.CONFIG.model_type", 'gpt2'):
            # Setup models
            adapter_model = apply_lora_adapter(gpt2_model)
            ref_model = gpt2_model
            old_model = apply_lora_adapter(gpt2_model)
            
            # Create and set test configuration
            test_config = test_config_factory()
            CONFIG.set_config(test_config)
            
            # Register embedding hooks
            query_dict, query_hook_remover = register_embedding_hook(adapter_model, embed_type="query")
            key_dict, key_hook_remover = register_embedding_hook(adapter_model, embed_type="key")
            
            try:
                # Mock data pipeline
                def mock_embedding_fn(tokens):
                    return torch.randn(tokens.shape[0], gpt2_model.config.n_embd)
                
                # Create mock KV stream
                mock_kv_pairs = []
                device = next(adapter_model.parameters()).device
                
                for i in range(CONFIG.num_kv_pairs):
                    kv_pair = KVPair(
                        key_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_key), device=device),
                        value_tokens=torch.randint(0, 1000, (1, CONFIG.tokens_per_value), device=device),
                        key_embedding=torch.randn(1, gpt2_model.config.n_embd, device=device),
                        key_text=[f"mock key {i}"],
                        value_text=[f"mock value {i}"]
                    )
                    mock_kv_pairs.append(kv_pair)
                
                # Generate trajectory
                context_tokens = torch.zeros((1, 3), dtype=torch.long, device=device)
                
                raw_trajectory = generate_trajectory(
                    context_tokens=context_tokens,
                    adapter_model=adapter_model,
                    tokenizer=gpt2_tokenizer,
                    available_qkv_steps=mock_kv_pairs,
                    batch_size=1,
                    config=test_config
                )
                
                # Compute rewards
                trajectory, _, _ = compute_trajectory_rewards(
                    raw_trajectory,
                    adapter_model,
                    ref_model,
                    context_tokens,
                    tokenizer=gpt2_tokenizer
                )
                
                # Training step
                optimizer = torch.optim.Adam(adapter_model.parameters(), lr=1e-4)
                reward_stats = {"mean": 0.0, "std": 1.0, "count": 1}
                
                total_loss, policy_loss, kl_loss, clipping_ratio = train_step(
                    trajectory=trajectory,
                    adapter_model=adapter_model,
                    ref_model=ref_model,
                    old_model=old_model,
                    optimizer=optimizer,
                    reward_stats=reward_stats,
                    kl_penalty_coef=0.01,
                    tokenizer=gpt2_tokenizer
                )
                
                # Verify pipeline completed successfully
                assert isinstance(total_loss, float)
                assert not torch.isnan(torch.tensor(total_loss))
                print(f"✅ End-to-end pipeline completed: loss={total_loss:.4f}")
                
            finally:
                query_hook_remover()
                key_hook_remover()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 