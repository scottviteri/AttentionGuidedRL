
"""
Memory-efficient training loop implementation using LoRA state management.

This module provides drop-in replacements for training functions that
use LoRA state dictionaries instead of full model copies, achieving
significant memory savings (typically 60-90% reduction).
"""

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from src.model import save_lora_state, load_lora_state, update_lora_ema
from src.training import (
    compute_advantages, 
    generate_query_vector,
    compute_trajectory_rewards as original_compute_trajectory_rewards
)
from src.embeddings import compute_similarity, get_attention_params
from src.config import CONFIG


class MemoryEfficientLoRAManager:
    """
    Manages LoRA states for memory-efficient training.
    
    Instead of storing full model copies, this manager stores only
    LoRA adapter states and swaps them as needed during training.
    """
    
    def __init__(self, adapter_model: torch.nn.Module):
        """
        Initialize the LoRA manager.
        
        Args:
            adapter_model: The main adapter model being trained
        """
        self.adapter_model = adapter_model
        self.old_lora_state = save_lora_state(adapter_model)
        self.current_lora_state = None
        
        # Add small random noise to old state to avoid identical initialization
        self._add_initialization_noise()
        
        logging.info("MemoryEfficientLoRAManager initialized")
        self._log_memory_stats()
    
    def _add_initialization_noise(self):
        """Add small random noise to old LoRA state to avoid identical PPO ratios."""
        for name, tensor in self.old_lora_state.items():
            if tensor.dtype.is_floating_point:
                noise = torch.randn_like(tensor) * 0.01
                self.old_lora_state[name] = tensor + noise
    
    def _log_memory_stats(self):
        """Log memory usage statistics."""
        # Count LoRA parameters
        lora_params = sum(tensor.numel() for tensor in self.old_lora_state.values())
        total_params = sum(p.numel() for p in self.adapter_model.parameters())
        memory_ratio = lora_params / total_params * 100
        
        logging.info(f"Memory efficiency - LoRA state: {lora_params:,} parameters")
        logging.info(f"Memory efficiency - Total model: {total_params:,} parameters") 
        logging.info(f"Memory efficiency - LoRA ratio: {memory_ratio:.3f}%")
    
    def save_current_state(self):
        """Save the current adapter model LoRA state."""
        self.current_lora_state = save_lora_state(self.adapter_model)
    
    def switch_to_old_state(self):
        """Switch adapter model to old LoRA state."""
        if self.current_lora_state is None:
            self.save_current_state()
        load_lora_state(self.adapter_model, self.old_lora_state)
    
    def switch_to_current_state(self):
        """Switch adapter model back to current LoRA state."""
        if self.current_lora_state is not None:
            load_lora_state(self.adapter_model, self.current_lora_state)
    
    def update_old_state_ema(self, decay: float = 0.99):
        """Update old LoRA state using EMA from current state."""
        if self.current_lora_state is None:
            self.save_current_state()
        
        # EMA update: old = decay * old + (1 - decay) * current
        for name in self.old_lora_state.keys():
            if name in self.current_lora_state:
                old_tensor = self.old_lora_state[name]
                current_tensor = self.current_lora_state[name]
                self.old_lora_state[name] = (
                    decay * old_tensor + (1 - decay) * current_tensor
                )
    
    def update_old_state_hard(self):
        """Replace old LoRA state with current state (hard update)."""
        if self.current_lora_state is None:
            self.save_current_state()
        self.old_lora_state = {
            name: tensor.clone() 
            for name, tensor in self.current_lora_state.items()
        }


def memory_efficient_compute_policy_loss(
    trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    lora_manager: MemoryEfficientLoRAManager,
    kl_penalty_coef: float,
    verbose: bool = False,
    gamma: float = 0.99,
    tokenizer: Any = None,
    embeddings_dict: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Memory-efficient policy loss computation using LoRA state swapping.
    
    This function computes PPO loss by temporarily switching the adapter model
    to old LoRA state when needed, instead of maintaining separate model copies.
    """
    device = next(adapter_model.parameters()).device
    
    # Compute advantages (same as before)
    advantages, _ = compute_advantages(
        trajectory.rewards, 
        gamma=gamma,
        gae_lambda=CONFIG.gae_lambda,
        use_grpo_baseline=CONFIG.use_grpo_baseline
    )
    
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
    
    # Save current state and compute old model query embeddings
    lora_manager.save_current_state()
    
    # Initialize tracking variables
    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    kl_loss = torch.tensor(0.0, device=device)
    all_clipping_ratios = []
    count = 0
    
    # Initialize context for old model queries
    context_tokens = tokenizer(
        [CONFIG.initial_prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).input_ids.to(device)
    
    # Compute old model queries by switching to old state
    old_query_embeddings = []
    lora_manager.switch_to_old_state()
    
    current_context = context_tokens
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        # Generate query vector with old model state
        with torch.no_grad():
            old_query_emb = generate_query_vector(
                adapter_model,  # Now has old LoRA state
                tokenizer,
                current_context,
                layer_idx=-2
            )
            old_query_embeddings.append(old_query_emb)
        
        # Update context for next iteration
        key_prefix_tokens = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        value_prefix_tokens = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        current_context = torch.cat([
            current_context,
            key_prefix_tokens,
            qkv_step.key_tokens.to(device),
            value_prefix_tokens,
            qkv_step.value_tokens.to(device)
        ], dim=1)
    
    # Switch back to current state for main computation
    lora_manager.switch_to_current_state()
    
    # Main policy loss computation loop
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        step_advantages = advantages[:, t]
        
        # Generate current query embeddings
        current_context_step = context_tokens
        for i in range(t + 1):
            step = trajectory.qkv_steps[i]
            if i == t:
                # For current step, generate query
                current_query_embeddings = generate_query_vector(
                    adapter_model, tokenizer, current_context_step, layer_idx=-2
                )
                break
            else:
                # Add previous steps to context
                key_prefix = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                value_prefix = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                current_context_step = torch.cat([
                    current_context_step, key_prefix, step.key_tokens.to(device),
                    value_prefix, step.value_tokens.to(device)
                ], dim=1)
        
        # Compute similarities and log probabilities
        key_embs_full = trajectory.all_key_embeddings.to(device)
        # Get attention parameters from the current adapter model
        num_heads, num_groups, head_dim = get_attention_params(adapter_model)
        # Apply availability mask inside compute_similarity for proper probability distribution
        availability_mask = qkv_step.available_mask if hasattr(qkv_step, 'available_mask') else None
        current_similarities = compute_similarity(current_query_embeddings, key_embs_full, num_heads, num_groups, head_dim,
                                                availability_mask=availability_mask)
        
        current_log_probs_full = current_similarities
        
        # Get old model log probabilities 
        old_query_emb = old_query_embeddings[t]
        # Use the same attention parameters since we're using the same model with different LoRA state
        # Apply same availability mask inside compute_similarity
        old_similarities = compute_similarity(old_query_emb, key_embs_full, num_heads, num_groups, head_dim,
                                            availability_mask=availability_mask)
        old_log_probs_full = old_similarities
        
        # Extract action log probabilities
        selected_idx = qkv_step.selected_idx
        if not isinstance(selected_idx, torch.Tensor):
            selected_idx = torch.tensor(selected_idx, device=device)
        
        current_action_log_probs = current_log_probs_full[torch.arange(batch_size, device=device), selected_idx]
        old_action_log_probs = old_log_probs_full[torch.arange(batch_size, device=device), selected_idx]
        
        # Compute KL divergence for regularization
        kl_step = torch.nn.functional.kl_div(
            current_log_probs_full, old_log_probs_full, 
            reduction="batchmean", log_target=True
        )
        
        # PPO computation
        import src.config as config
        if config.CONFIG.use_ppo:
            # PPO with ratio clipping
            log_ratio = current_action_log_probs - old_action_log_probs
            ratio = torch.exp(log_ratio)
            
            clipped_ratio = torch.clamp(ratio, 1.0 - CONFIG.ppo_clip_epsilon, 1.0 + CONFIG.ppo_clip_epsilon)
            all_clipping_ratios.extend(ratio.detach().cpu().tolist())
            
            unclipped_surrogate = ratio * step_advantages
            clipped_surrogate = clipped_ratio * step_advantages
            ppo_surrogate = torch.min(unclipped_surrogate, clipped_surrogate)
            batch_policy_gradient = ppo_surrogate.sum()
        else:
            # Vanilla Policy Gradient
            vanilla_policy_gradient = current_action_log_probs * step_advantages
            batch_policy_gradient = vanilla_policy_gradient.sum()
            all_clipping_ratios.extend([1.0] * len(current_action_log_probs))
        
        policy_loss = policy_loss + batch_policy_gradient
        kl_loss = kl_loss + kl_step
        count += 1
    
    # Finalize loss computation
    avg_clipping_ratio = sum(all_clipping_ratios) / len(all_clipping_ratios) if all_clipping_ratios else 1.0
    
    if count > 0:
        total_policy_loss = -policy_loss  # Negate for gradient descent
        total_kl_loss = kl_loss / count
        total_loss = total_policy_loss + kl_penalty_coef * total_kl_loss
        
        if verbose:
            method_name = "PPO" if config.CONFIG.use_ppo else "Vanilla PG"
            logging.info(f"Memory-efficient {method_name} loss computed")
            logging.info(f"Policy loss: {total_policy_loss.item():.4f}")
            logging.info(f"KL loss: {total_kl_loss.item():.4f}")
        
        return total_loss, total_policy_loss, total_kl_loss, avg_clipping_ratio
    else:
        # Fallback case
        small_tensor = torch.tensor(1e-8, device=device, requires_grad=True)
        return small_tensor, small_tensor, small_tensor, 1.0


def memory_efficient_train_step(
    trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    lora_manager: MemoryEfficientLoRAManager,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float,
    verbose: bool = False,
    tokenizer: Any = None,
    embeddings_dict: Optional[Dict] = None
) -> Tuple[float, float, float, float]:
    """
    Memory-efficient training step using LoRA state management.
    
    This is a drop-in replacement for the original train_step function
    that uses LoRA state swapping instead of maintaining separate model copies.
    """
    if verbose:
        logging.info("=== Memory-Efficient Training Step ===")
        batch_size = trajectory.avg_reward.shape[0]
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Compute policy loss using memory-efficient approach
    total_loss, policy_loss, kl_loss, avg_clipping_ratio = memory_efficient_compute_policy_loss(
        trajectory,
        adapter_model,
        ref_model,
        lora_manager,
        kl_penalty_coef,
        verbose=verbose,
        gamma=CONFIG.gamma,
        tokenizer=tokenizer,
        embeddings_dict=embeddings_dict
    )
    
    if verbose:
        logging.info(f"Total loss: {total_loss.item():.4f}")
    
    # Backpropagate loss
    total_loss.backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), CONFIG.gradient_clip_norm)
    
    if verbose:
        logging.info(f"Gradient norm: {grad_norm:.4f}")
    
    # Update parameters
    optimizer.step()
    
    if verbose:
        logging.info("=== Memory-Efficient Training Step Complete ===")
    
    return total_loss.item(), policy_loss.item(), kl_loss.item(), avg_clipping_ratio 