"""
Training module for the Attention-Guided RL project.

Contains the main training loop and related functions for trajectory generation and optimization.
"""

import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass

from src.data import QKVStep
from src.embeddings import extract_embeddings
from src.config import (
    DEVICE,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    KEY_PREFIX,
    VALUE_PREFIX,
    GRADIENT_CLIP_NORM,
    KL_PENALTY_COEFFICIENT,
    QUERY_VEC_TOKEN,
    NUM_KV_PAIRS,
    INITIAL_PROMPT,
    GAMMA,
    GAE_LAMBDA,
    USE_GRPO_BASELINE
)



@dataclass
class Trajectory:
    """
    A trajectory consisting of query-key-value steps and optional rewards.
    
    Attributes:
        qkv_steps: List of QKVStep objects selected during trajectory
        rewards: Optional tensor of rewards for each step [batch_size, num_steps]
        avg_reward: Average reward across steps [batch_size]
        all_key_embeddings: All available key embeddings for this trajectory [batch_size, num_keys, embedding_dim]
    """
    qkv_steps: List[QKVStep]
    rewards: Optional[torch.Tensor] = None
    avg_reward: Optional[torch.Tensor] = None
    all_key_embeddings: Optional[torch.Tensor] = None


def calculate_conditional_log_prob(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    context: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the conditional log probability of generating tokens given context.
    
    Args:
        model: The language model
        tokens: The tokens to evaluate [batch_size, seq_length]
        context: The context tokens [batch_size, context_length]
        
    Returns:
        torch.Tensor: Log probabilities for each batch item [batch_size]
    """
    # Combine context and tokens
    full_sequence = torch.cat([context, tokens], dim=1)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits
    
    # Extract logits for the token positions
    # We need logits from positions [context_length-1] to [context_length + seq_length - 2]
    # because we predict the next token at each position
    context_length = context.shape[1]
    token_logits = logits[:, context_length-1:context_length + tokens.shape[1] - 1, :]
    
    # Convert to log probabilities
    log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
    
    # Get the log probabilities of the actual tokens
    # tokens shape: [batch_size, seq_length]
    # log_probs shape: [batch_size, seq_length, vocab_size]
    # We need to gather the log probs of the actual tokens
    token_log_probs = torch.gather(log_probs, dim=2, index=tokens.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities across the sequence for each batch item
    return token_log_probs.sum(dim=1)


def generate_query_vector(
    model: torch.nn.Module,
    tokenizer: Any,
    context_tokens: torch.Tensor,
    layer_idx: int = -2
) -> torch.Tensor:
    """
    Generate a single query vector by appending QUERY_VEC_TOKEN and extracting query embeddings.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        context_tokens: Context tokens [batch_size, seq_len]
        layer_idx: Which layer to extract from (default: -2 for second-to-last)
        
    Returns:
        torch.Tensor: Query vector [batch_size, query_dim]
    """
    from src.config import QUERY_VEC_TOKEN
    from src.embeddings import register_embedding_hook, extract_embeddings
    
    # Get device from context tokens
    device = context_tokens.device
    batch_size = context_tokens.shape[0]
    
    # Register hook for query embeddings from specified layer
    embeddings_dict, hook_remover = register_embedding_hook(model, embed_type="query", layer_idx=layer_idx)
    
    try:
        # Tokenize the QUERY_VEC_TOKEN
        query_vec_token_ids = tokenizer(
            [QUERY_VEC_TOKEN] * batch_size,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(device)
        
        # Append QUERY_VEC_TOKEN to context
        input_tokens = torch.cat([context_tokens, query_vec_token_ids], dim=1)
        
        # Use extract_embeddings which handles the forward pass and extraction
        # This will give us the query embeddings from the attention layer
        # Set requires_grad=True for query embeddings during training
        query_vectors = extract_embeddings(model, input_tokens, embeddings_dict, requires_grad=True)
        
        return query_vectors
        
    finally:
        # Always remove the hook
        hook_remover()





def compute_trajectory_rewards(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    context_tokens: torch.Tensor,
    tokenizer: Any = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rewards for all query-key-value steps in a trajectory.
    
    Args:
        trajectory: The trajectory containing QKVStep objects
        adapter_model: The model with LoRA adapter
        base_model: The base model without LoRA
        context_tokens: Initial context tokens [batch_size, context_length]
        tokenizer: The tokenizer for processing text
        verbose: Flag to enable verbose logging
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            rewards: Rewards for each step [batch_size, num_steps]
            adapter_log_probs: Log probabilities from adapter model [batch_size, num_steps]
            baseline_log_probs: Log probabilities from baseline model [batch_size, num_steps]
    """
    batch_size = context_tokens.shape[0]
    num_steps = len(trajectory.qkv_steps)
    
    if verbose:
        print("\n=== Computing Trajectory Rewards ===")
        print(f"Batch size: {batch_size}")
        print(f"Number of steps: {num_steps}")
    
    # Ensure context_tokens is on the correct device
    device = context_tokens.device
    
    # Initialize tensors
    rewards = torch.zeros((batch_size, num_steps), device=device)
    adapter_log_probs = torch.zeros((batch_size, num_steps), device=device)
    baseline_log_probs = torch.zeros((batch_size, num_steps), device=device)
    
    # Build context incrementally, including each step
    current_context = context_tokens
    
    for i, qkv_step in enumerate(trajectory.qkv_steps):
        if verbose:
            print(f"\n--- Reward Calculation for Step {i+1}/{num_steps} ---")
            
            # Display query first if available
            if qkv_step.query_text is not None:
                print(f"Query: {qkv_step.query_text[0]}")
            
            # Then display key and value
            print(f"Key: {qkv_step.key_text[0]}")
            print(f"Value: {qkv_step.value_text[0]}")
            print(f"Current context length: {current_context.shape[1]} tokens")
        
        # Get key and value tokens and ensure they're on the same device as context
        key_tokens = qkv_step.key_tokens.to(device)
        value_tokens = qkv_step.value_tokens.to(device)
        
        # Compute log prob with adapter model
        adapter_log_prob = calculate_conditional_log_prob(
            adapter_model, 
            value_tokens, 
            current_context
        )
        
        # Compute log prob with base model
        base_log_prob = calculate_conditional_log_prob(
            base_model, 
            value_tokens, 
            current_context
        )
        
        # Store log probabilities
        adapter_log_probs[:, i] = adapter_log_prob
        baseline_log_probs[:, i] = base_log_prob
        
        # Calculate reward as improvement over base model
        rewards[:, i] = adapter_log_prob - base_log_prob
        
        if verbose:
            print(f"Adapter model log prob: {adapter_log_prob[0].item():.4f}")
            print(f"Base model log prob: {base_log_prob[0].item():.4f}")
            print(f"Reward: {rewards[0, i].item():.4f}")
        
        # Update context for next iteration
        # Append query, key and value tokens to context, all on the same device
        if tokenizer:
            # Add prefixes if tokenizer is available
            batch_size = current_context.shape[0]
            key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            # Vector queries - no query tokens to add to context
            current_context = torch.cat([
                current_context,
                key_prefix_tokens,
                key_tokens, 
                value_prefix_tokens,
                value_tokens
            ], dim=1)
            
            # Display vector query indicator in verbose mode
            if verbose and qkv_step.query_text is not None and "<VECTOR_QUERY>" in qkv_step.query_text[0]:
                print(f"Using vector query")
        else:
            # Fallback for tests or when tokenizer is not available
            current_context = torch.cat([
                current_context, 
                key_tokens, 
                value_tokens
            ], dim=1)
    
    # Compute average reward
    avg_reward = rewards.mean(dim=1)
    
    if verbose:
        print("\n=== Trajectory Summary ===")
        print(f"Average reward: {avg_reward[0].item():.4f}")
        print(f"Average adapter log prob: {adapter_log_probs.mean().item():.4f}")
        print(f"Average baseline log prob: {baseline_log_probs.mean().item():.4f}")
    
    # Store rewards in the trajectory object
    trajectory.rewards = rewards
    trajectory.avg_reward = avg_reward
    
    return rewards, adapter_log_probs, baseline_log_probs


def update_reward_stats(
    reward_stats: Dict[str, float],
    rewards: torch.Tensor
) -> Dict[str, float]:
    """
    Update the reward statistics.
    
    Args:
        reward_stats: Current reward statistics
        rewards: New rewards to include (batch_avg rewards) [batch_size]
        
    Returns:
        Dict[str, float]: Updated reward statistics
    """
    # Convert to numpy for easier calculation
    rewards_np = rewards.detach().cpu().numpy()
    
    # Update count
    new_count = reward_stats["count"] + len(rewards_np)
    
    # Calculate new mean (online formula)
    new_mean = (reward_stats["mean"] * reward_stats["count"] + rewards_np.sum()) / new_count
    
    # Calculate new standard deviation
    # Using Welford's online algorithm for numerical stability
    if reward_stats["count"] == 0:
        new_std = rewards_np.std() if len(rewards_np) > 1 else 1.0
    else:
        old_m = reward_stats["mean"]
        old_s = reward_stats["std"] ** 2 * reward_stats["count"]
        new_m = new_mean
        new_s = old_s + ((rewards_np - old_m) * (rewards_np - new_m)).sum()
        new_std = (new_s / new_count) ** 0.5
    
    return {"mean": new_mean, "std": new_std, "count": new_count}


def filter_trajectories_grpo(trajectory: Trajectory) -> Optional[Trajectory]:
    """
    Filter a batch of trajectories based on GRPO baseline.
    Keeps only trajectories with positive advantages (better than batch average at each step).
    
    Args:
        trajectory: The batch of trajectories to filter
        
    Returns:
        Filtered trajectory, or None if no trajectories pass the filter
    """
    if trajectory.avg_reward is None:
        return trajectory
    
    # Get batch rewards
    batch_rewards = trajectory.rewards
    
    # If batch size is 1, always keep it
    if batch_rewards.shape[0] == 1:
        return trajectory
    
    # Compute advantages using GRPO baseline
    advantages, _ = compute_advantages(
        batch_rewards, 
        values=None, 
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        use_grpo_baseline=True
    )
    
    # Get batch indices where overall advantage is positive
    # Sum advantages across time steps for each batch element
    batch_advantages = advantages.sum(dim=1)
    batch_mask = batch_advantages > 0
    
    # If no batch elements have positive advantage, return None
    if not torch.any(batch_mask):
        return None
    
    # Create a new trajectory with only the filtered batch elements
    filtered_qkv_steps = []
    
    for qkv_step in trajectory.qkv_steps:
        # Ensure batch_mask is on the same device as tensors
        batch_mask_device = batch_mask.to(qkv_step.key_tokens.device)
        
        # Prepare the filtered attributes
        filtered_attributes = {
            "key_tokens": qkv_step.key_tokens[batch_mask_device],
            "value_tokens": qkv_step.value_tokens[batch_mask_device],
            "key_embedding": qkv_step.key_embedding[batch_mask_device],
            "key_text": [qkv_step.key_text[i] for i in range(len(qkv_step.key_text)) if batch_mask[i]],
            "value_text": [qkv_step.value_text[i] for i in range(len(qkv_step.value_text)) if batch_mask[i]],
        }
        
        # Include query attributes if present
        if qkv_step.query_text is not None:
            filtered_attributes["query_text"] = [qkv_step.query_text[i] for i in range(len(qkv_step.query_text)) if batch_mask[i]]
        
        if qkv_step.query_tokens is not None and qkv_step.query_tokens.numel() > 0:
            # For vector queries, query_tokens may be empty tensors with shape [batch_size, 0]
            if qkv_step.query_tokens.shape[1] > 0:
                filtered_attributes["query_tokens"] = qkv_step.query_tokens[batch_mask_device]
            else:
                # Handle empty query tokens for vector queries
                filtered_batch_size = batch_mask_device.sum().item()
                filtered_attributes["query_tokens"] = torch.empty((filtered_batch_size, 0), dtype=qkv_step.query_tokens.dtype, device=qkv_step.query_tokens.device)
        
        if qkv_step.query_embedding is not None:
            filtered_attributes["query_embedding"] = qkv_step.query_embedding[batch_mask_device]
            
        if hasattr(qkv_step, 'query_log_probs') and qkv_step.query_log_probs is not None:
            filtered_attributes["query_log_probs"] = qkv_step.query_log_probs[batch_mask_device]
            
        if hasattr(qkv_step, 'query_mean') and qkv_step.query_mean is not None:
            filtered_attributes["query_mean"] = qkv_step.query_mean[batch_mask_device]
        
        # Create the filtered QKVStep
        filtered_qkv_step = QKVStep(**filtered_attributes)
        
        # Copy over additional attributes that are not part of the dataclass
        if hasattr(qkv_step, 'similarity_scores') and qkv_step.similarity_scores is not None:
            filtered_qkv_step.similarity_scores = qkv_step.similarity_scores[batch_mask_device]
        
        if hasattr(qkv_step, 'selected_idx'):
            filtered_qkv_step.selected_idx = qkv_step.selected_idx
        
        # Filter available_key_embeddings for KL divergence computation
        if hasattr(qkv_step, 'available_key_embeddings') and qkv_step.available_key_embeddings is not None:
            filtered_qkv_step.available_key_embeddings = qkv_step.available_key_embeddings[batch_mask_device]
        
        filtered_qkv_steps.append(filtered_qkv_step)
    
    # Create new trajectory with filtered elements
    filtered_trajectory = Trajectory(qkv_steps=filtered_qkv_steps)
    
    # Copy over rewards, filtering to keep only selected batch elements
    if trajectory.rewards is not None:
        # Get the device from one of the tensors
        device = filtered_qkv_steps[0].key_tokens.device
        filtered_trajectory.rewards = trajectory.rewards[batch_mask.to(device)]
    
    if trajectory.avg_reward is not None:
        # Ensure avg_reward is on the same device
        device = filtered_qkv_steps[0].key_tokens.device
        filtered_trajectory.avg_reward = trajectory.avg_reward[batch_mask.to(device)]
    
    # Copy over all_key_embeddings, filtering to keep only selected batch elements
    if trajectory.all_key_embeddings is not None:
        device = filtered_qkv_steps[0].key_tokens.device
        filtered_trajectory.all_key_embeddings = trajectory.all_key_embeddings[batch_mask.to(device)]
    
    return filtered_trajectory


def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted returns (rewards-to-go) for each timestep.
    
    Args:
        rewards: Tensor of rewards [batch_size, num_steps]
        gamma: Discount factor
        
    Returns:
        torch.Tensor: Returns for each timestep [batch_size, num_steps]
    """
    batch_size, num_steps = rewards.shape
    returns = torch.zeros_like(rewards)
    
    # Compute returns backwards
    returns[:, -1] = rewards[:, -1]
    for t in reversed(range(num_steps - 1)):
        returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]
    
    return returns


def compute_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_grpo_baseline: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Generalized Advantage Estimation (GAE) or GRPO-style baseline.
    
    Args:
        rewards: Tensor of rewards [batch_size, num_steps]
        values: Tensor of value estimates [batch_size, num_steps] (optional)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        use_grpo_baseline: If True and values is None, use GRPO-style per-timestep batch average
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    returns = compute_returns(rewards, gamma)
    
    if values is None:
        if use_grpo_baseline:
            # GRPO-style: Use batch average return at each timestep as baseline
            # This automatically handles the step-dependent nature of returns
            baseline = returns.mean(dim=0, keepdim=True)  # [1, num_steps]
            advantages = returns - baseline  # Broadcasting subtracts same baseline from each batch element
            
            # Note: We don't normalize here because GRPO advantages are already zero-centered
            # at each timestep by construction (mean across batch is 0)
        else:
            # Original: Use average across all timesteps (high variance)
            advantages = returns - returns.mean(dim=1, keepdim=True)
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        # Compute TD residuals with value function
        batch_size, num_steps = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        # Append a zero for the last value (terminal state)
        values_extended = torch.cat([values, torch.zeros(batch_size, 1, device=values.device)], dim=1)
        
        # Compute advantages using GAE
        last_advantage = 0
        for t in reversed(range(num_steps)):
            delta = rewards[:, t] + gamma * values_extended[:, t + 1] - values[:, t]
            advantages[:, t] = delta + gamma * gae_lambda * last_advantage
            last_advantage = advantages[:, t]
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns





def compute_policy_loss(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    previous_model: Any,
    kl_penalty_coef: float,
    verbose: bool = False,
    gamma: float = 0.99,
    tokenizer: Any = None,
    embeddings_dict: Dict = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compute the policy gradient loss with KL penalty for vector queries.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter
        previous_model: The model state before update
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        gamma: Discount factor for returns
        tokenizer: Tokenizer for reconstructing context
        embeddings_dict: Embeddings dictionary (unused but kept for compatibility)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: 
            total_loss: The combined loss value
            policy_loss: The policy gradient component
            kl_loss: The KL divergence component
            positive_adv_percentage: Percentage of steps with positive advantages
    """
    # Determine device to use
    device = next(adapter_model.parameters()).device
    
    # Initialize as tensors to maintain gradient flow
    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    kl_loss = torch.tensor(0.0, device=device, requires_grad=True) 
    count = 0
    
    # Ensure trajectory has rewards
    if trajectory.rewards is None or trajectory.avg_reward is None:
        raise ValueError("Trajectory must have rewards computed before policy loss")
    
    # Import needed configs
    from src.config import USE_GRPO_BASELINE, GAE_LAMBDA, INITIAL_PROMPT, KEY_PREFIX, VALUE_PREFIX, USE_POSITIVE_ADVANTAGES_ONLY
    
    # Compute returns and advantages
    returns = compute_returns(trajectory.rewards, gamma)
    advantages, _ = compute_advantages(
        trajectory.rewards, 
        values=None, 
        gamma=gamma,
        gae_lambda=GAE_LAMBDA,
        use_grpo_baseline=USE_GRPO_BASELINE
    )
    
    # Calculate positive advantage percentage for logging
    positive_advantages = (advantages > 0).float()
    positive_adv_percentage = positive_advantages.mean().item() * 100.0  # Convert to percentage
    
    # Compute what the previous model would have generated
    # We'll do this by reconstructing the context at each step
    previous_query_means = []
    if tokenizer is not None:
        # Get batch size from first step
        batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
        
        # Initialize context with the initial prompt
        context_tokens = tokenizer(
            [INITIAL_PROMPT] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).input_ids.to(device)
        
        # Reconstruct context and generate previous means for each step
        for t, qkv_step in enumerate(trajectory.qkv_steps):
            # Generate query vector with previous model
            with torch.no_grad():
                prev_query_mean = generate_query_vector(
                    previous_model,
                    tokenizer,
                    context_tokens,
                    layer_idx=-2
                )
                previous_query_means.append(prev_query_mean)
            
            # Update context for next iteration (add key and value tokens)
            key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            context_tokens = torch.cat([
                context_tokens,
                key_prefix_tokens,
                qkv_step.key_tokens.to(device),
                value_prefix_tokens,
                qkv_step.value_tokens.to(device)
            ], dim=1)
    
    # Process each step in the trajectory
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        if hasattr(qkv_step, 'similarity_scores') and qkv_step.similarity_scores is not None:
            # IMPORTANT: Recompute query embeddings with current model to enable gradient flow
            if tokenizer is not None:
                # Reconstruct context up to this step
                batch_size = qkv_step.key_tokens.shape[0]
                context_tokens = tokenizer(
                    [INITIAL_PROMPT] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False
                ).input_ids.to(device)
                
                # Add all previous steps to context
                for prev_t in range(t):
                    prev_step = trajectory.qkv_steps[prev_t]
                    key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    
                    context_tokens = torch.cat([
                        context_tokens,
                        key_prefix_tokens,
                        prev_step.key_tokens.to(device),
                        value_prefix_tokens,
                        prev_step.value_tokens.to(device)
                    ], dim=1)
                
                # Generate query vector with current adapter model (with gradients)
                current_query_embeddings = generate_query_vector(
                    adapter_model,
                    tokenizer,
                    context_tokens,
                    layer_idx=-2
                )
                
                # Recompute similarities with available key embeddings
                if hasattr(qkv_step, 'available_key_embeddings') and qkv_step.available_key_embeddings is not None:
                    from src.embeddings import compute_similarity
                    available_key_embeddings = qkv_step.available_key_embeddings.to(device)
                    
                    # Compute new similarity scores with gradients
                    recomputed_similarities = compute_similarity(
                        current_query_embeddings.unsqueeze(1),  # Add key dimension
                        available_key_embeddings,
                        adapter_model
                    )
                    
                    # Use recomputed similarities for policy gradient
                    from src.config import TEMPERATURE
                    log_probs = F.log_softmax(recomputed_similarities / TEMPERATURE, dim=-1)
                else:
                    # Fallback to stored similarities if key embeddings not available
                    similarity_scores = qkv_step.similarity_scores.to(device)
                    from src.config import TEMPERATURE
                    log_probs = F.log_softmax(similarity_scores / TEMPERATURE, dim=-1)
            else:
                # No tokenizer, use stored similarities
                similarity_scores = qkv_step.similarity_scores.to(device)
                from src.config import TEMPERATURE
                log_probs = F.log_softmax(similarity_scores / TEMPERATURE, dim=-1)
            
            selected_idx = qkv_step.selected_idx if hasattr(qkv_step, 'selected_idx') else 0
            step_advantages = advantages[:, t].to(device)
            selected_log_probs = log_probs[:, selected_idx]  # Get log prob of selected action
            
            # Policy gradient: log_prob * advantage (positive reinforces good actions)
            if USE_POSITIVE_ADVANTAGES_ONLY:
                # Only positive advantages contribute - this implements step-level filtering
                # while preserving sequential context for KL computation
                effective_advantages = torch.clamp(step_advantages, min=0.0)
            else:
                effective_advantages = step_advantages
            
            # Add small epsilon to prevent zero gradients when all advantages are clamped
            if torch.all(effective_advantages == 0):
                effective_advantages = effective_advantages + 1e-8
            
            # Policy gradient term (positive for reinforcement)
            batch_policy_gradient = (selected_log_probs * effective_advantages).mean()
            policy_loss = policy_loss + batch_policy_gradient  # Accumulate positive gradient
            
            # Compute KL divergence between current and previous softmax policies
            # Fixed: Use per-step available key embeddings to match similarity_scores dimensions
            if previous_query_means and hasattr(qkv_step, 'available_key_embeddings') and qkv_step.available_key_embeddings is not None:
                # Recompute similarities with previous query vectors
                query_embeddings_prev = previous_query_means[t].to(device)
                available_key_embeddings = qkv_step.available_key_embeddings.to(device)
                
                # Import compute_similarity to handle normalization properly
                from src.embeddings import compute_similarity
                
                # Compute similarities with previous model's query
                prev_similarities = compute_similarity(
                    query_embeddings_prev.unsqueeze(1),  # Add key dimension
                    available_key_embeddings,
                    adapter_model  # Pass model for potential model-specific handling
                )
                prev_log_probs = F.log_softmax(prev_similarities / TEMPERATURE, dim=-1)
                
                # KL divergence between two categorical distributions
                kl_divergence = F.kl_div(log_probs, prev_log_probs, reduction='batchmean', log_target=True)
                kl_loss = kl_loss + kl_divergence  # Use tensor addition to maintain gradient
                
                if verbose and t == 0:  # Print info for first step
                    print(f"Softmax KL divergence: {kl_divergence.item():.4f}")
                    if USE_POSITIVE_ADVANTAGES_ONLY:
                        pos_ratio = (step_advantages > 0).float().mean()
                        print(f"Positive advantage ratio: {pos_ratio.item():.2%}")
            
            count += 1
        else:
            # No query information - skip
            if verbose:
                print("Skipping step - no similarity scores")
            continue
    
    # Return average loss if there were trajectories, otherwise zero
    if count > 0:
        avg_policy_gradient = policy_loss / count if isinstance(policy_loss, torch.Tensor) else torch.tensor(policy_loss / count, device=device)
        avg_kl_loss = kl_loss / count if isinstance(kl_loss, torch.Tensor) else torch.tensor(kl_loss / count, device=device)
        
        # Convert to loss: negate policy gradient (since we want to maximize expected reward)
        # and add KL penalty (since we want to minimize divergence)
        avg_policy_loss = -avg_policy_gradient  # Only sign flip happens here!
        kl_penalty_term = kl_penalty_coef * avg_kl_loss
        total_loss = avg_policy_loss + kl_penalty_term
        
        if verbose:
            print(f"\n=== Loss Components ===")
            print(f"Policy gradient (before negation): {avg_policy_gradient.item():.4f}")
            print(f"Policy loss (after negation): {avg_policy_loss.item():.4f}")
            print(f"KL divergence loss: {avg_kl_loss.item():.4f}")
            print(f"KL penalty coefficient: {kl_penalty_coef:.4f}")
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"  = {avg_policy_loss.item():.4f} + {kl_penalty_term.item():.4f}")
            print(f"=== End Loss Components ===\n")
            
        return total_loss, avg_policy_loss, avg_kl_loss, positive_adv_percentage
    else:
        # Create a small non-zero tensor to ensure gradients can flow
        small_tensor = torch.tensor(1e-8, device=device, requires_grad=True)
        return small_tensor, small_tensor, small_tensor, 0.0


def train_step(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    previous_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float,
    verbose: bool = False,
    tokenizer: Any = None,
    embeddings_dict: Dict = None
) -> Tuple[float, float, float, float]:
    """
    Perform a single training step.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter
        base_model: The base language model without LoRA
        previous_model: The model before update (for KL divergence)
        optimizer: The optimizer
        reward_stats: Reward statistics (for logging only with GRPO)
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        tokenizer: Tokenizer (needed for vector queries)
        embeddings_dict: Embeddings dictionary (needed for vector queries)
        
    Returns:
        Tuple[float, float, float, float]: 
            total_loss: Total loss value 
            positive_adv_percentage: Percentage of steps with positive advantages
            policy_loss: Policy gradient component of the loss
            kl_loss: KL divergence component of the loss
    """
    if verbose:
        print("\n=== Training Step ===")
        
        if trajectory.avg_reward is not None:
            batch_size = trajectory.avg_reward.shape[0]
            print(f"Input trajectory batch size: {batch_size}")
            print(f"Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")
            
            # Print individual batch element rewards
            if batch_size > 0:
                rewards = [f"{i}: {reward.item():.4f}" for i, reward in enumerate(trajectory.avg_reward)]
                print(f"Batch element rewards: {', '.join(rewards)}")
    
    # Filter trajectory based on GRPO baseline
    filtered_trajectory = filter_trajectories_grpo(trajectory)
    
    # Skip update if no batch elements meet the criteria
    if filtered_trajectory is None:
        if verbose:
            print("No batch elements have positive advantage. Skipping update.")
        return 0.0, 0.0, 0.0, 0.0
    
    # Get the number of batch elements that passed filtering
    original_batch_size = trajectory.avg_reward.shape[0]
    filtered_batch_size = filtered_trajectory.avg_reward.shape[0]
    
    if verbose:
        print(f"GRPO filtered batch size: {filtered_batch_size}/{original_batch_size}")
        print(f"Keeping trajectories with positive advantage (better than batch average)")
    
    # No trajectory-level filtering - we use all trajectories
    # The step-level filtering via USE_POSITIVE_ADVANTAGES_ONLY handles this
    
    if verbose:
        batch_size = trajectory.avg_reward.shape[0]
        print(f"Processing all {batch_size} trajectories")
        print(f"Step-level filtering via positive advantages is enabled")
    
    # Use original trajectory without filtering
    filtered_batch_size = trajectory.avg_reward.shape[0]
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Import the new parameters
    from src.config import GAMMA
    
    # Compute policy loss
    total_loss, policy_loss, kl_loss, positive_adv_percentage = compute_policy_loss(
        trajectory,  # Use original trajectory, not filtered
        adapter_model,
        previous_model,
        kl_penalty_coef,
        verbose=verbose,
        gamma=GAMMA,
        tokenizer=tokenizer,
        embeddings_dict=embeddings_dict
    )
    
    if verbose:
        print(f"Total loss: {total_loss.item():.4f}")
    
    # Backpropagate loss
    total_loss.backward()
    
    # Get gradient norm for logging
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), GRADIENT_CLIP_NORM)
    
    if verbose:
        print(f"Gradient norm (before clipping): {grad_norm:.4f}")
    
    # Update parameters
    optimizer.step()
    
    if verbose:
        print("Parameters updated.")
        print(f"=== Training Step Complete ===\n")
    
    return total_loss.item(), positive_adv_percentage, policy_loss, kl_loss 