"""
Training module for the Attention-Guided RL project.

Contains functions for reinforcement learning training loop, reward calculation,
and policy optimization.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

from src.config import (
    DEVICE,
    TOKENS_PER_QUERY,
    TOKENS_PER_KEY,
    QUERY_PREFIX,
    VALUE_PREFIX,
    GRADIENT_CLIP_NORM,
    TEMPERATURE,
    TOP_P,
    KEY_PREFIX,
)
from src.model import create_model_copy
from src.data import QKVStep


@dataclass
class Trajectory:
    """
    A trajectory consisting of query-key-value steps and optional rewards.
    
    Attributes:
        qkv_steps: List of QKVStep objects selected during trajectory
        rewards: Optional tensor of rewards for each step [batch_size, num_steps]
        avg_reward: Average reward across steps [batch_size]
    """
    qkv_steps: List[QKVStep]
    rewards: Optional[torch.Tensor] = None
    avg_reward: Optional[torch.Tensor] = None


def calculate_conditional_log_prob(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    context: torch.Tensor
) -> torch.Tensor:
    """
    Calculate conditional log probability of tokens given context.
    
    Args:
        model: The language model
        tokens: Target tokens to evaluate [batch_size, seq_length]
        context: Context tokens [batch_size, context_length]
        
    Returns:
        torch.Tensor: Log probabilities [batch_size]
    """
    batch_size = tokens.shape[0]
    
    # Concatenate context and tokens to create the full sequence
    full_sequence = torch.cat([context, tokens], dim=1)
    
    # Forward pass to get logits (use the full sequence)
    with torch.no_grad():
        outputs = model(full_sequence)
    
    # Get logits for only the token positions (last part of the sequence)
    logits = outputs.logits[:, -tokens.shape[1]:, :]  # [batch_size, seq_length, vocab_size]
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Calculate log prob for each token
    token_log_probs = torch.gather(
        log_probs, 
        dim=2, 
        index=tokens.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_length]
    
    # Sum over sequence length
    sequence_log_probs = token_log_probs.mean(dim=1)  # [batch_size]
    
    return sequence_log_probs


def generate_query(
    model: torch.nn.Module,
    tokenizer: Any,
    context: List[str]
) -> torch.Tensor:
    """
    Generate a query from the model given context.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        context: Context text list [batch_size]
        max_length: Maximum number of tokens to generate
        
    Returns:
        torch.Tensor: Generated token IDs [batch_size, max_length]
    """
    # Tokenize the context
    batch_inputs = tokenizer(
        [f"{ctx}{QUERY_PREFIX}" for ctx in context],
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).to(DEVICE)
    
    # Generate query tokens - using same value for min and max tokens to ensure exact length
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            min_new_tokens=TOKENS_PER_QUERY,
            max_new_tokens=TOKENS_PER_QUERY,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,  # Use EOS as padding to avoid special tokens
            no_repeat_ngram_size=0,  # Disable n-gram penalties
            early_stopping=False,  # Don't stop before max_length
        )
    
    # Extract only the new tokens (query tokens)
    query_tokens = output_ids[:, batch_inputs["input_ids"].shape[1]:]
    
    return query_tokens


def compute_trajectory_rewards(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    context_tokens: torch.Tensor,
    tokenizer: Any = None,
    verbose: bool = False,
) -> torch.Tensor:
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
        torch.Tensor: Rewards for each step [batch_size, num_steps]
    """
    batch_size = context_tokens.shape[0]
    num_steps = len(trajectory.qkv_steps)
    
    if verbose:
        print("\n=== Computing Trajectory Rewards ===")
        print(f"Batch size: {batch_size}")
        print(f"Number of steps: {num_steps}")
    
    # Ensure context_tokens is on the correct device
    device = context_tokens.device
    
    # Initialize rewards tensor
    rewards = torch.zeros((batch_size, num_steps), device=device)
    
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
            query_prefix_tokens = tokenizer([QUERY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            # Use the stored query tokens if available, otherwise generate new ones
            query_tokens = None
            if qkv_step.query_tokens is not None:
                query_tokens = qkv_step.query_tokens.to(device)
            else:
                # Generate query tokens
                query_tokens = generate_query(
                    adapter_model,
                    tokenizer,
                    tokenizer.batch_decode(current_context)
                ).to(device)
            
                # Display generated query text in verbose mode
                if verbose and qkv_step.query_text is None:
                    query_text = tokenizer.batch_decode(query_tokens)
                    print(f"Generated Query: {query_text[0]}")
            
            current_context = torch.cat([
                current_context,
                query_prefix_tokens,
                query_tokens,
                key_prefix_tokens,
                key_tokens, 
                value_prefix_tokens,
                value_tokens
            ], dim=1)
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
    
    # Store rewards in the trajectory object
    trajectory.rewards = rewards
    trajectory.avg_reward = avg_reward
    
    return rewards


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


def filter_trajectories(trajectory: Trajectory, reward_stats: Dict[str, float]) -> Optional[Trajectory]:
    """
    Filter a batch of trajectories based on rewards, keeping only those that exceed a certain threshold.
    
    Args:
        trajectory: The batch of trajectories to filter
        reward_stats: Statistics about rewards used for normalization
        
    Returns:
        Filtered trajectory, or None if no trajectories pass the filter
    """
    if trajectory.avg_reward is None or reward_stats["count"] < 10:
        # Not enough data to filter meaningfully
        return trajectory
    
    # Calculate threshold as mean reward
    threshold = reward_stats["mean"]
    
    # Get batch indices where reward exceeds threshold
    batch_mask = trajectory.avg_reward > threshold
    
    # If no batch elements pass the threshold, return None
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
        
        if qkv_step.query_tokens is not None:
            filtered_attributes["query_tokens"] = qkv_step.query_tokens[batch_mask_device]
        
        if qkv_step.query_embedding is not None:
            filtered_attributes["query_embedding"] = qkv_step.query_embedding[batch_mask_device]
        
        # Create the filtered QKVStep
        filtered_qkv_step = QKVStep(**filtered_attributes)
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
    
    return filtered_trajectory


def compute_policy_loss(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    previous_model: Any,
    kl_penalty_coef: float,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the policy gradient loss with KL penalty.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter
        previous_model: The model state before update
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            total_loss: The combined loss value
            policy_loss: The policy gradient component
            kl_loss: The KL divergence component
    """
    policy_loss = 0.0
    kl_loss = 0.0
    count = 0
    
    # Determine device to use
    device = next(adapter_model.parameters()).device
    
    # Ensure trajectory has rewards
    if trajectory.rewards is None or trajectory.avg_reward is None:
        raise ValueError("Trajectory must have rewards computed before policy loss")
    
    # Get all queries from the trajectory's QKVSteps
    for qkv_step in trajectory.qkv_steps:
        # Get query tokens and ensure consistent device
        if qkv_step.query_tokens is not None:
            query_tokens = qkv_step.query_tokens.to(device)
        else:
            # In real usage, query_tokens should be populated by this point
            # This fallback is mainly for tests
            if verbose:
                print("WARNING: Using key_tokens as query_tokens since query_tokens is None. "
                      "In normal operation, query_tokens should be populated by this point.")
            query_tokens = qkv_step.key_tokens.to(device)
        
        # Use the average reward for the trajectory
        rewards = trajectory.avg_reward.to(device)
        
        # Forward pass with current model to compute log probabilities
        outputs_current = adapter_model(query_tokens)
        current_logits = outputs_current.logits
        
        # Forward pass with previous model
        with torch.no_grad():
            outputs_previous = previous_model(query_tokens)
            previous_logits = outputs_previous.logits
        
        # Compute log probabilities
        log_probs_current = F.log_softmax(current_logits, dim=-1)
        log_probs_previous = F.log_softmax(previous_logits, dim=-1)
        
        # Gather log probabilities of the actual tokens, skipping the first token
        # which is used as input
        token_indices = query_tokens[:, 1:].unsqueeze(-1)
        token_log_probs = torch.gather(
            log_probs_current[:, :-1, :], 
            dim=-1, 
            index=token_indices
        ).squeeze(-1)  # [batch_size, seq_length-1]
        
        # Compute policy gradient loss
        batch_policy_loss = -(token_log_probs.sum(dim=-1) * rewards).mean()
        policy_loss += batch_policy_loss
        
        # Compute KL divergence loss (regularization)
        batch_kl_loss = F.kl_div(
            log_probs_current.reshape(-1, log_probs_current.size(-1)),
            log_probs_previous.reshape(-1, log_probs_previous.size(-1)),
            reduction="batchmean",
            log_target=True
        )
        kl_loss += batch_kl_loss
        
        count += 1
    
    # Return average loss if there were trajectories, otherwise zero
    if count > 0:
        avg_policy_loss = policy_loss / count
        avg_kl_loss = kl_loss / count
        kl_penalty_term = kl_penalty_coef * avg_kl_loss
        total_loss = avg_policy_loss + kl_penalty_term
        
        if verbose:
            print(f"\n=== Loss Components ===")
            print(f"Policy loss: {avg_policy_loss.item():.4f}")
            print(f"KL divergence loss: {avg_kl_loss.item():.4f}")
            print(f"KL penalty coefficient: {kl_penalty_coef:.4f}")
            print(f"KL penalty term: {kl_penalty_term.item():.4f}")
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"=== End Loss Components ===\n")
            
        return total_loss, avg_policy_loss, avg_kl_loss
    else:
        zero_tensor = torch.tensor(0.0, device=device, requires_grad=True)
        return zero_tensor, zero_tensor, zero_tensor


def train_step(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    previous_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float,
    verbose: bool = False,
) -> Tuple[float, int, float, float]:
    """
    Perform a single training step.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter
        base_model: The base language model without LoRA
        previous_model: The model before update (for KL divergence)
        optimizer: The optimizer
        reward_stats: Reward statistics for filtering
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        
    Returns:
        Tuple[float, int, float, float]: 
            total_loss: Total loss value 
            num_filtered: Number of filtered batch elements
            policy_loss: Policy gradient component of the loss
            kl_loss: KL divergence component of the loss
    """
    # Import WARMUP_EPISODES only
    from src.config import WARMUP_EPISODES
    
    if verbose:
        print("\n=== Training Step ===")
        
        if trajectory.avg_reward is not None:
            batch_size = trajectory.avg_reward.shape[0]
            print(f"Input trajectory batch size: {batch_size}")
            print(f"Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, count={reward_stats['count']}")
            
            # Print individual batch element rewards
            if batch_size > 0:
                rewards = [f"{i}: {reward.item():.4f}" for i, reward in enumerate(trajectory.avg_reward)]
                print(f"Batch element rewards: {', '.join(rewards)}")
    
    # Filter trajectory based on reward
    filtered_trajectory = filter_trajectories(trajectory, reward_stats)
    
    # Skip update if no batch elements meet the criteria
    if filtered_trajectory is None:
        if verbose:
            print("No batch elements meet filtering criteria. Skipping update.")
        return 0.0, 0, 0.0, 0.0
    
    # Get the number of batch elements that passed filtering
    filtered_batch_size = filtered_trajectory.avg_reward.shape[0]
    
    if verbose:
        print(f"Filtered batch size: {filtered_batch_size}/{trajectory.avg_reward.shape[0]}")
        if reward_stats["count"] > WARMUP_EPISODES:
            threshold = reward_stats["mean"] + reward_stats["std"]
            print(f"Filtering threshold: {threshold:.4f}")
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Compute policy loss
    total_loss, policy_loss, kl_loss = compute_policy_loss(
        filtered_trajectory,
        adapter_model,
        previous_model,
        kl_penalty_coef,
        verbose=verbose
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
    
    return total_loss.item(), filtered_batch_size, policy_loss, kl_loss 