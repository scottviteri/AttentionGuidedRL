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
    WARMUP_EPISODES,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    QUERY_PREFIX,
    RESPONSE_PREFIX,
    GRADIENT_CLIP_NORM,
    TEMPERATURE,
    TOP_P,
)
from src.model import create_model_copy
from src.data import KeyValuePair


@dataclass
class Trajectory:
    """
    A trajectory consisting of key-value pairs and optional rewards.
    
    Attributes:
        kv_pairs: List of key-value pairs selected during trajectory
        rewards: Optional tensor of rewards for each pair [batch_size, num_pairs]
        avg_reward: Average reward across pairs [batch_size]
    """
    kv_pairs: List[KeyValuePair]
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
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(context)
    
    # Get logits for the tokens positions we want to evaluate
    # If context is longer than tokens, we only look at the last tokens.shape[1] positions
    if outputs.logits.size(1) >= tokens.shape[1]:
        logits = outputs.logits[:, -tokens.shape[1]:, :]  # [batch_size, seq_length, vocab_size]
    else:
        # Handle case where context is shorter than tokens (shouldn't happen normally)
        raise ValueError(f"Context length {outputs.logits.size(1)} is shorter than token length {tokens.shape[1]}")
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Calculate log prob for each token
    token_log_probs = torch.gather(
        log_probs, 
        dim=2, 
        index=tokens.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_length]
    
    # Sum over sequence length
    sequence_log_probs = token_log_probs.sum(dim=1)  # [batch_size]
    
    return sequence_log_probs


def generate_query(
    model: torch.nn.Module,
    tokenizer: Any,
    context: List[str],
    max_length: int = TOKENS_PER_KEY,
    ensure_exact_length: bool = True,
) -> torch.Tensor:
    """
    Generate a query from the model given context.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        context: Context text list [batch_size]
        max_length: Maximum number of tokens to generate
        ensure_exact_length: If True, ensure exactly max_length tokens are generated
        
    Returns:
        torch.Tensor: Generated token IDs [batch_size, max_length]
    """
    # Tokenize the context
    batch_inputs = tokenizer(
        [f"{ctx}{QUERY_PREFIX}" for ctx in context],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)
    
    # Configure min and max tokens to be the same if exact length is required
    min_new_tokens = max_length if ensure_exact_length else 1
    
    # Generate query tokens
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,  # Use EOS as padding to avoid special tokens
            no_repeat_ngram_size=0,  # Disable n-gram penalties
            early_stopping=False,  # Don't stop before max_length
        )
    
    # Extract only the new tokens (query tokens)
    query_tokens = output_ids[:, batch_inputs["input_ids"].shape[1]:]
    
    # Ensure exact length by truncating if necessary
    if query_tokens.shape[1] > max_length:
        query_tokens = query_tokens[:, :max_length]
    
    # Check if we got the exact length requested
    if ensure_exact_length and query_tokens.shape[1] != max_length:
        raise ValueError(f"Generated query length {query_tokens.shape[1]} does not match requested length {max_length}")
    
    return query_tokens


def compute_trajectory_rewards(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    context_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rewards for all key-value pairs in a trajectory.
    
    Args:
        trajectory: The trajectory containing key-value pairs
        adapter_model: The model with LoRA adapter
        base_model: The base model without LoRA
        context_tokens: Initial context tokens [batch_size, context_length]
        
    Returns:
        torch.Tensor: Rewards for each key-value pair [batch_size, num_pairs]
    """
    batch_size = context_tokens.shape[0]
    num_pairs = len(trajectory.kv_pairs)
    
    # Ensure context_tokens is on the correct device
    device = context_tokens.device
    
    # Initialize rewards tensor
    rewards = torch.zeros((batch_size, num_pairs), device=device)
    
    # Build context incrementally, including each key-value pair
    current_context = context_tokens
    
    for i, kv_pair in enumerate(trajectory.kv_pairs):
        # Get value tokens and ensure they're on the same device as context
        value_tokens = kv_pair.value_tokens.to(device)
        
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
        
        # Update context for next iteration
        # Append key and value tokens to context, ensuring same device
        current_context = torch.cat([
            current_context, 
            kv_pair.key_tokens.to(device), 
            kv_pair.value_tokens.to(device)
        ], dim=1)
    
    # Compute average reward
    avg_reward = rewards.mean(dim=1)
    
    # Update trajectory with computed rewards
    trajectory.rewards = rewards.detach().clone()
    trajectory.avg_reward = avg_reward.detach().clone()
    
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


def filter_trajectories(
    trajectories: List[Trajectory],
    reward_stats: Dict[str, float],
) -> List[Trajectory]:
    """
    Filter trajectories based on reward.
    
    Args:
        trajectories: List of trajectories
        reward_stats: Reward statistics for filtering
        
    Returns:
        List[Trajectory]: Filtered trajectories
    """
    # Only filter if we've collected enough data
    if reward_stats["count"] <= WARMUP_EPISODES:
        return trajectories
    
    # Calculate threshold for filtering (mean + std)
    threshold = reward_stats["mean"] + reward_stats["std"]
    
    # Filter trajectories with average reward exceeding the threshold
    # Check only the first element of avg_reward (per batch)
    filtered_trajectories = [
        traj for traj in trajectories
        if traj.avg_reward is not None and traj.avg_reward[0].item() > threshold
    ]
    
    # If no trajectories meet the threshold, keep the best one
    if not filtered_trajectories and trajectories:
        # Find trajectory with highest avg_reward[0]
        best_trajectory = max(
            trajectories, 
            key=lambda t: t.avg_reward[0].item() if t.avg_reward is not None else float('-inf')
        )
        filtered_trajectories = [best_trajectory]
    
    return filtered_trajectories


def compute_policy_loss(
    trajectories: List[Trajectory],
    adapter_model: torch.nn.Module,
    previous_model: Any,
    kl_penalty_coef: float
) -> torch.Tensor:
    """
    Compute the policy gradient loss with KL penalty.
    
    Args:
        trajectories: List of trajectories to train on
        adapter_model: The language model with LoRA adapter
        previous_model: The model state before update
        kl_penalty_coef: KL penalty coefficient (beta)
        
    Returns:
        torch.Tensor: The loss value
    """
    policy_loss = 0.0
    kl_loss = 0.0
    count = 0
    
    # Determine device to use
    device = next(adapter_model.parameters()).device if hasattr(adapter_model, 'parameters') else DEVICE
    
    # Process each trajectory
    for trajectory in trajectories:
        # Ensure trajectory has rewards
        if trajectory.rewards is None or trajectory.avg_reward is None:
            raise ValueError("Trajectory must have rewards computed before policy loss")
        
        # Get all queries from the trajectory's KVPairs
        for kv_pair in trajectory.kv_pairs:
            # Get query tokens (key_tokens in KVPair) and ensure consistent device
            query_tokens = kv_pair.key_tokens.to(device)
            
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
        total_loss = avg_policy_loss + kl_penalty_coef * avg_kl_loss
        return total_loss
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def train_step(
    trajectories: List[Trajectory],
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    previous_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float
) -> Tuple[float, int]:
    """
    Perform a single training step.
    
    Args:
        trajectories: List of trajectories
        adapter_model: The language model with LoRA adapter
        base_model: The base language model without LoRA
        previous_model: The model before update (for KL divergence)
        optimizer: The optimizer
        reward_stats: Reward statistics for filtering
        kl_penalty_coef: KL penalty coefficient (beta)
        
    Returns:
        Tuple[float, int]: Loss value and number of filtered trajectories
    """
    # Filter trajectories based on reward
    filtered_trajectories = filter_trajectories(trajectories, reward_stats)
    
    # Skip update if no trajectories meet the criteria
    if not filtered_trajectories:
        return 0.0, 0
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Compute policy loss
    loss = compute_policy_loss(
        filtered_trajectories,
        adapter_model,
        previous_model,
        kl_penalty_coef
    )
    
    # Backpropagate loss
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), GRADIENT_CLIP_NORM)
    
    # Update parameters
    optimizer.step()
    
    return loss.item(), len(filtered_trajectories) 