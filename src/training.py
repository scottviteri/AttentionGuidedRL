"""
Training module for the Attention-Guided RL project.

Contains the main training loop and related functions for trajectory generation and optimization.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any, Dict

from src.data import RawTrajectory, Trajectory
from src.embeddings import extract_embeddings, compute_similarity, register_embedding_hook
from src.config import CONFIG, TrainingConfig


# Helper to convert RawTrajectory after rewards are ready
def build_trajectory_from_raw(
    raw: "RawTrajectory",
    rewards: torch.Tensor,
    avg_reward: torch.Tensor,
) -> "Trajectory":
    return Trajectory(
        qkv_steps=raw.qkv_steps,
        rewards=rewards,
        avg_reward=avg_reward,
        all_key_embeddings=raw.all_key_embeddings,
    )


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
    
    # Return average log probability per token for better interpretability
    # This makes the plotted values much more meaningful (-5 vs -50)
    return token_log_probs.mean(dim=1)


def generate_query_vector(
    model: torch.nn.Module,
    tokenizer: Any,
    context_tokens: torch.Tensor,
    layer_idx: int = -2,
    config: Optional[TrainingConfig] = None
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
    # Get device from context tokens
    device = context_tokens.device
    batch_size = context_tokens.shape[0]
    
    # Register hook for query embeddings from specified layer
    embeddings_dict, hook_remover = register_embedding_hook(model, embed_type="query", layer_idx=layer_idx)
    
    try:
        # Tokenize the QUERY_VEC_TOKEN
        query_vec_token_ids = tokenizer(
            [CONFIG.query_vec_token] * batch_size,
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
    raw_traj: RawTrajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    context_tokens: torch.Tensor,
    tokenizer: Any = None,
    verbose: bool = False,
) -> Tuple[Trajectory, torch.Tensor, torch.Tensor]:
    """
    Compute rewards for all query-key-value steps in a trajectory.
    
    Args:
        raw_traj: The raw trajectory skeleton
        adapter_model: The model with LoRA adapter
        ref_model: The reference model without LoRA (pi_ref)
        context_tokens: Initial context tokens [batch_size, context_length]
        tokenizer: The tokenizer for processing text
        verbose: Flag to enable verbose logging
        
    Returns:
        Tuple[Trajectory, torch.Tensor, torch.Tensor]: 
            trajectory: The fully-specified trajectory with rewards
            rewards: Rewards for each step [batch_size, num_steps]
            adapter_log_probs: Log probabilities from adapter model [batch_size, num_steps]
            ref_log_probs: Log probabilities from reference model [batch_size, num_steps]
    """
    batch_size = context_tokens.shape[0]
    num_steps = len(raw_traj.qkv_steps)
    
    if verbose:
        print("\n=== Computing Trajectory Rewards ===")
        print(f"Batch size: {batch_size}")
        print(f"Number of steps: {num_steps}")
    
    # Ensure context_tokens is on the correct device
    device = context_tokens.device
    
    # Initialize tensors
    rewards = torch.zeros((batch_size, num_steps), device=device)
    adapter_log_probs = torch.zeros((batch_size, num_steps), device=device)
    ref_log_probs = torch.zeros((batch_size, num_steps), device=device)
    
    # Build context incrementally, including each step
    current_context = context_tokens
    
    for i, qkv_step in enumerate(raw_traj.qkv_steps):
        if verbose:
            print(f"\n--- Reward Calculation for Step {i+1}/{num_steps} ---")
            
            # Vector queries do not have textual query
            
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
        
        # Compute log prob with reference model (pi_ref)
        ref_log_prob = calculate_conditional_log_prob(
            ref_model, 
            value_tokens, 
            current_context
        )
        
        # Store log probabilities
        adapter_log_probs[:, i] = adapter_log_prob
        ref_log_probs[:, i] = ref_log_prob
        
        # Calculate reward - conditionally subtract reference model baseline
        if CONFIG.subtract_base_model_logprobs:
            # Classic approach: reward = improvement over reference model
            rewards[:, i] = adapter_log_prob - ref_log_prob
        else:
            # Simplified approach: use raw adapter performance, let GRPO handle baselines
            rewards[:, i] = adapter_log_prob
        
        if verbose:
            print(f"Adapter model log prob: {adapter_log_prob[0].item():.4f}")
            print(f"Reference model log prob: {ref_log_prob[0].item():.4f}")
            if CONFIG.subtract_base_model_logprobs:
                print(f"Reward (adapter - ref): {rewards[0, i].item():.4f}")
            else:
                print(f"Reward (raw adapter): {rewards[0, i].item():.4f}")
                print(f"Note: Using raw adapter log probs as rewards (SUBTRACT_BASE_MODEL_LOGPROBS=False)")
        
        # Update context for next iteration
        # Append query, key and value tokens to context, all on the same device
        if tokenizer:
            # Add prefixes if tokenizer is available
            batch_size = current_context.shape[0]
            key_prefix_tokens = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            value_prefix_tokens = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            # Vector queries - no query tokens to add to context
            current_context = torch.cat([
                current_context,
                key_prefix_tokens,
                key_tokens, 
                value_prefix_tokens,
                value_tokens
            ], dim=1)
            
            # Display vector query indicator (all steps are vector queries)
            if verbose:
                print("Using vector query")
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
        print(f"Average reference log prob: {ref_log_probs.mean().item():.4f}")
    
    # Build immutable Trajectory with rewards attached
    trajectory = build_trajectory_from_raw(raw_traj, rewards, avg_reward)

    return trajectory, adapter_log_probs, ref_log_probs


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
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    use_grpo_baseline: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Generalized Advantage Estimation (GAE) or GRPO-style baseline.
    
    Args:
        rewards: Tensor of rewards [batch_size, num_steps]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter for bias-variance tradeoff
        use_grpo_baseline: If True, use GRPO-style per-timestep batch average
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    returns = compute_returns(rewards, gamma)

    if use_grpo_baseline:
        # GRPO-style: per-timestep batch mean as baseline
        baseline = returns.mean(dim=0, keepdim=True)
        
        if gae_lambda < 1.0:
            # Modified GAE without value function:
            # Use exponentially weighted combination of future returns
            batch_size, num_steps = rewards.shape
            advantages = torch.zeros_like(rewards)
            
            # Work backwards from the end
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    # Last step: just use the return minus baseline
                    advantages[:, t] = returns[:, t] - baseline[:, t]
                else:
                    # GAE-style: combine current advantage with future
                    delta = rewards[:, t] + gamma * baseline[:, t + 1] - baseline[:, t]
                    advantages[:, t] = delta + gamma * gae_lambda * advantages[:, t + 1]
        else:
            # Standard Monte Carlo (lambda = 1.0)
            advantages = returns - baseline
    else:
        # Simpler: subtract per-trajectory mean then normalize
        advantages = returns - returns.mean(dim=1, keepdim=True)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns





def compute_policy_loss(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    old_model: torch.nn.Module,
    kl_penalty_coef: float,
    verbose: bool = False,
    gamma: float = 0.99,
    tokenizer: Any = None,
    embeddings_dict: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compute the PPO clipped surrogate loss with KL penalty for vector queries.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter
        ref_model: The reference model for KL divergence computation (pi_ref, fixed)
        old_model: The old model for probability ratios (pi_old, for PPO)
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        gamma: Discount factor for returns
        tokenizer: Tokenizer for reconstructing context (REQUIRED for vector queries)
        embeddings_dict: Embeddings dictionary (unused but kept for compatibility)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: 
            total_loss: The combined loss value
            policy_loss: The policy gradient component
            kl_loss: The KL divergence component
            avg_clipping_ratio: Average clipping ratio across all steps
    """
    # For vector queries, tokenizer is required
    if tokenizer is None:
        raise ValueError("Tokenizer is required for computing policy loss with vector queries")
    
    # Determine device to use
    device = next(adapter_model.parameters()).device
    
    # Initialize as tensors to maintain gradient flow
    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    kl_loss = torch.tensor(0.0, device=device, requires_grad=True) 
    count = 0
    
    # Track clipping ratios
    all_clipping_ratios = []
    
    # Import needed configs
    # Removed redundant local config import (constants available module-wide)
    
    # Compute returns and advantages
    advantages, _ = compute_advantages(
        trajectory.rewards, 
        gamma=gamma,
        gae_lambda=CONFIG.gae_lambda,
        use_grpo_baseline=CONFIG.use_grpo_baseline
    )
    

    
    # Compute what the old model would have generated (pi_old)
    # We'll do this by reconstructing the context at each step
    old_query_means = []
    
    # Get batch size from first step
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
    
    # Initialize context with the initial prompt
    context_tokens = tokenizer(
        [CONFIG.initial_prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).input_ids.to(device)
    
    # Reconstruct context and generate old model means for each step
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        # Generate query vector with old model (pi_old)
        with torch.no_grad():
            prev_query_mean = generate_query_vector(
                old_model,
                tokenizer,
                context_tokens,
                layer_idx=-2
            )
            old_query_means.append(prev_query_mean)
        
        # Update context for next iteration (add key and value tokens)
        key_prefix_tokens = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        value_prefix_tokens = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        context_tokens = torch.cat([
            context_tokens,
            key_prefix_tokens,
            qkv_step.key_tokens.to(device),
            value_prefix_tokens,
            qkv_step.value_tokens.to(device)
        ], dim=1)
    
    # Process each step in the trajectory
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        # For vector queries, we expect similarity_scores to be present
        if not hasattr(qkv_step, 'similarity_scores') or qkv_step.similarity_scores is None:
            if verbose:
                print(f"Warning: Skipping step {t} - no similarity scores")
            continue
        
        # Reconstruct context up to this step
        batch_size = qkv_step.key_tokens.shape[0]
        context_tokens = tokenizer(
            [CONFIG.initial_prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).input_ids.to(device)
        
        # Add all previous steps to context
        for prev_t in range(t):
            prev_step = trajectory.qkv_steps[prev_t]
            key_prefix_tokens = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            value_prefix_tokens = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            context_tokens = torch.cat([
                context_tokens,
                key_prefix_tokens,
                prev_step.key_tokens.to(device),
                value_prefix_tokens,
                prev_step.value_tokens.to(device)
            ], dim=1)
        
        # --- Recompute current-policy similarities so gradients flow into adapter_model ---
        # 1) Build query embedding with current (trainable) policy
        current_query_embeddings = generate_query_vector(
            adapter_model,
            tokenizer,
            context_tokens,
            layer_idx=-2,
        )  # [batch, hidden]

        # 2) Require full key embeddings to be present
        if trajectory.all_key_embeddings is None:
            raise ValueError("trajectory.all_key_embeddings is required but is None")

        key_embs_full = trajectory.all_key_embeddings.to(device)  # [B, K, H]

        # 3) Compute similarities with gradient through adapter_model parameters
        current_similarities = compute_similarity(current_query_embeddings, key_embs_full, adapter_model)

        # 4) Apply the availability mask
        if hasattr(qkv_step, 'available_mask') and qkv_step.available_mask is not None:
            masked_similarities = current_similarities + qkv_step.available_mask.to(device)
        else:
            masked_similarities = current_similarities

        # Compute current policy log-probabilities (masked)
        current_log_probs_full = masked_similarities

        selected_idx = qkv_step.selected_idx if hasattr(qkv_step, 'selected_idx') else torch.tensor([0]*batch_size, device=device)
        # Ensure selected_idx is tensor of shape [batch_size]
        if not isinstance(selected_idx, torch.Tensor):
            selected_idx = torch.tensor(selected_idx, device=device)
        current_action_log_probs = current_log_probs_full[torch.arange(batch_size, device=device), selected_idx]

        # --- Compute old-policy log-probabilities with same mask for PPO ---
        old_query_emb = old_query_means[t]  # [batch, hidden]

        # Require full key embeddings
        key_embs_full = trajectory.all_key_embeddings.to(device)
        old_similarities = compute_similarity(old_query_emb, key_embs_full, old_model)

        # Apply same availability mask
        old_masked_similarities = old_similarities + qkv_step.available_mask.to(device)

        old_log_probs_full = old_masked_similarities

        old_action_log_probs = old_log_probs_full[torch.arange(batch_size, device=device), selected_idx]

        # --- Compute reference-policy log-probabilities for KL divergence ---
        with torch.no_grad():
            ref_query_emb = generate_query_vector(ref_model, tokenizer, context_tokens, layer_idx=-2)
            ref_similarities = compute_similarity(ref_query_emb, key_embs_full, ref_model)
            ref_masked_similarities = ref_similarities + qkv_step.available_mask.to(device)
            ref_log_probs_full = ref_masked_similarities

        # Compute KL divergence between current and reference policies over available keys (masked)
        kl_step = F.kl_div(current_log_probs_full, ref_log_probs_full, reduction="batchmean", log_target=True)

        step_advantages = advantages[:, t].to(device)

        # Import config dynamically to get the current USE_PPO setting
        import src.config as config
        
        if CONFIG.use_ppo:
            # PPO: Use ratio clipping
            # Compute probability ratio: pi_theta(a|s) / pi_old(a|s)
            log_ratio = current_action_log_probs - old_action_log_probs
            ratio = torch.exp(log_ratio)
            
            # PPO clipped surrogate objective
            clipped_ratio = torch.clamp(ratio, 1.0 - CONFIG.ppo_clip_epsilon, 1.0 + CONFIG.ppo_clip_epsilon)
            
            # Track clipping ratios for monitoring
            all_clipping_ratios.extend(ratio.detach().cpu().tolist())
            
            # Compute both unclipped and clipped surrogate terms
            unclipped_surrogate = ratio * step_advantages
            clipped_surrogate = clipped_ratio * step_advantages
            
            # Take the minimum (more conservative update)
            ppo_surrogate = torch.min(unclipped_surrogate, clipped_surrogate)
            
            # Sum over batch (not average) - PPO typically sums over trajectory
            batch_policy_gradient = ppo_surrogate.sum()  # Sum over batch
            
            if verbose and t == 0:  # Print info for first step
                print(f"PPO ratio mean: {ratio.mean().item():.4f}, std: {ratio.std().item():.4f}")
                print(f"Clipped ratio range: [{clipped_ratio.min().item():.4f}, {clipped_ratio.max().item():.4f}]")
                print(f"KL divergence vs ref (masked): {kl_step.item():.4f}")
        else:
            # Vanilla Policy Gradient (REINFORCE): No ratio clipping
            # Direct policy gradient: log π(a|s) * A(s,a)
            vanilla_policy_gradient = current_action_log_probs * step_advantages
            batch_policy_gradient = vanilla_policy_gradient.sum()  # Sum over batch
            
            # Track "ratios" as 1.0 for monitoring compatibility
            all_clipping_ratios.extend([1.0] * len(current_action_log_probs))
            
            if verbose and t == 0:  # Print info for first step
                print(f"Vanilla PG: log_prob mean: {current_action_log_probs.mean().item():.4f}")
                print(f"Advantages mean: {step_advantages.mean().item():.4f}, std: {step_advantages.std().item():.4f}")
                print(f"KL divergence vs ref (masked): {kl_step.item():.4f}")
        
        policy_loss = policy_loss + batch_policy_gradient  # Accumulate across timesteps
        
        # Accumulate KL divergence for this step
        kl_loss = kl_loss + kl_step  # accumulate KL
        
        count += 1
    
    # Compute average clipping ratio
    avg_clipping_ratio = sum(all_clipping_ratios) / len(all_clipping_ratios) if all_clipping_ratios else 1.0
    
    # Compute final loss (sum over trajectory, not average)
    if count > 0:
        # Sum policy loss across trajectory steps (not average)
        total_policy_loss = policy_loss  # Already summed
        total_kl_loss = kl_loss / count  # Average KL loss across steps
        
        # Convert to loss: negate policy gradient (since we want to maximize expected reward)
        # and add KL penalty (since we want to minimize divergence)
        total_policy_loss = -total_policy_loss  # Negate for gradient ascent -> descent
        kl_penalty_term = kl_penalty_coef * total_kl_loss
        total_loss = total_policy_loss + kl_penalty_term
        
        if verbose:
            # Import config dynamically for the current USE_PPO setting
            import src.config as config
            method_name = "PPO" if CONFIG.use_ppo else "Vanilla PG"
            print(f"\n=== {method_name} Loss Components ===")
            print(f"Policy gradient sum (before negation): {-total_policy_loss.item():.4f}")
            print(f"Policy loss (after negation): {total_policy_loss.item():.4f}")
            print(f"KL divergence loss: {total_kl_loss.item():.4f}")
            print(f"KL penalty coefficient: {kl_penalty_coef:.4f}")
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"  = {total_policy_loss.item():.4f} + {kl_penalty_term.item():.4f}")
            if CONFIG.use_ppo:
                print(f"Average clipping ratio: {avg_clipping_ratio:.4f}")
            else:
                print(f"Method: Vanilla Policy Gradient (no clipping)")
            print(f"=== End {method_name} Loss Components ===\n")
            
        return total_loss, total_policy_loss, total_kl_loss, avg_clipping_ratio
    else:
        # Create a small non-zero tensor to ensure gradients can flow
        small_tensor = torch.tensor(1e-8, device=device, requires_grad=True)
        return small_tensor, small_tensor, small_tensor, 1.0


def train_step(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    old_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float,
    verbose: bool = False,
    tokenizer: Any = None,
    embeddings_dict: Optional[Dict] = None
) -> Tuple[float, float, float, float]:
    """
    Perform a single training step with simplified model architecture.
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter (trainable)
        ref_model: The reference language model without LoRA (pi_ref for KL computation)
        old_model: The old model for PPO probability ratios (pi_old, updated every N episodes)
        optimizer: The optimizer
        reward_stats: Reward statistics (for logging only with GRPO)
        kl_penalty_coef: KL penalty coefficient (beta)
        verbose: Flag to enable verbose logging
        tokenizer: Tokenizer (needed for vector queries)
        embeddings_dict: Embeddings dictionary (needed for vector queries)
        
    Returns:
        Tuple[float, float, float, float]: 
            total_loss: Total loss value 
            policy_loss: Policy gradient component of the loss
            kl_loss: KL divergence component of the loss
            avg_clipping_ratio: Average clipping ratio across all steps
    """
    if verbose:
        print("\n=== Training Step ===")
        batch_size = trajectory.avg_reward.shape[0]
        print(f"Input trajectory batch size: {batch_size}")
        print(f"Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")
        if batch_size > 0:
            rewards_str = ', '.join(f"{i}: {r.item():.4f}" for i, r in enumerate(trajectory.avg_reward))
            print(f"Batch element rewards: {rewards_str}")
    
    # No trajectory-level filtering - we use all trajectories
    
    if verbose:
        print(f"Processing all {trajectory.avg_reward.shape[0]} trajectories")
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Removed redundant local GAMMA import; constant already available
    # from src.config import GAMMA
    
    # Compute policy loss using ref_model for KL computation and old_model for PPO
    total_loss, policy_loss, kl_loss, avg_clipping_ratio = compute_policy_loss(
        trajectory,  # Use original trajectory, not filtered
        adapter_model,
        ref_model,  # Use ref_model for KL computation (fixed reference)
        old_model,  # Use old_model for PPO probability ratios
        kl_penalty_coef,
        verbose=verbose,
        gamma=CONFIG.gamma,
        tokenizer=tokenizer,
        embeddings_dict=embeddings_dict
    )
    
    if verbose:
        print(f"Total loss: {total_loss.item():.4f}")
    
    # Backpropagate loss
    total_loss.backward()
    
    # Get gradient norm for logging
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), CONFIG.gradient_clip_norm)
    
    if verbose:
        print(f"Gradient norm (before clipping): {grad_norm:.4f}")
    
    # Update parameters
    optimizer.step()
    
    if verbose:
        print("Parameters updated.")
        print(f"=== Training Step Complete ===\n")
    
    return total_loss.item(), policy_loss.item(), kl_loss.item(), avg_clipping_ratio 