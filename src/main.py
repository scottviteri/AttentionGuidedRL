"""
Main entry point for the Attention-Guided RL project.

This script sets up the training environment, creates models, loads data,
and runs the training loop.
"""

import os
import logging
import argparse
import torch
import torch.optim as optim
import math
from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from typing import List, Any, Iterator, Dict, Tuple, Optional
import sys
import time 

from src.config import CONFIG, TrainingConfig, create_training_config_from_args, log_training_config, training_config_to_dict
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path, update_model_ema
from src.data import KVPair, QKVSelection
from src.embeddings import register_embedding_hook, compute_similarity, sample_key_value, extract_embeddings, get_attention_params
from src.training import (
    RawTrajectory,
    Trajectory,  
    compute_trajectory_rewards,
    update_reward_stats,
    compute_advantages,
    generate_query_vector,
    calculate_conditional_log_prob,
    compute_returns,
)
from src.plotting import PlotData, save_plot_data
from src.metrics import (
    compute_wikipedia_order_consistency,
    compute_batch_selection_entropy,
    compute_advantage_distribution,
    compute_similarity_score_stats,
)

import wandb


def policy_gradient_train_step(
    trajectory: Trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,  
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    verbose: bool = False,
    tokenizer: Any = None,
    embeddings_dict: Optional[Dict] = None
) -> Tuple[float, float, float, float]:
    """
    Perform θ-dependent reward chain rule training step.
    
    Implements: ∇J(θ) = E[R_θ(τ)·∇log π_θ(τ) + ∇R_θ(τ)]
    
    Args:
        trajectory: The trajectory to train on
        adapter_model: The language model with LoRA adapter (trainable)
        ref_model: Kept for logging purposes
        optimizer: The optimizer
        reward_stats: Reward statistics (for logging only with GRPO)
        verbose: Flag to enable verbose logging
        tokenizer: Tokenizer (needed for vector queries)
        embeddings_dict: Embeddings dictionary (needed for vector queries)
        
    Returns:
        Tuple[float, float, float, float]: 
            total_loss, policy_loss, reward_loss, avg_clipping_ratio (always 1.0)
    """
    if verbose:
        print("\n=== θ-Dependent Chain Rule Training Step ===")
        batch_size = trajectory.avg_reward.shape[0]
        print(f"Input trajectory batch size: {batch_size}")
        print(f"Reward stats: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")

    # Sanity check: reference model must always be provided
    assert ref_model is not None, "ref_model (base/reference) must not be None"

    # Zero gradients
    optimizer.zero_grad()
    
    # Initialize loss tracking
    device = next(adapter_model.parameters()).device
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
    
    policy_term = torch.tensor(0.0, device=device, requires_grad=True)
    reward_term = torch.tensor(0.0, device=device, requires_grad=True)
    kl_loss_term = torch.tensor(0.0, device=device, requires_grad=True)
    kl_count = 0
    
    # Initialize empty context for computing current action probabilities and rewards
    current_context = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    
    # Use average reward over the entire trajectory for each action
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
    T = len(trajectory.qkv_steps)
    
    # Compute average reward over entire trajectory: R̄ = (1/T) * Σ_{s=1}^T r_s
    avg_trajectory_reward = trajectory.rewards.mean(dim=1, keepdim=True)  # [batch_size, 1]
    if CONFIG.use_grpo:
        # Center with batch mean baseline (stop-grad in policy term)
        batch_mean = avg_trajectory_reward.mean().detach()
        advantages = (avg_trajectory_reward - batch_mean).detach().expand(-1, T)  # [batch_size, T]
    else:
        # Uncentered uniform credit assignment
        advantages = avg_trajectory_reward.detach().expand(-1, T)
    
    for t, qkv_step in enumerate(trajectory.qkv_steps):
        # === POLICY GRADIENT TERM: A_t * ∇log π_θ(a_t|s_t) ===
        
        # Generate current query embedding (append SEP to trigger query-vector extraction)
        sep_for_query = tokenizer([CONFIG.chunk_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        query_context_for_emb = torch.cat([current_context, sep_for_query], dim=1)
        current_query_emb = generate_query_vector(
            adapter_model,
            tokenizer,
            query_context_for_emb,
            layer_idx=-2
        )
        
        # Get attention parameters and compute current action probabilities
        num_heads, num_groups, head_dim = get_attention_params(adapter_model)
        
        current_similarities = compute_similarity(
            current_query_emb,
            trajectory.all_key_embeddings,
            num_heads, 
            num_groups, 
            head_dim,
            availability_mask=qkv_step.available_mask if hasattr(qkv_step, 'available_mask') else None
        )
        
        # Get current action log probabilities
        current_action_log_probs = torch.gather(current_similarities, 1, qkv_step.selected_idx.unsqueeze(1)).squeeze(1)
        
        # Policy gradient term: A_t * log π_θ(a_t|s_t) (advantage-based)
        step_advantages = advantages[:, t]  # Advantage for this time step
        policy_gradient_t = step_advantages * current_action_log_probs
        policy_term = policy_term + policy_gradient_t.sum()  # Sum over batch
        
        # === REWARD GRADIENT TERM: ∇_θ r_{θ,t} ===
        # Prepare context for reward computation
        sep_tokens = tokenizer([CONFIG.chunk_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        kv_sep_tokens = tokenizer([CONFIG.kv_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        # Context for reward: includes key prefix + selected key + value prefix
        reward_context = torch.cat([
            current_context,
            sep_tokens,
            qkv_step.key_tokens.to(device),
            kv_sep_tokens
        ], dim=1)

        # === KL REGULARIZATION TERM (Value tokens): D_KL(p_θ(v_t|context) || p_ref(v_t|context)) ===
        if CONFIG.kl_penalty_coefficient > 0.0:
            # Adapter logits for value tokens (enable grad)
            adapter_full = torch.cat([reward_context, qkv_step.value_tokens.to(device)], dim=1)
            adapter_logits = adapter_model(adapter_full).logits
            ctx_len = reward_context.shape[1]
            adapter_token_logits = adapter_logits[:, ctx_len-1:ctx_len + qkv_step.value_tokens.shape[1] - 1, :]
            adapter_log_probs = F.log_softmax(adapter_token_logits, dim=-1)

            # Reference logits (no grad)
            with torch.no_grad():
                ref_full = adapter_full  # same sequence
                ref_logits = ref_model(ref_full).logits
                ref_token_logits = ref_logits[:, ctx_len-1:ctx_len + qkv_step.value_tokens.shape[1] - 1, :]
                ref_log_probs = F.log_softmax(ref_token_logits, dim=-1)

            # KL over tokens, average across batch and tokens
            kl_step = F.kl_div(adapter_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
            kl_loss_term = kl_loss_term + kl_step
            kl_count += 1
        
        # Recompute reward differentiably if enabled; otherwise fall back to cached rewards
        if CONFIG.differentiable_rewards:
            step_reward = calculate_conditional_log_prob(
                adapter_model,
                qkv_step.value_tokens.to(device),
                reward_context,
                differentiable=True,
            )  # [batch_size]
        else:
            step_reward = trajectory.rewards[:, t]
        
        # Scale reward term to be comparable in magnitude to policy term
        reward_scaling = 0.1
        reward_term = reward_term + reward_scaling * step_reward.sum()
        
        # Update context for next iteration
        current_context = torch.cat([
            current_context,
            qkv_step.key_tokens.to(device),
            kv_sep_tokens,
            qkv_step.value_tokens.to(device)
        ], dim=1)
        
        if verbose and t == 0:
            print(f"Step {t}: log_prob mean: {current_action_log_probs.mean().item():.4f}")
            print(f"Step {t}: advantage mean: {step_advantages.mean().item():.4f}")
            print(f"Step {t}: reward_t mean: {step_reward.mean().item():.4f}")
    
    # === CHAIN RULE LOSS: -(A_t·∇log π_θ(τ) + ∇R_θ(τ)) + β·D_KL(π_θ||π_ref) ===
    
    # Negative for gradient ascent -> descent
    total_loss = -(policy_term + reward_term)
    if CONFIG.kl_penalty_coefficient > 0.0 and kl_count > 0:
        avg_kl = kl_loss_term / kl_count
        total_loss = total_loss + CONFIG.kl_penalty_coefficient * avg_kl
    
    if verbose:
        print(f"Policy term: {policy_term.item():.4f}")
        print(f"Reward term: {reward_term.item():.4f}")
        print(f"Total loss (negated): {total_loss.item():.4f}")
    
    # Backpropagate; optionally drop policy term but still update reward model
    if CONFIG.freeze_policy:
        # Only train reward model: drop policy gradient contribution
        total_loss = -(reward_term)
    # else keep existing total_loss as -(policy_term + reward_term)
    total_loss.backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), CONFIG.gradient_clip_norm)
    
    if verbose:
        print(f"Gradient norm: {grad_norm:.4f}")
    
    # Update parameters (always step so reward model can train even if policy is frozen)
    optimizer.step()
    
    if verbose:
        print("=== Chain Rule Training Step Complete ===\n")
    
    # Return values for logging compatibility
    return total_loss.item(), policy_term.item(), reward_term.item(), 1.0

def setup_logging(config: TrainingConfig, args):
    """
    Set up logging for the training run.
    
    Args:
        config: Resolved training configuration
        args: Command-line arguments (for dataset, run_name, etc.)
    """
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set up formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log the resolved configuration using the clean method
    log_training_config(config, logging)
    logging.info(f"Dataset: {args.dataset}")
    
    # Initialize wandb if enabled
    if config.enable_wandb:
        wandb_config = training_config_to_dict(config)
        wandb_config['dataset'] = args.dataset  # Add runtime info
        wandb.init(
            project="attention-guided-rl",
            name=args.run_name if args.run_name else None,
            config=wandb_config
        )
        logging.info("Weights & Biases logging enabled")
    
    return log_dir


# === Trajectory generation helpers ===

def _build_available_mask(available_indices_per_batch: List[List[int]], num_keys: int, device: torch.device) -> torch.Tensor:
    """Return a mask tensor with 0 for available keys and -inf for masked ones."""
    batch_size = len(available_indices_per_batch)
    mask = torch.full((batch_size, num_keys), float('-inf'), device=device)
    for b, avail in enumerate(available_indices_per_batch):
        mask[b, avail] = 0.0
    return mask


def _append_step(raw_traj: RawTrajectory, step: QKVSelection) -> RawTrajectory:
    """Return a NEW RawTrajectory with *step* appended (dataclasses are immutable)."""
    return RawTrajectory(
        qkv_steps=raw_traj.qkv_steps + [step],
        all_key_embeddings=raw_traj.all_key_embeddings,
    )


def _update_context(context: torch.Tensor, step: QKVSelection, tokenizer, batch_size: int, device: torch.device) -> torch.Tensor:
    """Grow the autoregressive context by concatenating key/value tokens (no query tokens in vector mode)."""
    kv_sep_tokens = tokenizer([CONFIG.kv_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return torch.cat([
        context,
        step.key_tokens.to(device),
        kv_sep_tokens,
        step.value_tokens.to(device),
    ], dim=1)
# === End helpers ===


def generate_trajectory(
    context_tokens: torch.Tensor,
    adapter_model: torch.nn.Module,
    tokenizer: Any,
    available_qkv_steps: List[KVPair],
    batch_size: int,
    config: TrainingConfig,
    verbose: bool = False,
    rng: Optional[torch.Generator] = None,
) -> RawTrajectory:
    """
    Generate a single trajectory using vector queries (refactored, purely functional).
    """
    # === Setup ================================================================
    device = next(adapter_model.parameters()).device
    current_context = context_tokens.to(device)

    num_keys = len(available_qkv_steps)
    available_indices_per_batch: List[List[int]] = [list(range(num_keys)) for _ in range(batch_size)]
    # Track per-batch selections to assert no duplicate key selection within a trajectory
    selected_indices_seen_per_batch = [set() for _ in range(batch_size)]

    # Ensure each key embedding has batch dimension = batch_size
    key_emb_list = []
    for kv in available_qkv_steps:
        emb = kv.key_embedding.to(device)
        if emb.shape[0] == 1 and batch_size > 1:
            emb = emb.expand(batch_size, -1)  # broadcast singleton to whole batch
        key_emb_list.append(emb)

    trajectory_key_embeddings = torch.stack(key_emb_list, dim=1)  # [batch_size, num_keys, hidden]

    # Start with an empty immutable RawTrajectory
    traj: RawTrajectory = RawTrajectory(qkv_steps=[], all_key_embeddings=trajectory_key_embeddings)

    if verbose:
        decoded = tokenizer.batch_decode(current_context, skip_special_tokens=True)
        print("\n=== Starting New Trajectory ===")
        print(f"Initial context: {decoded[0][:50]}...")
        print(f"Vector-query mode. Pool size: {num_keys}")

    # === Autoregressive selection loop =======================================
    for _ in range(config.num_kv_pairs):
        # 1) Build query embedding using separator as the trigger
        sep_tokens = tokenizer([CONFIG.chunk_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        query_context = torch.cat([current_context, sep_tokens], dim=1)
        query_emb = generate_query_vector(adapter_model, tokenizer, query_context)

        # 2) Compute similarities
        # Get attention parameters from the adapter model
        num_heads, num_groups, head_dim = get_attention_params(adapter_model)
        # Create availability mask
        available_mask = _build_available_mask(available_indices_per_batch, num_keys, device).clamp(min=-1e9)
        similarity_scores = compute_similarity(query_emb, traj.all_key_embeddings, num_heads, num_groups, head_dim, 
                                             availability_mask=available_mask)

        # 4) Select an index per batch item
        if CONFIG.force_linear_action_order:
            # Deterministic: pick the smallest available index for each batch (linear order)
            selected_indices = [avail[0] for avail in available_indices_per_batch]
        else:
            # Stochastic: sample from policy distribution over available keys
            selected_indices, _ = sample_key_value(similarity_scores, available_indices_per_batch, batch_size, rng=rng)

        # === ASSERTION: No duplicate selection within the same batch element ===
        for b, idx in enumerate(selected_indices):
            if idx in selected_indices_seen_per_batch[b]:
                raise AssertionError(f"Duplicate key selection detected in batch {b}: index {idx} already selected in this trajectory")
            selected_indices_seen_per_batch[b].add(idx)

        # 5) Assemble tensors for the chosen KV pair
        selected_key_tokens = []
        selected_value_tokens = []
        selected_key_embeddings = []
        selected_key_texts = []
        selected_value_texts = []

        for b, idx in enumerate(selected_indices):
            kv = available_qkv_steps[idx]
            # Handle both single-batch and multi-batch KVPairs
            if kv.key_tokens.shape[0] == 1:
                # Single batch - use index 0
                selected_key_tokens.append(kv.key_tokens[0])
                selected_value_tokens.append(kv.value_tokens[0])
                selected_key_embeddings.append(kv.key_embedding[0])
                selected_key_texts.append(kv.key_text[0])
                selected_value_texts.append(kv.value_text[0])
            else:
                # Multi-batch - use batch index b
                selected_key_tokens.append(kv.key_tokens[b])
                selected_value_tokens.append(kv.value_tokens[b])
                selected_key_embeddings.append(kv.key_embedding[b])
                selected_key_texts.append(kv.key_text[b])
                selected_value_texts.append(kv.value_text[b])

        selected_key_tokens = torch.stack(selected_key_tokens, dim=0)
        selected_value_tokens = torch.stack(selected_value_tokens, dim=0)
        selected_key_embeddings = torch.stack(selected_key_embeddings, dim=0)

        step_data = KVPair(
            key_tokens=selected_key_tokens,
            value_tokens=selected_value_tokens,
            key_embedding=selected_key_embeddings,
            key_text=selected_key_texts,
            value_text=selected_value_texts,
        )

        qkv_step = QKVSelection(
            data=step_data,
            query_embedding=query_emb,
            similarity_scores=similarity_scores,
            selected_idx=torch.tensor(selected_indices, device=device),
            available_mask=available_mask,  # Store the mask for later reference
        )

        # 6) Append to immutable trajectory
        traj = _append_step(traj, qkv_step)

        # 7) Update bookkeeping lists and context
        for b, idx in enumerate(selected_indices):
            if idx in available_indices_per_batch[b]:
                available_indices_per_batch[b].remove(idx)

        # Insert a separator before appending the next KV chunk to context
        sep_tokens = tokenizer([CONFIG.chunk_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        current_context = torch.cat([_update_context(current_context, qkv_step, tokenizer, batch_size, device), sep_tokens], dim=1)

        # Optional debug: print the full context verbatim
        if CONFIG.debug_print_context and (verbose or True):
            decoded_ctx = tokenizer.batch_decode(current_context, skip_special_tokens=False)
            print("\n[DEBUG] Full context after step:")
            print(decoded_ctx[0])

    if verbose:
        full_ctx = tokenizer.batch_decode(current_context, skip_special_tokens=True)[0]
        print("\n=== Trajectory Complete ===")
        print(full_ctx)

    return traj


def parse_args():
    """Parse command-line arguments. Defaults are handled by TrainingConfig."""
    parser = argparse.ArgumentParser(description="Train a model using Attention-Guided RL")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--log-interval", type=int, help="Logging interval")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose trajectory logging")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("--run-name", type=str, help="Name for this training run")
    parser.add_argument("--dataset", type=str, default="wikipedia", 
                        choices=["wikipedia", "twenty_questions"],
                        help="Dataset to use for training")
    parser.add_argument("--model-type", type=str, choices=['gpt2', 'llama'], help='Model type to use')
    
    # Configuration parameters
    parser.add_argument('--enable-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--no-subtract-base-logprobs', action='store_true', help='Disable base model logprob subtraction (reward uses raw adapter logprobs)')
    parser.add_argument('--use-grpo', action='store_true', help='Enable GRPO-style centered baseline (default OFF)')
    parser.add_argument('--disable-grpo', action='store_true', help='Disable GRPO-style centered baseline (use uncentered rewards)')
    parser.add_argument('--force-linear-order', action='store_true', help='Always pick keys in linear order (disable sampling)')
    parser.add_argument('--freeze-policy', action='store_true', help='Do not update policy parameters (freeze policy gradient)')
    parser.add_argument('--debug-print-context', action='store_true', help='Print full context verbatim during trajectory and reward computation')
    # Note: subtraction defaults to ON; use --no-subtract-base-logprobs to disable
    parser.add_argument('--disable-differentiable-rewards', action='store_true', help='Disable differentiable reward backprop through value tokens')
    parser.add_argument('--debug-generators', action='store_true', help='Enable detailed debugging of generator pipelines')
    parser.add_argument('--kl-penalty-coefficient', type=float, help='KL penalty coefficient beta; 0 disables KL regularization')
    
    return parser.parse_args()


# moved to src.metrics.compute_wikipedia_order_consistency


# moved to src.metrics.compute_batch_selection_entropy


def test_edit_distance_function():
    """
    Test cases for the edit distance function used in Wikipedia order consistency.
    
    This function tests various scenarios to ensure the edit distance calculation
    and consistency scoring work correctly.
    """
    # Define the edit distance function (same as in compute_wikipedia_order_consistency)
    def edit_distance(seq1, seq2):
        """Compute edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    
    # Helper function to compute consistency score
    def compute_consistency_score(selected_indices):
        n = len(selected_indices)
        perfect_sequence = list(range(n))
        distance = edit_distance(selected_indices, perfect_sequence)
        max_distance = n
        if max_distance == 0:
            return 1.0
        consistency_score = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, consistency_score))
    
    # Test cases
    test_cases = [
        # (selected_indices, expected_consistency_score, description)
        ([0, 1, 2, 3], 1.0, "Perfect sequential order"),
        ([3, 2, 1, 0], 0.0, "Perfect reverse order (maximum distance)"),
        ([0, 1, 2], 1.0, "Perfect sequential order (3 elements)"),
        ([2, 1, 0], 0.33, "Perfect reverse order (3 elements, distance=2/3)"),
        ([0, 1, 2], 1.0, "Perfect sequential order (3 elements)"),
        ([0, 2, 1], 0.33, "One swap needed (2/3 distance)"),
        ([1, 0, 2], 0.33, "One swap needed (2/3 distance)"),
        ([0], 1.0, "Single element (perfect by definition)"),
        ([0, 1], 1.0, "Two elements in order"),
        ([1, 0], 0.0, "Two elements reversed"),
        ([1, 2, 0], 0.33, "Circular shift (distance=2/3)"),
        ([0, 2, 1, 3], 0.5, "One out of place (2/4 distance)"),
        ([1, 0, 3, 2], 0.25, "Two swaps needed (distance=3/4)"),
    ]
    
    print("\n=== Testing Edit Distance Function for Wikipedia Order Consistency ===")
    
    all_passed = True
    for selected_indices, expected_score, description in test_cases:
        actual_score = compute_consistency_score(selected_indices)
        n = len(selected_indices)
        perfect_sequence = list(range(n))
        distance = edit_distance(selected_indices, perfect_sequence)
        
        # Allow for small floating point differences
        tolerance = 0.01
        passed = abs(actual_score - expected_score) < tolerance
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {description}")
        print(f"  Selected: {selected_indices}, Perfect: {perfect_sequence}")
        print(f"  Edit distance: {distance}, Max distance: {n}")
        print(f"  Expected score: {expected_score:.2f}, Actual score: {actual_score:.2f}")
        print()
    
    if all_passed:
        print("🎉 All edit distance tests PASSED!")
    else:
        print("❌ Some edit distance tests FAILED!")
    
    print("=== End Edit Distance Tests ===\n")
    
    return all_passed


# Removed legacy key-selection KL function. Value-KL diagnostics retained below.


def compute_value_kl_from_reference(
    trajectory: "Trajectory",
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
) -> float:
    """Compute KL(adapter || ref) over value-token distributions for an entire trajectory (diagnostic).

    For each step, we reconstruct the reward context and extract logits at the
    value-token positions for both adapter and reference models, compute log-softmax
    and KL in log-space, then average over tokens and steps (batchmean).
    """
    device = next(adapter_model.parameters()).device
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]
    
    # Initialize empty context for value KL diagnostic
    context_tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
    
    kl_vals = []
    for step in trajectory.qkv_steps:
        key_tokens = step.key_tokens.to(device)
        value_tokens = step.value_tokens.to(device)
        sep = tokenizer([CONFIG.chunk_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        kv = tokenizer([CONFIG.kv_separator] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        reward_context = torch.cat([context_tokens, sep, key_tokens, kv], dim=1)
        
        # Adapter logits (no grad for diagnostic)
        with torch.no_grad():
            adapter_full = torch.cat([reward_context, value_tokens], dim=1)
            adapter_logits = adapter_model(adapter_full).logits
            ctx_len = reward_context.shape[1]
            adapter_token_logits = adapter_logits[:, ctx_len-1:ctx_len + value_tokens.shape[1] - 1, :]
            adapter_log_probs = F.log_softmax(adapter_token_logits, dim=-1)
            
            # Reference logits
            ref_full = adapter_full  # same sequence
            ref_logits = ref_model(ref_full).logits
            ref_token_logits = ref_logits[:, ctx_len-1:ctx_len + value_tokens.shape[1] - 1, :]
            ref_log_probs = F.log_softmax(ref_token_logits, dim=-1)
            
            # KL over tokens, average across batch and tokens
            kl_step = F.kl_div(adapter_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
            kl_vals.append(kl_step.item())
        
        # Advance context by appending key/value
        context_tokens = torch.cat([reward_context, value_tokens, sep], dim=1)
    
    if not kl_vals:
        return 0.0
    return sum(kl_vals) / len(kl_vals)


def main():
    """Main training function."""
    # Start overall training timer
    overall_start_time = time.time()

    # Parse arguments
    args = parse_args()
    
    # Create resolved configuration object - SINGLE point of configuration!
    config = create_training_config_from_args(args)
    
    # Set the runtime config for use throughout the system
    CONFIG.set_config(config)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
    
    # Set up logging with resolved config
    log_dir = setup_logging(config, args)
    
    # Log query mode
    logging.info("Query mode: Vector queries")
    
    # Log reward computation mode
    if CONFIG.subtract_base_model_logprobs:
        logging.info("Reward computation: Using adapter - base model (classic baseline subtraction)")
    else:
        logging.info("Reward computation: Using raw adapter log probabilities")
    
    # Log training algorithm
    logging.info("Training algorithm: Advantage-weighted policy gradient with differentiable reward (chain rule)")
    
    # Set up models and tokenizer
    logging.info("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Token configuration is already logged by config.log_configuration()
    # No need to manually set model configuration - TrainingConfig handles everything!
    
    # Separate hooks: one for query (train-time) and one for key (data loading)
    query_embeddings_dict, query_hook_remover = register_embedding_hook(adapter_model, embed_type="query")
    key_embeddings_dict, key_hook_remover   = register_embedding_hook(adapter_model, embed_type="key")

    # Helper to compute key embeddings once during data loading using the key-specific hook
    def compute_key_embedding(key_token_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return extract_embeddings(
                adapter_model,
                key_token_batch.to(CONFIG.device),
                key_embeddings_dict,
                requires_grad=False,
            ).detach()
    
    # Make sure hook is removed at the end
    try:
        # Create optimizer
        optimizer = optim.Adam(adapter_model.parameters(), lr=CONFIG.learning_rate)
        
        # Initialize reward stats
        reward_stats = {"mean": 0.0, "std": 1.0, "count": 0}
        
        # Track initial model weights for verification
        initial_base_weights = {}
        for name, param in base_model.named_parameters():
            if 'lora' not in name:  # Only track non-LoRA parameters
                initial_base_weights[name] = param.data.clone()
        
        # Try to load checkpoint if resume is specified
        start_episode = 0
        if args.resume:
            # First try to load the "latest" checkpoint
            if load_checkpoint(adapter_model, "latest"):
                # Get the checkpoint filename to parse the episode number
                latest_path = get_checkpoint_path("latest")
                # For backward compatibility, we'll set start_episode to the last episode
                # This ensures we don't restart from 0
                start_episode = CONFIG.num_episodes - 1
                logging.info(f"Resumed from latest checkpoint, continuing from episode {start_episode}")
            else:
                # Fall back to checking numbered checkpoints
                for episode in range(CONFIG.num_episodes, 0, -1):
                    if load_checkpoint(adapter_model, episode):
                        start_episode = episode
                        logging.info(f"Resumed from episode {start_episode}")
                        break
        
        # Policy gradient with simple LoRA adapter training
        # We only use the current adapter_model for both sampling and training

        # Import the data iterator and repeat function
        from src.data import (
            iter_key_value_pairs_unified_with_tokenizer, 
            repeat_n_times,
            debug_stream,
            count_stream,
            time_stream,
            peek_stream,
            KVPair
        )
        from typing import Iterator, cast
        
        # Standard batching approach
        # This creates a batch where each unique item appears multiple times
        base_iterator: Iterator[KVPair] = iter_key_value_pairs_unified_with_tokenizer(
                dataset_name=args.dataset,
                batch_size=1,  # Generate single items
                tokenizer=tokenizer,
                embedding_fn=compute_key_embedding,
        )
        
        # Add debugging if requested
        if args.debug_generators:
            logging.info("Debug mode enabled for generators")
            base_iterator = cast(Iterator[KVPair], peek_stream(base_iterator, peek_count=1))
            base_iterator = cast(Iterator[KVPair], debug_stream(base_iterator, "unique_kv_pairs", max_items=2))
            base_iterator = cast(Iterator[KVPair], time_stream(base_iterator, "kv_generation"))
            
        # Repeat each item batch_size times for batching
        kv_pair_generator: Iterator[KVPair] = cast(Iterator[KVPair], repeat_n_times(CONFIG.batch_size, base_iterator))
        
        if args.debug_generators:
            kv_pair_generator = cast(Iterator[KVPair], debug_stream(kv_pair_generator, "repeated_kv_pairs", max_items=CONFIG.batch_size + 1))
        
        # Log model setup
        logging.info("Model setup complete:")
        logging.info("- adapter_model: LoRA adapter (trainable)")
        logging.info("- ref_model: Reference model (pi_ref, for reward computation)")
        logging.info(f"- Batching: Enabled (repeat each data point {CONFIG.batch_size} times)")
        logging.info(f"- GRPO: {'ON' if CONFIG.use_grpo else 'OFF'} (batch repetition is independent of GRPO)")
        
        # Function to compute gradient statistics
        def get_gradient_stats(model):
            total_norm = 0.0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_count += 1
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            return total_norm, param_count
        
        # Function to compute weight change
        def compute_weight_change(model1, model2):
            total_change = 0.0
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                if p1.requires_grad:
                    total_change += (p1 - p2).norm(2).item() ** 2
            return total_change ** 0.5

        def track_lora_layer_gradients(model):
            """
            Track gradient magnitudes for each LoRA layer.
            
            Returns:
                Dict[int, float]: Mapping from layer index to gradient magnitude
            """
            layer_grads = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None and 'lora' in name.lower():
                    # Extract layer index from parameter name
                    # Examples for GPT-2: "base_model.model.transformer.h.10.attn.c_attn.lora_A.default.weight"
                    # Examples for Llama: "base_model.model.layers.10.self_attn.q_proj.lora_A.default.weight"
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        # Handle both GPT-2 (h.{layer}) and Llama (layers.{layer}) naming conventions
                        if (part == 'layers' or part == 'h') and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                grad_norm = param.grad.norm().item()
                                
                                # Accumulate gradients for the same layer (multiple LoRA parameters per layer)
                                if layer_idx in layer_grads:
                                    layer_grads[layer_idx] += grad_norm
                                else:
                                    layer_grads[layer_idx] = grad_norm
                                break
                            except (ValueError, IndexError):
                                continue
            
            return layer_grads

        # moved to src.metrics.compute_advantage_distribution

        # moved to src.metrics.compute_similarity_score_stats
        
        # Training loop
        logging.info("Starting training...")
        episodes_range = range(start_episode, CONFIG.num_episodes)
        progress_bar = tqdm(episodes_range)
        
        # Initialize plotting data structure (replaces global variables)
        plot_data = PlotData()
        
        # Training loop
        reward_history = []
        loss_history = []
        gradient_history = []
        weight_changes = []  # Track weight changes over time
        policy_gradients = []  # Track policy gradients (before negation) for conceptual clarity
        kl_from_ref = []  # Track KL divergence from reference model (for diagnostics only)

        
        for episode in progress_bar:
            if args.verbose:
                print(f"\n\n======== EPISODE {episode}/{CONFIG.num_episodes} ========")
            
            # Get a batch of key-value pairs
            available_qkv_steps = [next(kv_pair_generator) for _ in range(CONFIG.num_kv_pairs)]  # Get a pool of QKV steps
            
            if args.verbose:
                print(f"Generated pool of {len(available_qkv_steps)} query-key-value steps")
            
            # Initialize empty context; the first query will be triggered by appending SEP
            batch_size = CONFIG.batch_size
            device = next(adapter_model.parameters()).device
            initial_tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            
            # Policy gradient: always sample with current model
            if args.verbose:
                print("Sampling trajectory with current adapter_model (policy gradient)")
            
            # Generate a *raw* trajectory (no rewards yet)
            raw_traj = generate_trajectory(
                context_tokens=initial_tokens,
                adapter_model=adapter_model,
                tokenizer=tokenizer,
                available_qkv_steps=available_qkv_steps,
                batch_size=CONFIG.batch_size,
                config=CONFIG,
                verbose=args.verbose,
            )
            
            # Promote raw -> full trajectory and compute log probs using reference model
            trajectory, adapter_log_probs_batch, ref_log_probs_batch = compute_trajectory_rewards(
                raw_traj, 
                adapter_model, 
                base_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=args.verbose
            )
            
            # Policy gradient doesn't need old model log probs
            old_log_probs_batch = adapter_log_probs_batch

            # KL diagnostics: only value-token KL
            kl_keys_from_ref_value = 0.0
            kl_values_from_ref_value = compute_value_kl_from_reference(
                trajectory,
                adapter_model,
                base_model,
                tokenizer,
            )
            kl_from_ref_value = kl_values_from_ref_value
            
            # Update reward stats
            reward_stats = update_reward_stats(reward_stats, trajectory.avg_reward)
            if args.verbose:
                print(f"\nUpdated reward stats:")
                print(f"  Mean: {reward_stats['mean']:.4f}")
                print(f"  Std: {reward_stats['std']:.4f}")
                print(f"  Count: {reward_stats['count']}")
            
            # Perform θ-dependent chain rule training step
            train_step_results = policy_gradient_train_step(
                trajectory,
                adapter_model,
                base_model,  # Reference model
                optimizer,
                reward_stats,
                verbose=args.verbose,
                tokenizer=tokenizer,
                embeddings_dict=query_embeddings_dict,
            )
            
            # Unpack results (handle both old and new return formats)
            if len(train_step_results) == 12:
                (total_loss, policy_term, reward_term, avg_clipping_ratio,
                 policy_term_value, reward_term_value, total_returns_mean_val,
                 total_returns_std_val, policy_term_variance_val, reward_term_variance_val,
                 reward_gradient_norm_val, policy_reward_ratio_val) = train_step_results
            else:
                # Legacy format for compatibility
                total_loss, policy_term, reward_term, avg_clipping_ratio = train_step_results
                policy_term_value = reward_term_value = total_returns_mean_val = 0.0
                total_returns_std_val = policy_term_variance_val = reward_term_variance_val = 0.0
                reward_gradient_norm_val = policy_reward_ratio_val = 0.0
            
            # Note: clipping ratio not used in chain rule training (always 1.0)
            
            # Simple weight change tracking (gradient norm as proxy)
            weight_change = 0.0  # Simplified for policy gradient
            weight_changes.append(weight_change)
            
            # Policy gradient doesn't need old model updates
            
            # DEBUG: Removed - EMA updates are now enabled
            
            # Periodically verify weight changes (every 5 episodes) – check LoRA params correctly
            if (episode + 1) % 5 == 0:
                # Build initial snapshot of LoRA params if not already
                if 'initial_lora_weights' not in locals():
                    initial_lora_weights = {name: p.data.clone() for name, p in adapter_model.named_parameters() if 'lora' in name}

                # Check any LoRA param diverged from initial snapshot
                any_changed = False
                for name, param in adapter_model.named_parameters():
                    if 'lora' in name and name in initial_lora_weights:
                        if not torch.allclose(initial_lora_weights[name], param.data):
                            any_changed = True
                            break

                if any_changed:
                    logging.info("Adapter LoRA weights are changing as expected ✅")
                else:
                    logging.warning("Adapter LoRA weights have not changed – investigate optimizer/grad flow ⚠️")
            
            # Save checkpoint if needed
            if episode > 0 and episode % CONFIG.checkpoint_interval == 0:
                save_checkpoint(adapter_model, "latest")
                if args.verbose:
                    print(f"\nCheckpoint saved at episode {episode}")
                
            # Calculate average reward across the batch
            avg_reward = trajectory.avg_reward.mean().item()
            
            # Compute Wikipedia order consistency metric
            if args.dataset == "wikipedia":
                order_consistency = compute_wikipedia_order_consistency(trajectory)
            else:
                order_consistency = 0.5  # Neutral for non-Wikipedia datasets
            
            # Compute batch selection entropy
            selection_entropy = compute_batch_selection_entropy(trajectory)
            
            # Periodically log detailed trajectory information
            if episode % 50 == 0 or episode < 5:  # Log first few and then every 50 episodes
                trajectory_info = {
                    'episode': episode,
                    'selections': [],
                    'key_texts': [],
                    'value_texts': [],
                    'rewards': trajectory.rewards.tolist() if trajectory.rewards is not None else [],
                    'avg_reward': trajectory.avg_reward.tolist() if trajectory.avg_reward is not None else [],
                    'index_sequences': [],  # Add index sequences for each batch item
                }
                
                # Extract index sequences for each batch item
                for b in range(batch_size):
                    batch_sequence = []
                    for step in trajectory.qkv_steps:
                        if hasattr(step, 'selected_idx') and step.selected_idx is not None:
                            batch_sequence.append(step.selected_idx[b].item())
                    trajectory_info['index_sequences'].append(batch_sequence)
                
                # Extract selection information from each step
                for step_idx, step in enumerate(trajectory.qkv_steps):
                    step_info = {
                        'step': step_idx,
                        'selected_idx': step.selected_idx if hasattr(step, 'selected_idx') else None,
                        'key_text': step.key_text[0] if step.key_text else None,  # First batch item
                        'value_text': step.value_text[0] if step.value_text else None,
                    }
                    trajectory_info['selections'].append(step_info)
                    
                    # Store key/value texts for first batch item
                    if step.key_text:
                        trajectory_info['key_texts'].append(step.key_text[0])
                    if step.value_text:
                        trajectory_info['value_texts'].append(step.value_text[0])
                
                # trajectory_info will be passed to PlotData later
                
                # Also log to console if verbose or first few episodes
                if args.verbose or episode < 3:
                    print(f"\n=== Trajectory Sample Episode {episode} ===")
                    print(f"Selection entropy: {selection_entropy:.3f}")
                    print(f"Average reward: {avg_reward:.4f}")
                    
                    # Show index sequences for all batch items
                    print("\nIndex sequences (pool indices, not original key IDs):")
                    for b, seq in enumerate(trajectory_info['index_sequences'][:4]):  # Show up to 4 batch items
                        print(f"  Batch {b}: {seq}")
                    
                    # Check if all batch items have same sequence
                    unique_sequences = len(set(tuple(seq) for seq in trajectory_info['index_sequences']))
                    print(f"Unique sequences: {unique_sequences}/{batch_size}")
                    
                    print("\nSelected indices across batch (first 5 steps):")
                    for info in trajectory_info['selections'][:5]:  # Show first 5
                        print(f"  Step {info['step']}: idx={info['selected_idx']}")
                        if info['key_text']:
                            print(f"    Key: {info['key_text'][:50]}...")
                        if info['value_text']:
                            print(f"    Value: {info['value_text'][:50]}...")
            
            # Calculate derived metrics for plotting
            traj_log_prob = adapter_log_probs_batch.mean().item()
            # KL penalty term (use diagnostic KL if available)
            kl_penalty_term = CONFIG.kl_penalty_coefficient * kl_from_ref_value if CONFIG.kl_penalty_coefficient > 0.0 else 0.0
            # Compute reward variance (handle single batch case)
            if trajectory.avg_reward.numel() > 1:
                reward_var = trajectory.avg_reward.var().item()
            else:
                reward_var = 0.0  # Single batch, no variance
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute and track average advantages from this episode
            if trajectory.rewards is not None:
                # Simple trajectory average advantages (no GAE or baseline)
                advantages = trajectory.rewards.mean(dim=1, keepdim=True).expand_as(trajectory.rewards)
                avg_advantage = advantages.mean().item()
                advantage_dist = compute_advantage_distribution(advantages)
            else:
                avg_advantage = 0.0
                advantage_dist = {
                    'positive_percentage': 0.0, 'negative_percentage': 0.0, 'zero_percentage': 100.0,
                    'mean': 0.0, 'std': 0.0
                }
            
            # Policy gradient for visualization (before negation)
            policy_gradient = -(policy_term.item() if isinstance(policy_term, torch.Tensor) else policy_term)
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode}/{CONFIG.num_episodes}, "
                f"Loss: {total_loss:.4f}, "

                f"Reward: {avg_reward:.4f}"
            )
            
            # Policy gradient doesn't need baseline model updates
            
            # DEBUG: Removed - EMA updates are now enabled
            
            # Periodically verify weight changes (every 5 episodes) – check LoRA params correctly
            if (episode + 1) % 5 == 0:
                # Build initial snapshot of LoRA params if not already
                if 'initial_lora_weights' not in locals():
                    initial_lora_weights = {name: p.data.clone() for name, p in adapter_model.named_parameters() if 'lora' in name}

                # Check any LoRA param diverged from initial snapshot
                any_changed = False
                for name, param in adapter_model.named_parameters():
                    if 'lora' in name and name in initial_lora_weights:
                        if not torch.allclose(initial_lora_weights[name], param.data):
                            any_changed = True
                            break

                if any_changed:
                    logging.info("Adapter LoRA weights are changing as expected ✅")
                else:
                    logging.warning("Adapter LoRA weights have not changed – investigate optimizer/grad flow ⚠️")
            
            # Save checkpoint if needed
            if episode > 0 and episode % CONFIG.checkpoint_interval == 0:
                save_checkpoint(adapter_model, "latest")
                if args.verbose:
                    print(f"\nCheckpoint saved at episode {episode}")
                
            # Calculate additional metrics
            gradient_stats = get_gradient_stats(adapter_model)
            gradient_history.append(gradient_stats)
            gradient_magnitude = gradient_stats[0]
            
            # Track LoRA layer-specific gradients
            layer_grads = track_lora_layer_gradients(adapter_model)
            
            # Track similarity score statistics
            similarity_stats = compute_similarity_score_stats(trajectory)
            
            # Track reward and loss history
            reward_history.append(avg_reward)
            loss_history.append(total_loss)
            
            # Calculate log probabilities for plotting
            adapter_log_prob = adapter_log_probs_batch.mean().item()
            baseline_log_prob = old_log_probs_batch.mean().item()
            base_log_prob = ref_log_probs_batch.mean().item()
            
            # Track log probabilities by step index (for the new plot)
            # Compute log probabilities of selected actions at each step
            step_log_probs_episode = []
            step_selected_indices_episode = []
            for step in trajectory.qkv_steps:
                if hasattr(step, 'similarity_scores') and step.similarity_scores is not None:
                    # TEMPERATURE and F are already in global scope
                    
                    # Apply available mask if present (same as in policy loss computation)
                    similarities = step.similarity_scores
                    if hasattr(step, 'available_mask') and step.available_mask is not None:
                        # Mask out unavailable keys
                        masked_similarities = similarities + step.available_mask
                    else:
                        masked_similarities = similarities
                    
                    log_probs = masked_similarities
                    # Get log prob of selected action (average over batch)
                    if not isinstance(step.selected_idx, torch.Tensor):
                        raise TypeError("selected_idx must be a torch.Tensor")
                    selected_idx = step.selected_idx
                    selected_log_prob = log_probs[torch.arange(log_probs.shape[0]), selected_idx].mean().item()
                    step_log_probs_episode.append(selected_log_prob)
                    # Average selected key index across batch for this step
                    step_selected_indices_episode.append(selected_idx.float().mean().item())
            # Add episode data to plot_data using the clean functional approach
            trajectory_sample = trajectory_info if (episode % 50 == 0 or episode < 5) and 'trajectory_info' in locals() else None
            # Compute centered reward for plotting (for clarity only)
            centered_reward = 0.0
            if trajectory.avg_reward is not None:
                centered = trajectory.avg_reward - trajectory.avg_reward.mean()
                centered_reward = centered.mean().item()

            plot_data = plot_data.add_episode_data(
                episode=episode,
                total_loss=total_loss,
                policy_loss=policy_term.item() if isinstance(policy_term, torch.Tensor) else policy_term,
                kl_loss=0.0,  # No KL loss in chain rule approach
                avg_reward=avg_reward,
                avg_centered_reward=centered_reward,
                adapter_log_prob=adapter_log_prob,
                baseline_log_prob=baseline_log_prob,
                base_log_prob=base_log_prob,
                avg_advantage=avg_advantage,
                trajectory_log_prob=traj_log_prob,
                wikipedia_order_consistency=order_consistency,
                entropy_value=0.0,  # TODO: Calculate actual entropy if needed
                kl_penalty_term=kl_penalty_term,
                reward_variance=reward_var,
                gradient_magnitude=gradient_magnitude,
                step_log_probs_episode=step_log_probs_episode,
                step_selected_indices_episode=step_selected_indices_episode,
                clipping_ratio=1.0,  # Chain rule training doesn't use clipping
                batch_selection_entropy=selection_entropy,
                kl_from_ref_value=kl_from_ref_value,
                kl_keys_from_ref_value=kl_keys_from_ref_value,
                kl_values_from_ref_value=kl_values_from_ref_value,
                lora_layer_gradients_episode=layer_grads,
                advantage_distribution=advantage_dist,
                similarity_score_stats=similarity_stats,
                policy_gradient=policy_gradient,
                # New chain rule metrics
                policy_term_value=policy_term_value,
                reward_term_value=reward_term_value,
                total_returns_mean_val=total_returns_mean_val,
                total_returns_std_val=total_returns_std_val,
                policy_term_variance_val=policy_term_variance_val,
                reward_term_variance_val=reward_term_variance_val,
                reward_gradient_norm_val=reward_gradient_norm_val,
                policy_reward_ratio_val=policy_reward_ratio_val,
                trajectory_sample=trajectory_sample
            )
            
            # Save and plot metrics periodically
            if episode > 0 and episode % 15 == 0:
                # Add metadata to plot data and save
                save_plot_data(plot_data, log_dir)
                plot_metrics(log_dir, policy_gradients)
            
            # Log every log_interval episodes
            if episode % CONFIG.log_interval == 0:
                # Convert tensors to floats for logging
                policy_term_val = policy_term.item() if isinstance(policy_term, torch.Tensor) else policy_term
                reward_term_val = reward_term.item() if isinstance(reward_term, torch.Tensor) else reward_term
                
                base_msg = (
                    f"Episode {episode}/{CONFIG.num_episodes}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Policy Term: {policy_term_val:.4f}, "
                    f"Reward Term: {reward_term_val:.4f}, "
                    f"Reward: {avg_reward:.4f}, "
                    f"Reward Mean: {reward_stats['mean']:.4f}, "
                    f"Reward Std: {reward_stats['std']:.4f}"
                )
                if CONFIG.kl_penalty_coefficient > 0.0:
                    base_msg += f", KL Penalty (beta*D_KL): {(CONFIG.kl_penalty_coefficient * kl_from_ref_value):.4f}"
                logging.info(base_msg)
                
                # Log to wandb if enabled
                if CONFIG.enable_wandb:
                    log_dict = {
                        "episode": episode,
                        "total_loss": total_loss,
                        "policy_term": policy_term_val,
                        "reward_term": reward_term_val,
                        "chain_rule_loss": total_loss,
                        "reward": avg_reward,
                        "reward_mean": reward_stats["mean"],
                        "reward_std": reward_stats["std"],
                    }
                    if CONFIG.kl_penalty_coefficient > 0.0:
                        log_dict.update({
                            "kl_from_ref": kl_from_ref_value,
                            "kl_penalty_beta": CONFIG.kl_penalty_coefficient,
                            "kl_penalty_term": CONFIG.kl_penalty_coefficient * kl_from_ref_value,
                        })
                    wandb.log(log_dict)
                
                # Log gradient diagnostics
                if len(plot_data.gradient_magnitudes) > 0:
                    logging.info(
                        f"  Gradient Stats - Norm: {plot_data.gradient_magnitudes[-1]:.6f}, "
                        f"Weight Change: {weight_changes[-1]:.6f}"
                    )
                    
                    # Check if weights are updating
                    if weight_changes[-1] < 1e-6:
                        logging.warning("  WARNING: Weights are not changing!")
                    
                    # Check gradient health
                    if plot_data.gradient_magnitudes[-1] < 1e-8:
                        logging.warning("  WARNING: Gradients are vanishing!")
                    elif plot_data.gradient_magnitudes[-1] > 100:
                        logging.warning("  WARNING: Gradients are exploding!")
            
        # Save final checkpoint
        save_checkpoint(adapter_model, "latest")
        
        # Save final plot data and create plots
        save_plot_data(plot_data, log_dir)
        plot_metrics(log_dir, None)
        
        logging.info("Training complete!")
        
        # Close wandb if enabled
        if CONFIG.enable_wandb:
            wandb.finish()
    
    finally:
        # Remove hooks
        query_hook_remover()
        key_hook_remover()
        # End overall training timer
        overall_end_time = time.time()
        total_overall_time_minutes = (overall_end_time - overall_start_time) / 60
        logging.info(f"Overall training duration: {total_overall_time_minutes:.2f} minutes")


# Old save_plot_data function removed - now using the one from src.plotting module


def plot_metrics(log_dir, policy_gradients_data=None):
    """
    Create and save comprehensive visualization of training metrics by calling generate_plots.py.
    
    Args:
        log_dir: Directory where logs and plots are saved
        policy_gradients_data: List of policy gradient values for plotting (unused, kept for compatibility)
    """
    import subprocess
    
    # Get the plot data file
    plots_dir = f"{log_dir}/plots"
    latest_pickle = f"{plots_dir}/plot_data.pkl"
    
    if not os.path.exists(latest_pickle):
        logging.warning(f"No plot data found at {latest_pickle}")
        return
    
    # Get the absolute path to generate_plots.py (now in src directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_plots_script = os.path.join(script_dir, "generate_plots.py")
    
    if not os.path.exists(generate_plots_script):
        logging.error(f"generate_plots.py not found at {generate_plots_script}")
        return
    
    # Call generate_plots.py with the latest data
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, generate_plots_script, latest_pickle],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log any output from the script
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    logging.info(f"generate_plots.py: {line}")
                    
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line:
                    logging.warning(f"generate_plots.py stderr: {line}")
                    
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run generate_plots.py: {e}")
        if e.stdout:
            logging.error(f"stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"stderr: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error running generate_plots.py: {e}")


if __name__ == "__main__":
    main() 