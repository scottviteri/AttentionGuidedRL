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
from typing import List, Any, Iterator
import sys
import time # Import time for overall timing

# Import configuration management - clean pattern!
from src.config import CONFIG, TrainingConfig, create_training_config_from_args, log_training_config, training_config_to_dict
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path, update_model_ema
from src.data import KVPair, QKVSelection
from src.embeddings import register_embedding_hook, compute_similarity, sample_key_value, extract_embeddings, get_attention_params
from src.training import (
    RawTrajectory,
    Trajectory,  # for type hints
    compute_trajectory_rewards,
    update_reward_stats,
    train_step,
    generate_query_vector,
)
from src.plotting import PlotData, save_plot_data, create_metadata

# Import memory-efficient training components
from src.memory_efficient_training import (
    MemoryEfficientLoRAManager,
    memory_efficient_train_step
)

import wandb

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
    key_prefix_tokens = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    value_prefix_tokens = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return torch.cat([
        context,
        key_prefix_tokens,
        step.key_tokens.to(device),
        value_prefix_tokens,
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
) -> RawTrajectory:
    """
    Generate a single trajectory using vector queries (refactored, purely functional).
    """
    # === Setup ================================================================
    device = next(adapter_model.parameters()).device
    current_context = context_tokens.to(device)

    num_keys = len(available_qkv_steps)
    available_indices_per_batch: List[List[int]] = [list(range(num_keys)) for _ in range(batch_size)]

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
        # 1) Build query embedding
        query_emb = generate_query_vector(adapter_model, tokenizer, current_context)

        # 2) Compute similarities
        # Get attention parameters from the adapter model
        num_heads, num_groups, head_dim = get_attention_params(adapter_model)
        # Create availability mask
        available_mask = _build_available_mask(available_indices_per_batch, num_keys, device).clamp(min=-1e9)
        similarity_scores = compute_similarity(query_emb, traj.all_key_embeddings, num_heads, num_groups, head_dim, 
                                             availability_mask=available_mask)

        # 4) Sample an index per batch item, respecting already-used keys
        selected_indices, _ = sample_key_value(similarity_scores, available_indices_per_batch, batch_size)

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

        current_context = _update_context(current_context, qkv_step, tokenizer, batch_size, device)

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
    parser.add_argument("--grpo-batching", action="store_true", help="Use GRPO-style batching (repeat each data point)")
    parser.add_argument("--model-type", type=str, choices=['gpt2', 'llama'], help='Model type to use')
    parser.add_argument('--use-grpo-baseline', action='store_true', help='Use GRPO baseline in advantages')
    
    # Configuration parameters
    parser.add_argument('--kl-penalty-coef', type=float, help='KL penalty coefficient for regularization')
    parser.add_argument('--enable-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--ppo-clip-epsilon', type=float, help='PPO clipping parameter (epsilon)')
    parser.add_argument('--baseline-update-freq', type=int, help='How often to update baseline model (episodes)')
    parser.add_argument('--subtract-base-logprobs', action='store_true', help='Subtract base model logprobs in reward computation')
    parser.add_argument('--debug-generators', action='store_true', help='Enable detailed debugging of generator pipelines')
    parser.add_argument('--ema-decay', type=float, help='EMA decay for smooth baseline updates (0.01-0.1, higher=smoother)')
    parser.add_argument('--use-ema-baseline', action='store_true', help='Use EMA baseline updates instead of hard updates (reduces spikes)')
    parser.add_argument('--vanilla-pg', action='store_true', help='Use vanilla policy gradient (REINFORCE) instead of PPO')
    parser.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient LoRA state management (saves 60-90%% memory)')
    
    return parser.parse_args()


def compute_wikipedia_order_consistency(trajectory) -> float:
    """
    Compute how consistently the model selects keys in their original Wikipedia article order.
    
    Uses edit distance from the perfect sequential order to measure consistency.
    Perfect order consistency = 1.0 (sequence matches 0,1,2,3,...)
    Maximum disorder = 0.0 (sequence requires maximum edits to fix)
    
    Args:
        trajectory: The trajectory containing selected key indices
        
    Returns:
        float: Order consistency score between 0.0 and 1.0
    """
    # import torch  # redundant (already at module level)
    if not trajectory.qkv_steps:
        raise ValueError("Trajectory must contain qkv_steps")

    first_step = trajectory.qkv_steps[0]
    if not isinstance(first_step.selected_idx, torch.Tensor):
        raise TypeError("selected_idx must be a torch.Tensor")
    if first_step.selected_idx.numel() == 0:
        raise ValueError("selected_idx tensor is empty")
    
    batch_size = trajectory.qkv_steps[0].selected_idx.shape[0]
    all_batch_consistency_scores = []

    # Define the edit_distance function locally for clarity
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

    for b in range(batch_size):
        selected_indices_for_batch_item = []
        for step in trajectory.qkv_steps:
            if hasattr(step, 'selected_idx') and isinstance(step.selected_idx, torch.Tensor):
                selected_indices_for_batch_item.append(step.selected_idx[b].item())
        
        if len(selected_indices_for_batch_item) < 2:
            all_batch_consistency_scores.append(0.5) # Neutral if not enough steps for this batch item
            continue
        
        n = len(selected_indices_for_batch_item)
        perfect_sequence = list(range(n))
        
        # Compute edit distance from perfect sequence
        distance = edit_distance(selected_indices_for_batch_item, perfect_sequence)
        
        # Normalize: maximum possible edit distance is n (for complete reversal or large differences)
        max_distance = n
        
        if max_distance == 0:
            consistency_score = 1.0  # Edge case: single element
        else:
            consistency_score = 1.0 - (distance / max_distance)
        
        all_batch_consistency_scores.append(max(0.0, min(1.0, consistency_score)))  # Clamp to [0, 1]

    return sum(all_batch_consistency_scores) / len(all_batch_consistency_scores) if all_batch_consistency_scores else 0.5


def compute_batch_selection_entropy(trajectory) -> float:
    """
    Compute the entropy of key selection orders within a batch.
    
    High entropy indicates diverse selection patterns across batch items.
    Low entropy indicates similar/identical selection patterns.
    
    Args:
        trajectory: The trajectory containing selected key indices for all batch items
        
    Returns:
        float: Shannon entropy of the selection order distribution
    """
    # Redundant local imports removed – all symbols available at module level
    # import math
    # from collections import Counter
    # import torch

    if not trajectory.qkv_steps:
        raise ValueError("Trajectory must contain qkv_steps")

    if not isinstance(trajectory.qkv_steps[0].selected_idx, torch.Tensor):
        raise TypeError("selected_idx must be a torch.Tensor")

    if trajectory.qkv_steps[0].selected_idx.numel() == 0:
        raise ValueError("selected_idx tensor is empty")

    batch_size = trajectory.qkv_steps[0].selected_idx.shape[0]

    if batch_size <= 1:
        return 0.0
    
    all_batch_sequences = [] # List of tuples, each inner tuple is a sequence for one batch item
    
    # For each batch item, build its selection sequence of scalar indices
    for b in range(batch_size):
        sequence = []
        for step in trajectory.qkv_steps:
            if hasattr(step, 'selected_idx') and isinstance(step.selected_idx, torch.Tensor):
                sequence.append(step.selected_idx[b].item())
        all_batch_sequences.append(tuple(sequence)) # Convert to tuple for hashing
    
    if not all_batch_sequences:
        return 0.0
    
    # Count unique sequences and their frequencies
    sequence_counts = Counter(all_batch_sequences)
    
    # Compute Shannon entropy
    total_sequences = len(all_batch_sequences)
    entropy_val = 0.0
    
    for count in sequence_counts.values():
        if count > 0:
            p = count / total_sequences
            entropy_val -= p * math.log2(p) # Using log2 for bits
    
    # Normalize by maximum possible entropy, which is log2(batch_size) if all sequences are unique
    max_entropy = math.log2(batch_size) 
    normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


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


def compute_kl_from_reference(
    trajectory: "Trajectory",
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
) -> float:
    """Compute KL(adapter || ref) over the key-selection distribution for an entire trajectory.

    The adapter distribution is already stored in ``step.similarity_scores``.
    We rebuild the reference distribution by generating a query vector with ``ref_model``
    at each timestep and re-using the stored key embeddings / availability mask.

    Returns
    -------
    float
        Average KL divergence across all steps in the trajectory (batch-mean inside each step).
    """
    all_key_embs = getattr(trajectory, "all_key_embeddings", None)
    if not isinstance(all_key_embs, torch.Tensor):
        raise TypeError(f"trajectory.all_key_embeddings must be a torch.Tensor, got {type(all_key_embs)}")
    


    # Ensure context_tokens from tokenizer is a real tensor (unit-tests may return MagicMocks)
    device = next(adapter_model.parameters()).device
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]

    context_tokens_obj = tokenizer(
        [CONFIG.initial_prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    if not hasattr(context_tokens_obj, "input_ids") or not isinstance(context_tokens_obj.input_ids, torch.Tensor):
        raise TypeError("Tokenizer must return an object with a tensor 'input_ids' field")

    context_tokens = context_tokens_obj.input_ids.to(device)

    kl_vals = []

    # Iterate step-by-step so we respect the autoregressive context growth
    for step in trajectory.qkv_steps:
        # 1) Adapter distribution (already computed)
        adapter_log_probs = step.similarity_scores

        # 2) Reference distribution – build a query vector, compute similarities, apply mask
        with torch.no_grad():
            ref_query = generate_query_vector(ref_model, tokenizer, context_tokens, layer_idx=-2)
            key_embs_full = all_key_embs.to(device)
            # Get attention parameters from the reference model
            num_heads, num_groups, head_dim = get_attention_params(ref_model)
            # Apply availability mask inside compute_similarity for proper probability distribution
            availability_mask = step.available_mask if hasattr(step, "available_mask") else None
            ref_sims = compute_similarity(ref_query, key_embs_full, num_heads, num_groups, head_dim,
                                        availability_mask=availability_mask)  # [B, K]

        ref_log_probs = ref_sims

        # 3) KL(adapter || ref) in log-space.  "log_target=True" expects both inputs are log-probs.
        kl_step = F.kl_div(adapter_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        kl_vals.append(kl_step.item())

        # 4) Advance context for next timestep
        kp = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        vp = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        context_tokens = torch.cat([
            context_tokens,
            kp,
            step.key_tokens.to(device),
            vp,
            step.value_tokens.to(device),
        ], dim=1)

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
        logging.info("Reward computation: Using raw adapter log probabilities (GRPO handles baselines)")
    
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
        optimizer = optim.Adam(adapter_model.parameters(), lr=CONFIG.training_config.learning_rate)
        
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
                start_episode = args.episodes - 1
                logging.info(f"Resumed from latest checkpoint, continuing from episode {start_episode}")
            else:
                # Fall back to checking numbered checkpoints
                for episode in range(args.episodes, 0, -1):
                    if load_checkpoint(adapter_model, episode):
                        start_episode = episode
                        logging.info(f"Resumed from episode {start_episode}")
                        break
        
        # Create a copy of the adapter model to serve as the old model (pi_old)
        # This will be updated every BASELINE_UPDATE_FREQUENCY episodes
        if CONFIG.training_config.memory_efficient_lora:
            # Use memory-efficient LoRA state management
            lora_manager = MemoryEfficientLoRAManager(adapter_model)
            old_model = None  # Not needed with memory-efficient training
            old_embeddings_dict = None
            old_hook_remover = lambda: None  # No-op function
            logging.info("🚀 Using memory-efficient LoRA state management")
        else:
            # Traditional approach with full model copy
            old_model = create_model_copy(adapter_model)
            
            # Add small random noise to old_model parameters to avoid identical initialization
            # This ensures PPO ratios are not 1.0 from the very first episode
            with torch.no_grad():
                for name, param in old_model.named_parameters():
                    if 'lora' in name and param.requires_grad:
                        # Add small Gaussian noise (std=0.01) to LoRA parameters only
                        noise = torch.randn_like(param) * 0.01
                        param.data.add_(noise)
            
            if args.verbose:
                print("Added small initialization noise to old_model LoRA parameters")
            
            # Register embedding hook for the old model (for KEY embeddings)
            old_embeddings_dict, old_hook_remover = register_embedding_hook(old_model, embed_type="key")
            lora_manager = None  # Not needed with traditional training
            logging.info("Using traditional model copying approach")

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
        
        # Determine if we're using GRPO-style batching
        use_grpo_batching = args.grpo_batching
        
        if use_grpo_batching:
            # GRPO approach: repeat each data point batch_size times
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
                
            # Repeat each item batch_size times for GRPO-style batching
            kv_pair_generator: Iterator[KVPair] = cast(Iterator[KVPair], repeat_n_times(args.batch_size, base_iterator))
            
            if args.debug_generators:
                kv_pair_generator = cast(Iterator[KVPair], debug_stream(kv_pair_generator, "repeated_kv_pairs", max_items=args.batch_size + 1))
        else:
            # Standard approach: different items in each batch position
            kv_pair_generator = iter_key_value_pairs_unified_with_tokenizer(
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                embedding_fn=compute_key_embedding,
            )
            
            # Add debugging if requested
            if args.debug_generators:
                logging.info("Debug mode enabled for generators")
                kv_pair_generator = cast(Iterator[KVPair], peek_stream(kv_pair_generator, peek_count=1))
                kv_pair_generator = cast(Iterator[KVPair], debug_stream(kv_pair_generator, "standard_kv_pairs", max_items=2))
                kv_pair_generator = cast(Iterator[KVPair], time_stream(kv_pair_generator, "kv_generation"))
        
        # Get reference to the base model (pi_ref)
        ref_model = base_model
        
        # Log model setup
        logging.info("Model setup complete:")
        logging.info("- adapter_model: LoRA adapter (trainable)")
        logging.info("- ref_model: Reference model (pi_ref, for reward computation)")
        logging.info("- old_model: Old model (pi_old, for KL computation, updated periodically)")
        logging.info(f"- GRPO batching: {'Enabled' if use_grpo_batching else 'Disabled'}")
        # No separate previous_model - use old_model for KL computation
        logging.info(f"Old model will be updated every {CONFIG.training_config.baseline_update_frequency} episodes")
        
        # Log baseline update method
        if CONFIG.training_config.use_ema_baseline:
            logging.info(f"✅ Using EMA baseline updates (decay={CONFIG.training_config.ema_decay:.3f}) - eliminates spikes")
        else:
            logging.info(f"⚠️  Using HARD baseline updates - may cause training spikes every {CONFIG.training_config.baseline_update_frequency} episodes")
        
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

        def compute_advantage_distribution(advantages):
            """
            Compute the distribution of positive vs negative advantages.
            
            Args:
                advantages: Tensor of shape [batch, steps]
                
            Returns:
                Dict with positive_percentage, negative_percentage, zero_percentage
            """
            total_advantages = advantages.numel()
            positive_count = (advantages > 0).sum().item()
            negative_count = (advantages < 0).sum().item()
            zero_count = (advantages == 0).sum().item()
            
            return {
                'positive_percentage': positive_count / total_advantages * 100,
                'negative_percentage': negative_count / total_advantages * 100,
                'zero_percentage': zero_count / total_advantages * 100,
                'mean': advantages.mean().item(),
                'std': advantages.std().item()
            }

        def compute_similarity_score_stats(trajectory):
            """
            Compute statistics about similarity scores in a trajectory.
            
            Args:
                trajectory: Trajectory object with qkv_steps
                
            Returns:
                Dict with mean, std, entropy, max, min of similarity scores
            """
            all_similarities = []
            
            for step in trajectory.qkv_steps:
                if hasattr(step, 'similarity_scores') and step.similarity_scores is not None:
                    similarities = step.similarity_scores
                    probs = torch.exp(similarities)
                    
                    all_similarities.append({
                        'mean': probs.mean().item(),
                        'std': probs.std().item(),
                        'entropy': -(probs * similarities).sum(dim=-1).mean().item(),
                        'max': probs.max().item(),
                        'min': probs.min().item()
                    })
            
            if not all_similarities:
                return {'mean': 0.0, 'std': 0.0, 'entropy': 0.0, 'max': 0.0, 'min': 0.0}
            
            # Average across all steps
            return {
                'mean': sum(s['mean'] for s in all_similarities) / len(all_similarities),
                'std': sum(s['std'] for s in all_similarities) / len(all_similarities),
                'entropy': sum(s['entropy'] for s in all_similarities) / len(all_similarities),
                'max': sum(s['max'] for s in all_similarities) / len(all_similarities),
                'min': sum(s['min'] for s in all_similarities) / len(all_similarities)
            }
        
        # Training loop
        logging.info("Starting training...")
        episodes_range = range(start_episode, args.episodes)
        progress_bar = tqdm(episodes_range)
        
        # Initialize plotting data structure (replaces global variables)
        plot_data = PlotData()
        
        # Training loop
        reward_history = []
        loss_history = []
        gradient_history = []
        weight_changes = []  # Track weight changes over time
        policy_gradients = []  # Track policy gradients (before negation) for conceptual clarity
        clipping_ratios = []  # Track PPO clipping ratios
        kl_from_ref = []  # Track KL divergence from reference model (pi_ref)
        
        for episode in progress_bar:
            if args.verbose:
                print(f"\n\n======== EPISODE {episode}/{args.episodes} ========")
            
            # Get a batch of key-value pairs
            available_qkv_steps = [next(kv_pair_generator) for _ in range(CONFIG.training_config.num_kv_pairs)]  # Get a pool of QKV steps
            
            if args.verbose:
                print(f"Generated pool of {len(available_qkv_steps)} query-key-value steps")
            
            # Create initial context with a prompt explaining the task
            # Note: The token count of this prompt is accounted for in 
            # the TrainingConfig calculation to ensure we don't exceed the context window
            batch_size = args.batch_size
            
            # Tokenize the initial prompt
            device = next(adapter_model.parameters()).device
            initial_tokens = tokenizer(
                [CONFIG.initial_prompt] * batch_size,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).input_ids.to(device)
            
            # Generate a *raw* trajectory (no rewards yet)
            raw_traj = generate_trajectory(
                context_tokens=initial_tokens,
                adapter_model=adapter_model,
                tokenizer=tokenizer,
                available_qkv_steps=available_qkv_steps,
                batch_size=args.batch_size,
                config=CONFIG.training_config,
                verbose=args.verbose,
            )
            
            # Promote raw -> full trajectory and compute log probs using reference model
            trajectory, adapter_log_probs_batch, ref_log_probs_batch = compute_trajectory_rewards(
                raw_traj, 
                adapter_model, 
                ref_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=args.verbose
            )
            
            # Also get old model log probabilities for comparison plotting
            if not CONFIG.training_config.memory_efficient_lora:
                # Only compute old model log probs for traditional mode (for plotting)
                assert old_model is not None, "old_model should not be None in traditional mode"
                _, _, old_log_probs_batch = compute_trajectory_rewards(
                    raw_traj, 
                    adapter_model, 
                    old_model, 
                    initial_tokens,
                    tokenizer=tokenizer,
                    verbose=False  # Don't print twice
                )
            else:
                # For memory-efficient mode, use adapter log probs as old log probs
                old_log_probs_batch = adapter_log_probs_batch

            # Exact KL(adapter || reference) over key-selection distribution
            kl_from_ref_value = compute_kl_from_reference(
                trajectory,
                adapter_model,
                ref_model,
                tokenizer,
            )
            kl_from_ref.append(kl_from_ref_value)
            
            # Update reward stats
            reward_stats = update_reward_stats(reward_stats, trajectory.avg_reward)
            if args.verbose:
                print(f"\nUpdated reward stats:")
                print(f"  Mean: {reward_stats['mean']:.4f}")
                print(f"  Std: {reward_stats['std']:.4f}")
                print(f"  Count: {reward_stats['count']}")
            
            # Perform training step
            if CONFIG.training_config.memory_efficient_lora:
                # Use memory-efficient training step
                assert lora_manager is not None, "lora_manager should not be None in memory-efficient mode"
                total_loss, policy_loss, kl_loss, avg_clipping_ratio = memory_efficient_train_step(
                    trajectory,
                    adapter_model,
                    ref_model,  # Use ref_model for reward computation
                    lora_manager,  # Use LoRA manager instead of old_model
                    optimizer,
                    reward_stats,
                    CONFIG.training_config.kl_penalty_coefficient,
                    verbose=args.verbose,
                    tokenizer=tokenizer,
                    embeddings_dict=query_embeddings_dict,
                )
            else:
                # Use traditional training step
                assert old_model is not None, "old_model should not be None in traditional mode"
                total_loss, policy_loss, kl_loss, avg_clipping_ratio = train_step(
                    trajectory,
                    adapter_model,
                    ref_model,  # Use ref_model for reward computation
                    old_model,  # Use old_model for KL computation
                    optimizer,
                    reward_stats,
                    CONFIG.training_config.kl_penalty_coefficient,
                    verbose=args.verbose,
                    tokenizer=tokenizer,
                    embeddings_dict=query_embeddings_dict, # Use query embeddings for training
                )
            
            # Track clipping ratio
            clipping_ratios.append(avg_clipping_ratio)
            
            # Track weight change BEFORE updating old_model
            weight_change = compute_weight_change(adapter_model, old_model)
            weight_changes.append(weight_change)
            
            # NO automatic old_model update after each training step - only at intervals
            # This allows KL divergence to accumulate over multiple episodes for meaningful regularization
            
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
                    
                    print("\nFirst batch item selections (first 5 steps):")
                    for info in trajectory_info['selections'][:5]:  # Show first 5
                        print(f"  Step {info['step']}: idx={info['selected_idx']}")
                        if info['key_text']:
                            print(f"    Key: {info['key_text'][:50]}...")
                        if info['value_text']:
                            print(f"    Value: {info['value_text'][:50]}...")
            
            # Calculate derived metrics for plotting
            traj_log_prob = adapter_log_probs_batch.mean().item()
            kl_penalty_term = (kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss) * CONFIG.training_config.kl_penalty_coefficient
            reward_var = trajectory.avg_reward.var().item()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute and track average advantages from this episode
            if trajectory.rewards is not None:
                from src.training import compute_advantages
                pass  # All config values now accessed via CONFIG
                advantages, _ = compute_advantages(
                    trajectory.rewards,
                    gamma=CONFIG.gamma,
                    gae_lambda=CONFIG.gae_lambda,
                    use_grpo_baseline=CONFIG.training_config.use_grpo_baseline,
                )
                avg_advantage = advantages.mean().item()
                advantage_dist = compute_advantage_distribution(advantages)
            else:
                avg_advantage = 0.0
                advantage_dist = {
                    'positive_percentage': 0.0, 'negative_percentage': 0.0, 'zero_percentage': 100.0,
                    'mean': 0.0, 'std': 0.0
                }
            
            # Policy gradient for visualization (before negation)
            policy_gradient = -(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss)
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode}/{args.episodes}, "
                f"Loss: {total_loss:.4f}, "

                f"Reward: {avg_reward:.4f}"
            )
            
            # Update old_model using EMA (smooth) updates to prevent gradient spikes
            # Based on debugging: frozen old_model causes accumulated changes -> gradient explosions
            if CONFIG.training_config.memory_efficient_lora:
                # Use memory-efficient LoRA state updates
                assert lora_manager is not None, "lora_manager should not be None in memory-efficient mode"
                if CONFIG.training_config.use_ema_baseline:
                    lora_manager.update_old_state_ema(decay=CONFIG.training_config.ema_decay)
                    if episode % CONFIG.training_config.baseline_update_frequency == 0:
                        logging.info(f"Memory-efficient EMA LoRA state update (decay={CONFIG.training_config.ema_decay:.3f})")
                else:
                    # Hard update at intervals
                    if (episode + 1) % CONFIG.training_config.baseline_update_frequency == 0:
                        lora_manager.update_old_state_hard()
                        logging.info("Memory-efficient hard LoRA state update")
            else:
                # Traditional model updates
                assert old_model is not None, "old_model should not be None in traditional mode"
                if CONFIG.training_config.use_ema_baseline:
                    # Smooth EMA update every episode - prevents spikes
                    update_model_ema(old_model, adapter_model, decay=CONFIG.training_config.ema_decay)
                    if episode % CONFIG.training_config.baseline_update_frequency == 0:
                        logging.info(f"Smooth EMA baseline update (decay={CONFIG.training_config.ema_decay:.3f}) - prevents gradient spikes")
                else:
                    # Hard update at intervals - may cause gradient spikes
                    if (episode + 1) % CONFIG.training_config.baseline_update_frequency == 0:
                        # Update the old_model to reflect learning progress
                        old_model = create_model_copy(adapter_model)
                        
                        # Re-register the embedding hook for the new old_model (for KEY embeddings)
                        old_hook_remover()  # Remove old hook
                        old_embeddings_dict, old_hook_remover = register_embedding_hook(old_model, embed_type="key")
                        
                        # NO need to recreate kv_pair_generator; embeddings will be recomputed on-the-fly
                        logging.info("Old_model updated; KV generator preserved to avoid data repetition")
            
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
            # Add episode data to plot_data using the clean functional approach
            trajectory_sample = trajectory_info if (episode % 50 == 0 or episode < 5) and 'trajectory_info' in locals() else None
            plot_data = plot_data.add_episode_data(
                episode=episode,
                total_loss=total_loss,
                policy_loss=policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss,
                kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
                avg_reward=avg_reward,
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
                clipping_ratio=avg_clipping_ratio,
                batch_selection_entropy=selection_entropy,
                kl_from_ref_value=kl_from_ref_value,
                lora_layer_gradients_episode=layer_grads,
                advantage_distribution=advantage_dist,
                similarity_score_stats=similarity_stats,
                policy_gradient=policy_gradient,
                trajectory_sample=trajectory_sample
            )
            
            # Save and plot metrics periodically
            if episode > 0 and episode % 15 == 0:
                # Add metadata to plot data and save
                metadata = create_metadata(episode, {
                    'KL_PENALTY_COEFFICIENT': CONFIG.training_config.kl_penalty_coefficient,
                    'GAMMA': CONFIG.gamma,
                    'GAE_LAMBDA': CONFIG.gae_lambda,
                    'USE_GRPO_BASELINE': CONFIG.training_config.use_grpo_baseline,
                    'TEMPERATURE': CONFIG.training_config.temperature,
                    'NUM_KV_PAIRS': CONFIG.training_config.num_kv_pairs,
                    'BASELINE_UPDATE_FREQUENCY': CONFIG.training_config.baseline_update_frequency,
                })
                plot_data_with_metadata = plot_data.with_metadata(metadata)
                save_plot_data(plot_data_with_metadata, log_dir)
                plot_metrics(log_dir, policy_gradients)
            
            # Log every log_interval episodes
            if episode % CONFIG.training_config.log_interval == 0:
                # Convert tensors to floats for logging
                policy_loss_val = policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
                kl_loss_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                
                logging.info(
                    f"Episode {episode}/{args.episodes}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Policy Loss: {policy_loss_val:.4f}, "
                    f"KL Loss: {kl_loss_val:.4f}, "

                    f"Reward: {avg_reward:.4f}, "
                    f"Reward Mean: {reward_stats['mean']:.4f}, "
                    f"Reward Std: {reward_stats['std']:.4f}"
                )
                
                # Log to wandb if enabled
                if CONFIG.training_config.enable_wandb:
                    wandb.log({
                        "episode": episode,
                        "total_loss": total_loss,
                        "policy_loss": policy_loss_val,
                        "kl_loss": kl_loss_val,
                        "kl_penalty_term": kl_loss_val * CONFIG.training_config.kl_penalty_coefficient,
                        "reward": avg_reward,
                        "reward_mean": reward_stats["mean"],
                        "reward_std": reward_stats["std"],

                    })
                
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
        final_metadata = create_metadata(episode, {
            'KL_PENALTY_COEFFICIENT': CONFIG.training_config.kl_penalty_coefficient,
            'GAMMA': CONFIG.gamma,
            'GAE_LAMBDA': CONFIG.gae_lambda,
            'USE_GRPO_BASELINE': CONFIG.training_config.use_grpo_baseline,
            'TEMPERATURE': CONFIG.training_config.temperature,
            'NUM_KV_PAIRS': CONFIG.training_config.num_kv_pairs,
            'BASELINE_UPDATE_FREQUENCY': CONFIG.training_config.baseline_update_frequency,
        })
        final_plot_data = plot_data.with_metadata(final_metadata)
        save_plot_data(final_plot_data, log_dir)
        plot_metrics(log_dir, None)
        
        logging.info("Training complete!")
        
        # Close wandb if enabled
        if CONFIG.training_config.enable_wandb:
            wandb.finish()
    
    finally:
        # Remove hooks
        query_hook_remover()
        key_hook_remover()
        if 'old_hook_remover' in locals():
            old_hook_remover()
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
    
    # Get the absolute path to generate_plots.py
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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