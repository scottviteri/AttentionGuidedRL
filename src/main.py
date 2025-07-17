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
from typing import List, Any
import sys

from src.config import (
    MODEL_NAME,
    DEVICE,
    LEARNING_RATE,
    NUM_EPISODES,
    CHECKPOINT_INTERVAL,
    TRAINING_BATCH_SIZE,
    KL_PENALTY_COEFFICIENT,
    LOG_INTERVAL,
    ENABLE_WANDB,
    INITIAL_PROMPT,
    KEY_PREFIX,
    VALUE_PREFIX,
    NUM_KV_PAIRS,
    BASELINE_UPDATE_FREQUENCY,
    SUBTRACT_BASE_MODEL_LOGPROBS,
    TEMPERATURE,
)
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path
from src.data import KVPair, QKVSelection
from src.embeddings import register_embedding_hook, compute_similarity, sample_key_value, extract_embeddings
from src.training import (
    RawTrajectory,
    Trajectory,  # for type hints
    compute_trajectory_rewards,
    update_reward_stats,
    train_step,
    generate_query_vector,
)

import wandb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Lists to store metrics for plotting
training_steps = []
total_losses = []
policy_losses = []
kl_losses = []
avg_rewards = []
adapter_log_probs = []
baseline_log_probs = []
base_log_probs = []  # Base model log probabilities

# NEW: Additional metrics for comprehensive plotting
avg_advantages = []  # Original advantages (including negative)
trajectory_log_probs = []
wikipedia_order_consistency = []  # Order consistency metric (1.0 = perfect order, 0.0 = perfect reverse)
entropy_values = []  # Policy entropy for exploration tracking
kl_penalty_terms = []  # Actual KL penalty terms added to loss
reward_variance = []  # Variance of rewards within each trajectory
gradient_magnitudes = []  # Gradient magnitudes over time
step_log_probs = []  # Log probabilities by step index (list of lists)
clipping_ratios = []  # PPO clipping ratios
kl_from_ref = []  # KL divergence from reference model (pi_ref)
batch_selection_entropy = []  # Entropy of key selection orders within each batch
trajectory_samples = []  # Sample trajectories for detailed analysis

def setup_logging(args):
    """
    Set up logging for the training run.
    
    Args:
        args: Command-line arguments
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
    
    # Log basic info
    logging.info(f"Starting training run with configuration:")
    logging.info(f"  Model: {MODEL_NAME}")
    logging.info(f"  Device: {DEVICE}")
    logging.info(f"  Dataset: {args.dataset}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {LEARNING_RATE}")
    logging.info(f"  Episodes: {NUM_EPISODES}")
    
    # Initialize wandb if enabled
    if ENABLE_WANDB:
        wandb_config = {
            "learning_rate": args.learning_rate,
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "kl_penalty": KL_PENALTY_COEFFICIENT,
            "num_kv_pairs": NUM_KV_PAIRS,
        }
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
    key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
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
    for _ in range(NUM_KV_PAIRS):
        # 1) Build query embedding
        query_emb = generate_query_vector(adapter_model, tokenizer, current_context)

        # 2) Compute similarities
        similarity_scores = compute_similarity(query_emb, traj.all_key_embeddings, adapter_model)

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
            selected_key_tokens.append(kv.key_tokens[0])
            selected_value_tokens.append(kv.value_tokens[0])
            selected_key_embeddings.append(kv.key_embedding[0])
            selected_key_texts.append(kv.key_text[0])
            selected_value_texts.append(kv.value_text[0])

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

        # Use large negative value instead of -inf to keep log-softmax finite
        available_mask = _build_available_mask(available_indices_per_batch, num_keys, device).clamp(min=-1e9)

        qkv_step = QKVSelection(
            data=step_data,
            query_embedding=query_emb,
            similarity_scores=similarity_scores,
            selected_idx=torch.tensor(selected_indices, device=device),
            available_mask=available_mask,
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
    """Parse command-line arguments."""
    # TRAINING_BATCH_SIZE already imported at module level – redundant
    
    parser = argparse.ArgumentParser(description="Train a model using Attention-Guided RL")
    parser.add_argument("--batch-size", type=int, default=TRAINING_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to train")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose trajectory logging")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for training")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--dataset", type=str, default="wikipedia", 
                        choices=["wikipedia", "twenty_questions"],
                        help="Dataset to use for training")
    parser.add_argument("--grpo-batching", action="store_true", default=True,
                        help="Use GRPO-style batching (repeat each data point) (default: True)")
    parser.add_argument("--model-type", type=str, default='gpt2', choices=['gpt2', 'llama'], help='Model type to use')
    parser.add_argument('--use-grpo-baseline', action='store_true', default=True, help='Use GRPO baseline in advantages')
    
    # Add new CLI flags for previously env-var configs
    parser.add_argument('--key-embedding-batch-size', type=int, default=4, help='Number of keys to process together in forward pass')
    parser.add_argument('--kl-penalty-coef', type=float, default=0.1, help='KL penalty coefficient for regularization')
    parser.add_argument('--enable-wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    parser.add_argument('--ppo-clip-epsilon', type=float, default=0.2, help='PPO clipping parameter (epsilon)')
    parser.add_argument('--baseline-update-freq', type=int, default=10, help='How often to update baseline model (episodes)')
    parser.add_argument('--subtract-base-logprobs', action='store_true', default=False, help='Subtract base model logprobs in reward computation')
    
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
        raise TypeError("trajectory.all_key_embeddings must be a torch.Tensor")

    # Ensure context_tokens from tokenizer is a real tensor (unit-tests may return MagicMocks)
    device = next(adapter_model.parameters()).device
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]

    context_tokens_obj = tokenizer(
        [INITIAL_PROMPT] * batch_size,
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
        adapter_log_probs = F.log_softmax(step.similarity_scores / TEMPERATURE, dim=-1)

        # 2) Reference distribution – build a query vector, compute similarities, apply mask
        with torch.no_grad():
            ref_query = generate_query_vector(ref_model, tokenizer, context_tokens, layer_idx=-2)
            key_embs_full = all_key_embs.to(device)
            ref_sims = compute_similarity(ref_query, key_embs_full, ref_model)  # [B, K]

        if hasattr(step, "available_mask") and step.available_mask is not None:
            ref_sims = ref_sims + step.available_mask.to(device)

        ref_log_probs = F.log_softmax(ref_sims / TEMPERATURE, dim=-1)

        # 3) KL(adapter || ref) in log-space.  "log_target=True" expects both inputs are log-probs.
        kl_step = F.kl_div(ref_log_probs, adapter_log_probs, reduction="batchmean", log_target=True)
        kl_vals.append(kl_step.item())

        # 4) Advance context for next timestep
        kp = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        vp = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
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
    # Parse arguments
    args = parse_args()
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
    
    # Set up logging
    log_dir = setup_logging(args)
    
    # Log query mode
    logging.info("Query mode: Vector queries")
    
    # Log reward computation mode
    if SUBTRACT_BASE_MODEL_LOGPROBS:
        logging.info("Reward computation: Using adapter - base model (classic baseline subtraction)")
    else:
        logging.info("Reward computation: Using raw adapter log probabilities (GRPO handles baselines)")
    
    # Set up models and tokenizer
    logging.info("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Log the dynamically calculated token counts
    import src.config as config
    logging.info(f"Token count configuration:")
    logging.info(f"  Key prefix tokens: {config.PREFIX_TOKENS_PER_KEY}")
    logging.info(f"  Value prefix tokens: {config.PREFIX_TOKENS_PER_VALUE}")
    logging.info(f"  Total tokens per round: {config.TOKENS_PER_ROUND}")
    logging.info(f"  Initial prompt tokens: {config.INITIAL_PROMPT_TOKENS}")
    logging.info(f"  Number of KV pairs: {config.NUM_KV_PAIRS}")
    
    # Set model type and names based on args
    config.MODEL_TYPE = args.model_type
    if config.MODEL_TYPE == 'llama':
        config.MODEL_NAME = 'meta-llama/Llama-3.2-3B'
        config.TOKENIZER_NAME = 'meta-llama/Llama-3.2-3B'
    elif config.MODEL_TYPE == 'gpt2':
        config.MODEL_NAME = 'gpt2'
        config.TOKENIZER_NAME = 'gpt2'
    else:
        raise ValueError(f'Invalid model type: {config.MODEL_TYPE}')

    config.USE_GRPO_BASELINE = args.use_grpo_baseline
    
    # Set additional config values from CLI args
    config.KEY_EMBEDDING_BATCH_SIZE = args.key_embedding_batch_size
    config.KL_PENALTY_COEFFICIENT = args.kl_penalty_coef
    config.ENABLE_WANDB = args.enable_wandb
    config.PPO_CLIP_EPSILON = args.ppo_clip_epsilon
    config.BASELINE_UPDATE_FREQUENCY = args.baseline_update_freq
    config.SUBTRACT_BASE_MODEL_LOGPROBS = args.subtract_base_logprobs
    
    # Separate hooks: one for query (train-time) and one for key (data loading)
    query_embeddings_dict, query_hook_remover = register_embedding_hook(adapter_model, embed_type="query")
    key_embeddings_dict, key_hook_remover   = register_embedding_hook(adapter_model, embed_type="key")

    # Helper to compute key embeddings once during data loading using the key-specific hook
    def compute_key_embedding(key_token_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return extract_embeddings(
                adapter_model,
                key_token_batch.to(DEVICE),
                key_embeddings_dict,
                requires_grad=False,
            ).detach()
    
    # Make sure hook is removed at the end
    try:
        # Create optimizer
        optimizer = optim.Adam(adapter_model.parameters(), lr=args.learning_rate)
        
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
        old_model = create_model_copy(adapter_model)
        
        # Register embedding hook for the old model (for KEY embeddings)
        old_embeddings_dict, old_hook_remover = register_embedding_hook(old_model, embed_type="key")
        
        # Import the data iterator and repeat function
        from src.data import iter_key_value_pairs_unified_with_tokenizer, repeat_n_times
        
        # Determine if we're using GRPO-style batching
        use_grpo_batching = args.grpo_batching
        
        if use_grpo_batching:
            # GRPO approach: repeat each data point batch_size times
            # This creates a batch where each unique item appears multiple times
            base_iterator = iter_key_value_pairs_unified_with_tokenizer(
                dataset_name=args.dataset,
                batch_size=1,  # Generate single items
                tokenizer=tokenizer,
                embedding_fn=compute_key_embedding,
            )
            # Repeat each item batch_size times for GRPO-style batching
            kv_pair_generator = repeat_n_times(args.batch_size, base_iterator)
        else:
            # Standard approach: different items in each batch position
            kv_pair_generator = iter_key_value_pairs_unified_with_tokenizer(
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                embedding_fn=compute_key_embedding,
            )
        
        # Get reference to the base model (pi_ref)
        ref_model = base_model
        
        # Log model setup
        logging.info("Model setup complete:")
        logging.info("- adapter_model: LoRA adapter (trainable)")
        logging.info("- ref_model: Reference model (pi_ref, for reward computation)")
        logging.info("- old_model: Old model (pi_old, for KL computation, updated periodically)")
        logging.info(f"- GRPO batching: {'Enabled' if use_grpo_batching else 'Disabled'}")
        # No separate previous_model - use old_model for KL computation
        logging.info(f"Old model will be updated every {BASELINE_UPDATE_FREQUENCY} episodes")
        
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
        
        # Training loop
        logging.info("Starting training...")
        episodes_range = range(start_episode, args.episodes)
        progress_bar = tqdm(episodes_range)
        
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
            available_qkv_steps = [next(kv_pair_generator) for _ in range(NUM_KV_PAIRS)]  # Get a pool of QKV steps
            
            if args.verbose:
                print(f"Generated pool of {len(available_qkv_steps)} query-key-value steps")
            
            # Create initial context with a prompt explaining the task
            # Note: The token count of this prompt is accounted for in 
            # the NUM_KV_PAIRS calculation in config.py to ensure we don't exceed the context window
            batch_size = args.batch_size
            
            # Tokenize the initial prompt
            device = next(adapter_model.parameters()).device
            initial_tokens = tokenizer(
                [INITIAL_PROMPT] * batch_size,
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
            _, _, old_log_probs_batch = compute_trajectory_rewards(
                raw_traj, 
                adapter_model, 
                old_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=False  # Don't print twice
            )

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
            total_loss, policy_loss, kl_loss, avg_clipping_ratio = train_step(
                trajectory,
                adapter_model,
                ref_model,  # Use ref_model for reward computation
                old_model,  # Use old_model for KL computation
                optimizer,
                reward_stats,
                KL_PENALTY_COEFFICIENT,
                verbose=args.verbose,
                tokenizer=tokenizer,
                embeddings_dict=query_embeddings_dict # Use query embeddings for training
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
                
            # Store metrics for plotting
            training_steps.append(episode)
            total_losses.append(total_loss)
            policy_losses.append(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss)
            kl_losses.append(kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss)
            avg_rewards.append(avg_reward)
            
            # NEW: Track additional metrics for comprehensive plotting
            
            # Compute Wikipedia order consistency metric
            if args.dataset == "wikipedia":
                order_consistency = compute_wikipedia_order_consistency(trajectory)
                wikipedia_order_consistency.append(order_consistency)
            else:
                wikipedia_order_consistency.append(0.5)  # Neutral for non-Wikipedia datasets
            
            # Compute batch selection entropy
            selection_entropy = compute_batch_selection_entropy(trajectory)
            batch_selection_entropy.append(selection_entropy)
            
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
                
                trajectory_samples.append(trajectory_info)
                
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
            
            # Track trajectory-level log probabilities (average across all trajectory steps)
            traj_log_prob = adapter_log_probs_batch.mean().item()
            trajectory_log_probs.append(traj_log_prob)
            
            # Track KL penalty terms (actual penalty added to loss)
            kl_penalty_term = (kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss) * KL_PENALTY_COEFFICIENT
            kl_penalty_terms.append(kl_penalty_term)
            
            # Track reward variance within trajectory
            reward_var = trajectory.avg_reward.var().item()
            reward_variance.append(reward_var)
            
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute and track average advantages from this episode
            # We'll compute this from the trajectory using the same logic as in training
            if trajectory.rewards is not None:
                from src.training import compute_advantages
                from src.config import GAMMA, GAE_LAMBDA, USE_GRPO_BASELINE
                advantages, _ = compute_advantages(
                    trajectory.rewards,
                    gamma=GAMMA,
                    use_grpo_baseline=USE_GRPO_BASELINE,
                )
                avg_advantage = advantages.mean().item()
            else:
                avg_advantage = 0.0
            avg_advantages.append(avg_advantage)
            
            # For conceptual clarity, also track the policy gradient (before negation)
            # This helps visualize what we're actually reinforcing
            policy_gradient = -(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss)
            policy_gradients.append(policy_gradient)
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode}/{args.episodes}, "
                f"Loss: {total_loss:.4f}, "

                f"Reward: {avg_reward:.4f}"
            )
            
            # Update old_model at configurable frequency
            if (episode + 1) % BASELINE_UPDATE_FREQUENCY == 0:
                # Update the old_model to reflect learning progress
                old_model = create_model_copy(adapter_model)
                
                # Re-register the embedding hook for the new old_model (for KEY embeddings)
                old_hook_remover()  # Remove old hook
                old_embeddings_dict, old_hook_remover = register_embedding_hook(old_model, embed_type="key")
                
                # NO need to recreate kv_pair_generator; embeddings will be recomputed on-the-fly
                logging.info("Old_model updated; KV generator preserved to avoid data repetition")
            
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
            if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(adapter_model, "latest")
                if args.verbose:
                    print(f"\nCheckpoint saved at episode {episode}")
                
            # Track gradient statistics
            gradient_stats = get_gradient_stats(adapter_model)
            gradient_history.append(gradient_stats)
            gradient_magnitudes.append(gradient_stats[0])
            
            # Track reward history
            reward_history.append(avg_reward)
            loss_history.append(total_loss)
            
            # Track log probabilities (average across batch and timesteps)
            adapter_log_probs.append(adapter_log_probs_batch.mean().item())
            baseline_log_probs.append(old_log_probs_batch.mean().item())
            base_log_probs.append(ref_log_probs_batch.mean().item())
            
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
                    
                    log_probs = F.log_softmax(masked_similarities / TEMPERATURE, dim=-1)
                    # Get log prob of selected action (average over batch)
                    if not isinstance(step.selected_idx, torch.Tensor):
                        raise TypeError("selected_idx must be a torch.Tensor")
                    selected_idx = step.selected_idx
                    selected_log_prob = log_probs[torch.arange(log_probs.shape[0]), selected_idx].mean().item()
                    step_log_probs_episode.append(selected_log_prob)
            step_log_probs.append(step_log_probs_episode)
            
            # Save and plot metrics more frequently (every 15 episodes) for better monitoring
            if episode > 0 and episode % 15 == 0:
                save_plot_data(log_dir, episode, policy_gradients)
                plot_metrics(log_dir, policy_gradients)
            
            # Log every LOG_INTERVAL episodes
            if episode % LOG_INTERVAL == 0:
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
                if ENABLE_WANDB:
                    wandb.log({
                        "episode": episode,
                        "total_loss": total_loss,
                        "policy_loss": policy_loss_val,
                        "kl_loss": kl_loss_val,
                        "kl_penalty_term": kl_loss_val * KL_PENALTY_COEFFICIENT,
                        "reward": avg_reward,
                        "reward_mean": reward_stats["mean"],
                        "reward_std": reward_stats["std"],

                    })
                
                # Log gradient diagnostics
                if len(gradient_magnitudes) > 0:
                    logging.info(
                        f"  Gradient Stats - Norm: {gradient_magnitudes[-1]:.6f}, "
                        f"Weight Change: {weight_changes[-1]:.6f}"
                    )
                    
                    # Check if weights are updating
                    if weight_changes[-1] < 1e-6:
                        logging.warning("  WARNING: Weights are not changing!")
                    
                    # Check gradient health
                    if gradient_magnitudes[-1] < 1e-8:
                        logging.warning("  WARNING: Gradients are vanishing!")
                    elif gradient_magnitudes[-1] > 100:
                        logging.warning("  WARNING: Gradients are exploding!")
            
        # Save final checkpoint
        save_checkpoint(adapter_model, "latest")
        
        # Save final plot data and create plots
        save_plot_data(log_dir, episode, policy_gradients)
        plot_metrics(log_dir, policy_gradients)
        
        logging.info("Training complete!")
        
        # Close wandb if enabled
        if ENABLE_WANDB:
            wandb.finish()
    
    finally:
        # Remove hooks
        query_hook_remover()
        key_hook_remover()
        if 'old_hook_remover' in locals():
            old_hook_remover()


def save_plot_data(log_dir, episode, policy_gradients_data=None, all_data=None):
    """
    Save all plotting data to a pickle file for later visualization.
    
    Args:
        log_dir: Directory where logs are saved
        episode: Current episode number (for filename)
        policy_gradients_data: Policy gradients data (passed separately)
        all_data: Optional pre-collected data dict, otherwise collect from globals
    """
    import pickle
    import datetime
    from src.config import (
        KL_PENALTY_COEFFICIENT, GAMMA, GAE_LAMBDA, USE_GRPO_BASELINE,
        TEMPERATURE, NUM_KV_PAIRS, BASELINE_UPDATE_FREQUENCY
    )
    
    # Create plots directory
    plots_dir = f"{log_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect all data if not provided
    if all_data is None:
        all_data = {
            'training_steps': training_steps.copy(),
            'total_losses': total_losses.copy(),
            'policy_losses': policy_losses.copy(),
            'kl_losses': kl_losses.copy(),
            'avg_rewards': avg_rewards.copy(),
            'adapter_log_probs': adapter_log_probs.copy(),
            'baseline_log_probs': baseline_log_probs.copy(),
            'base_log_probs': base_log_probs.copy(),
            'avg_advantages': avg_advantages.copy(),
            'trajectory_log_probs': trajectory_log_probs.copy(),
            'wikipedia_order_consistency': wikipedia_order_consistency.copy(),
            'kl_penalty_terms': kl_penalty_terms.copy(),
            'reward_variance': reward_variance.copy(),
            'gradient_magnitudes': gradient_magnitudes.copy(),
            'step_log_probs': step_log_probs.copy(),
            'policy_gradients': policy_gradients_data.copy() if policy_gradients_data else [],
            'clipping_ratios': clipping_ratios.copy(),
            'kl_from_ref': kl_from_ref.copy(),
            'batch_selection_entropy': batch_selection_entropy.copy(),
            'trajectory_samples': trajectory_samples.copy(),  # Add trajectory samples
            # Add metadata
            'metadata': {
                'episode': episode,
                'timestamp': datetime.datetime.now().isoformat(),
                'config': {
                    'KL_PENALTY_COEFFICIENT': KL_PENALTY_COEFFICIENT,
                    'GAMMA': GAMMA,
                    'GAE_LAMBDA': GAE_LAMBDA,
                    'USE_GRPO_BASELINE': USE_GRPO_BASELINE,
                    'TEMPERATURE': TEMPERATURE,
                    'NUM_KV_PAIRS': NUM_KV_PAIRS,
                    'BASELINE_UPDATE_FREQUENCY': BASELINE_UPDATE_FREQUENCY,
                }
            }
        }
    
    # Save to pickle file
    filename = f"{plots_dir}/plot_data_episode_{episode}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)
    
    logging.info(f"Saved plot data to {filename}")
    
    # Also save a "latest" version for easy access
    latest_filename = f"{plots_dir}/plot_data_latest.pkl"
    with open(latest_filename, 'wb') as f:
        pickle.dump(all_data, f)


def plot_metrics(log_dir, policy_gradients_data=None):
    """
    Create and save comprehensive visualization of training metrics by calling generate_plots.py.
    
    Args:
        log_dir: Directory where logs and plots are saved
        policy_gradients_data: List of policy gradient values for plotting (unused, kept for compatibility)
    """
    import subprocess
    
    # Get the latest plot data file
    plots_dir = f"{log_dir}/plots"
    latest_pickle = f"{plots_dir}/plot_data_latest.pkl"
    
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