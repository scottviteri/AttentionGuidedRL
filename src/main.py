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
from tqdm import tqdm
from datetime import datetime
from typing import List, Optional, Dict, Callable, Any, Tuple
import numpy as np
from copy import deepcopy
import sys

from src.config import (
    MODEL_NAME,
    TOKENIZER_NAME,
    DEVICE,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LEARNING_RATE,
    GRADIENT_CLIP_NORM,
    NUM_EPISODES,
    CHECKPOINT_INTERVAL,
    TRAINING_BATCH_SIZE,
    KL_PENALTY_COEFFICIENT,
    CHECKPOINT_DIR,
    LOG_DIR,
    LOG_INTERVAL,
    ENABLE_WANDB,
    WANDB_PROJECT,
    INITIAL_PROMPT,
    KEY_PREFIX,
    VALUE_PREFIX,
    NUM_KV_PAIRS,
    BASELINE_UPDATE_FREQUENCY,
    SUBTRACT_BASE_MODEL_LOGPROBS,
)
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path
from src.data import iter_key_value_pairs, iter_key_value_pairs_unified, QKVStep, repeat_n_times
from src.embeddings import register_embedding_hook, extract_embeddings, compute_similarity, sample_key_value
from src.training import (
    Trajectory,
    compute_trajectory_rewards,
    update_reward_stats,
    train_step,
    generate_query_vector,
)

# Import wandb for logging
import wandb

# Setup matplotlib for plotting
import matplotlib.pyplot as plt
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


def generate_trajectory(
    context_tokens: torch.Tensor,
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer: Any,
    embeddings_dict: Dict,
    hook_remover: Callable,
    available_qkv_steps: List[QKVStep],
    batch_size: int,
    verbose: bool = False,
) -> Trajectory:
    """
    Generate a single trajectory using vector queries.
    
    Args:
        context_tokens: Initial context tokens
        adapter_model: The model with LoRA adapter for generation
        base_model: The base model (unused but kept for interface compatibility)
        tokenizer: The tokenizer
        embeddings_dict: Dictionary for storing embeddings from hooks
        hook_remover: Function to remove hooks
        available_qkv_steps: List of available key-value pairs to choose from
        batch_size: Number of trajectories to generate in parallel
        verbose: Whether to enable verbose logging
        
    Returns:
        Trajectory object containing the selected steps and rewards
    """
    # Ensure the context is on the same device as the model
    device = next(adapter_model.parameters()).device
    current_context = context_tokens.to(device)
    
    # Decode initial context for logging
    context_text = tokenizer.batch_decode(current_context, skip_special_tokens=True)
    
    if verbose:
        print("\n=== Starting New Trajectory ===")
        print(f"Initial context: {context_text[0][:50]}...")
        print(f"Available query-key-value steps: {len(available_qkv_steps)}")
        print(f"Query mode: Vector queries")
    
    # Initialize selected steps list
    selected_steps = []
    
    # Save all key embeddings at the beginning to store at trajectory level
    all_initial_key_embeddings = []
    for qkv_step in available_qkv_steps:
        key_emb = qkv_step.key_embedding.to(device)
        all_initial_key_embeddings.append(key_emb)
    trajectory_key_embeddings = torch.stack(all_initial_key_embeddings, dim=1)  # [batch_size, num_keys, embedding_dim]
    
    # Loop until we've selected a fixed number of steps
    for step_idx in range(NUM_KV_PAIRS):
        # Generate query vector (deterministic mean)
        query_embeddings = generate_query_vector(
            adapter_model,
            tokenizer,
            current_context
        )
        
        # For vector queries, we don't have query tokens or text
        # We'll use placeholders for compatibility
        query_tokens = torch.tensor([[]], device=device).long()  # Empty tensor
        query_text = ["<VECTOR_QUERY>"] * batch_size
        
        # Use pre-computed key embeddings from available steps
        key_embs = []
        for qkv_step in available_qkv_steps:
            # Use the pre-computed embedding, just move to device
            key_emb = qkv_step.key_embedding.to(device)
            # If key_emb has batch size 1 but we need batch_size, broadcast it
            if key_emb.shape[0] == 1 and batch_size > 1:
                key_emb = key_emb.expand(batch_size, -1)  # Broadcast to [batch_size, hidden_size]
            key_embs.append(key_emb)
            
        # Stack key embeddings with shape [batch_size, num_keys, hidden_size]
        key_embeddings = torch.stack(key_embs, dim=1)
        
        # Compute similarity scores
        similarity_scores = compute_similarity(query_embeddings, key_embeddings, adapter_model)
                   
        # Sample next step
        available_indices = list(range(len(available_qkv_steps)))
        sampled_indices, _ = sample_key_value(
            similarity_scores, 
            [available_indices] * batch_size,
            batch_size
        )
        
        # For simplicity, use the first batch item's choice
        selected_idx = sampled_indices[0]
        selected_step = available_qkv_steps[selected_idx]
        
        # Create a copy of the selected step with tensors on the correct device
        # and broadcast to correct batch size if needed
        key_tokens = selected_step.key_tokens.to(device)
        value_tokens = selected_step.value_tokens.to(device)
        key_embedding = selected_step.key_embedding.to(device)
        key_text = selected_step.key_text
        value_text = selected_step.value_text
        
        # If tokens have batch size 1 but we need batch_size, broadcast them
        if key_tokens.shape[0] == 1 and batch_size > 1:
            key_tokens = key_tokens.expand(batch_size, -1)
        if value_tokens.shape[0] == 1 and batch_size > 1:
            value_tokens = value_tokens.expand(batch_size, -1)
        if key_embedding.shape[0] == 1 and batch_size > 1:
            key_embedding = key_embedding.expand(batch_size, -1)
        
        # If text lists have length 1 but we need batch_size, replicate them
        if len(key_text) == 1 and batch_size > 1:
            key_text = key_text * batch_size
        if len(value_text) == 1 and batch_size > 1:
            value_text = value_text * batch_size
            
        device_selected_step = QKVStep(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text
        )
        
        # Store query text, tokens and embedding with the selected step for later display
        device_selected_step.query_text = query_text
        device_selected_step.query_tokens = query_tokens
        device_selected_step.query_embedding = query_embeddings
        
        # Store the softmax probabilities for policy gradient
        # We'll compute log probabilities from the softmax distribution
        device_selected_step.similarity_scores = similarity_scores
        device_selected_step.selected_idx = selected_idx
        
        # IMPORTANT: Store the current available key embeddings at this step
        # This fixes the KL divergence dimension mismatch issue
        # Use setattr to avoid linter issues with dynamic attributes
        setattr(device_selected_step, 'available_key_embeddings', key_embeddings)
        
        # Add selected step to the list
        selected_steps.append(device_selected_step)
        
        # Remove the selected step from available steps
        available_qkv_steps.pop(selected_idx)
        
        # Update context for next iteration
        # For vector queries, we don't add query tokens to context
        # Only add key and value tokens
        key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        current_context = torch.cat([
            current_context,
            key_prefix_tokens, 
            device_selected_step.key_tokens,
            value_prefix_tokens,
            device_selected_step.value_tokens
        ], dim=1)
        
        # Update context text
        context_text = tokenizer.batch_decode(current_context, skip_special_tokens=True)
    
    if verbose:
        # Print the full context at the end of the trajectory
        full_context = tokenizer.batch_decode(current_context)[0]
        print(f"\n=== Complete Context from Trajectory ===")
        print(full_context)
        print("\n=== Trajectory Complete ===\n")
    
    # Create trajectory object with the pre-saved key embeddings
    trajectory = Trajectory(
        qkv_steps=selected_steps,
        all_key_embeddings=trajectory_key_embeddings
    )
    
    return trajectory


def parse_args():
    """Parse command-line arguments."""
    from src.config import TRAINING_BATCH_SIZE
    
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
    parser.add_argument("--no-grpo-batching", dest="grpo_batching", action="store_false",
                        help="Disable GRPO-style batching")
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
    if not trajectory.qkv_steps or len(trajectory.qkv_steps) < 2:
        return 0.5  # Neutral if not enough steps
    
    # Get the sequence of selected indices
    selected_indices = []
    for step in trajectory.qkv_steps:
        if hasattr(step, 'selected_idx') and step.selected_idx is not None:
            selected_indices.append(step.selected_idx)
    
    if len(selected_indices) < 2:
        return 0.5  # Need at least 2 selections to measure consistency
    
    # Convert to list if needed and ensure we have integers
    if isinstance(selected_indices[0], torch.Tensor):
        selected_indices = [idx.item() if hasattr(idx, 'item') else int(idx) for idx in selected_indices]
    
    # Since keys are removed after selection, we need to reconstruct the original indices
    # If we selected indices [2, 1, 0] from pools of size [5, 4, 3], the original indices were [2, 2, 2]
    # But we want to track the relative order within the original set
    
    # For edit distance, we'll use the indices as selected (accounting for pool shrinkage)
    # and compare against the sequential order [0, 1, 2, ..., n-1]
    n = len(selected_indices)
    perfect_sequence = list(range(n))
    
    # Compute edit distance (Levenshtein distance)
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
    
    # Compute edit distance from perfect sequence
    distance = edit_distance(selected_indices, perfect_sequence)
    
    # Normalize: maximum possible edit distance is min(n, max_val) where max_val is the largest index
    # For our case, worst case is when sequence is completely reversed
    max_distance = n  # At most n replacements needed
    
    # Convert to consistency score: 1.0 = perfect (distance 0), 0.0 = worst (max distance)
    if max_distance == 0:
        return 1.0  # Edge case: single element
    
    consistency_score = 1.0 - (distance / max_distance)
    return max(0.0, min(1.0, consistency_score))  # Clamp to [0, 1]


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
    
    # Register embedding hooks
    embeddings_dict, hook_remover = register_embedding_hook(adapter_model)
    
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
                embedding_fn=None  # embeddings computed later
            )
            # Repeat each item batch_size times for GRPO-style batching
            kv_pair_generator = repeat_n_times(args.batch_size, base_iterator)
        else:
            # Standard approach: different items in each batch position
            kv_pair_generator = iter_key_value_pairs_unified_with_tokenizer(
                dataset_name=args.dataset,
                batch_size=args.batch_size, 
                tokenizer=tokenizer,
                embedding_fn=None
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
            
            # Generate a trajectory
            trajectory = generate_trajectory(
                initial_tokens,
                adapter_model,
                ref_model,
                tokenizer,
                embeddings_dict,
                hook_remover,
                available_qkv_steps,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            
            # Compute trajectory rewards and log probabilities using ref_model (for actual rewards)
            _, adapter_log_probs_batch, ref_log_probs_batch = compute_trajectory_rewards(
                trajectory, 
                adapter_model, 
                ref_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=args.verbose
            )
            
            # Also get old model log probabilities for comparison plotting
            _, _, old_log_probs_batch = compute_trajectory_rewards(
                trajectory, 
                adapter_model, 
                old_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=False  # Don't print twice
            )
            
            # Compute KL divergence from reference model (pi_ref) for monitoring
            # This is different from the KL in the loss which is from pi_old
            kl_from_ref_value = 0.0
            if trajectory.qkv_steps:
                # Compute KL between adapter and ref model for each step
                kl_values = []
                for step in trajectory.qkv_steps:
                    if hasattr(step, 'similarity_scores') and step.similarity_scores is not None:
                        # Get log probs from adapter and ref
                        from src.config import TEMPERATURE
                        import torch.nn.functional as F
                        
                        # For simplicity, use the stored similarity scores as proxy
                        # In a full implementation, we'd recompute with ref model
                        log_probs_adapter = F.log_softmax(step.similarity_scores / TEMPERATURE, dim=-1)
                        # Approximate ref model probs as uniform (or could recompute)
                        num_keys = step.similarity_scores.shape[-1]
                        log_probs_ref = torch.full_like(log_probs_adapter, -torch.log(torch.tensor(num_keys)).item())
                        
                        # KL(adapter || ref)
                        kl_step = F.kl_div(log_probs_ref, log_probs_adapter, reduction='batchmean', log_target=True)
                        kl_values.append(kl_step.item())
                
                if kl_values:
                    kl_from_ref_value = sum(kl_values) / len(kl_values)
            
            kl_from_ref.append(kl_from_ref_value)
            
            # Update reward stats
            if trajectory.avg_reward is not None:
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
                embeddings_dict=embeddings_dict
            )
            
            # Track clipping ratio
            clipping_ratios.append(avg_clipping_ratio)
            
            # Track weight change BEFORE updating old_model
            weight_change = compute_weight_change(adapter_model, old_model)
            weight_changes.append(weight_change)
            
            # NO automatic old_model update after each training step - only at intervals
            # This allows KL divergence to accumulate over multiple episodes for meaningful regularization
            
            # Calculate average reward across the batch
            if trajectory.avg_reward is not None and trajectory.avg_reward.numel() > 0:
                avg_reward = trajectory.avg_reward.mean().item()
            else:
                avg_reward = 0.0
                
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
            
            # Track trajectory-level log probabilities (average across all trajectory steps)
            traj_log_prob = adapter_log_probs_batch.mean().item()
            trajectory_log_probs.append(traj_log_prob)
            
            # Track KL penalty terms (actual penalty added to loss)
            kl_penalty_term = (kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss) * KL_PENALTY_COEFFICIENT
            kl_penalty_terms.append(kl_penalty_term)
            
            # Track reward variance within trajectory
            if trajectory.avg_reward is not None and trajectory.avg_reward.numel() > 1:
                reward_var = trajectory.avg_reward.var().item()
            else:
                reward_var = 0.0
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
                    values=None, 
                    gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA,
                    use_grpo_baseline=USE_GRPO_BASELINE
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
            
            # Periodically verify weight changes (every 5 episodes)
            if (episode + 1) % 5 == 0:
                # Check that adapter model weights are changing
                adapter_weights_changed = False
                for name, param in adapter_model.named_parameters():
                    if 'lora' in name and name in initial_base_weights:
                        if not torch.allclose(initial_base_weights[name], param.data):
                            adapter_weights_changed = True
                            break
                
                if adapter_weights_changed:
                    logging.info("Adapter model weights verification: CHANGED (correct)")
                else:
                    logging.warning("Adapter model weights are NOT changing! This may indicate a training issue.")
            
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
                    # Convert similarity scores to log probabilities
                    from src.config import TEMPERATURE
                    import torch.nn.functional as F
                    log_probs = F.log_softmax(step.similarity_scores / TEMPERATURE, dim=-1)
                    # Get log prob of selected action (average over batch)
                    selected_idx = step.selected_idx if hasattr(step, 'selected_idx') else 0
                    selected_log_prob = log_probs[:, selected_idx].mean().item()
                    step_log_probs_episode.append(selected_log_prob)
            step_log_probs.append(step_log_probs_episode)
            
            # Save and plot metrics more frequently (every 25 episodes) for better monitoring
            if episode > 0 and episode % 25 == 0:
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
        hook_remover()
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