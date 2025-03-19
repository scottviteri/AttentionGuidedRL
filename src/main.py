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
from typing import List, Optional

from src.config import (
    DEVICE,
    MODEL_NAME,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    NUM_EPISODES,
    GENERATION_BATCH_SIZE,
    WARMUP_EPISODES,
    LEARNING_RATE,
    KL_PENALTY_COEFFICIENT,
    QUERY_PREFIX,
    RESPONSE_PREFIX,
)
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy
from src.data import iter_key_value_pairs
from src.embeddings import register_embedding_hook, extract_embeddings, compute_similarity, sample_key_value
from src.training import (
    Trajectory,
    generate_query,
    compute_trajectory_rewards,
    update_reward_stats,
    train_step,
)


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
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {LEARNING_RATE}")
    logging.info(f"  Episodes: {NUM_EPISODES}")
    
    return log_dir


def generate_trajectory(
    context_tokens: torch.Tensor,
    adapter_model,
    base_model,
    tokenizer,
    embeddings_dict,
    hook_remover,
    available_kv_pairs: List,
    batch_size: int,
    trajectory_length: int = 3,
) -> Trajectory:
    """
    Generate a single trajectory by sequentially selecting key-value pairs.
    
    Args:
        context_tokens: Initial context tokens
        adapter_model: The model with LoRA adapter
        base_model: The base model without LoRA
        tokenizer: The tokenizer
        embeddings_dict: Dictionary to store embeddings from hooks
        hook_remover: Function to remove hooks
        available_kv_pairs: List of available key-value pairs
        batch_size: Batch size
        trajectory_length: Number of key-value pairs to select (default: 3)
        
    Returns:
        Trajectory: The generated trajectory
    """
    # Initialize context for each batch item
    current_context = context_tokens
    context_text = tokenizer.batch_decode(current_context)
    
    # Initialize selected pair list
    selected_pairs = []
    
    # Loop until we've selected a fixed number of key-value pairs
    for _ in range(trajectory_length):
        # Generate query
        query_tokens = generate_query(
            adapter_model,
            tokenizer,
            context_text,
            max_length=TOKENS_PER_KEY,
            ensure_exact_length=True,
        )
        
        # Extract query embeddings
        query_embeddings = extract_embeddings(adapter_model, query_tokens, embeddings_dict)
        
        # Get key embeddings from available pairs
        key_embeddings = torch.stack([kv_pair.key_embedding for kv_pair in available_kv_pairs])
        
        # Compute similarity scores
        similarity_scores = compute_similarity(query_embeddings, key_embeddings, adapter_model)
        
        # Sample next key-value pair
        available_indices = list(range(len(available_kv_pairs)))
        sampled_indices, _ = sample_key_value(
            similarity_scores, 
            [available_indices] * batch_size,
            batch_size
        )
        
        # For simplicity, use the first batch item's choice
        selected_idx = sampled_indices[0]
        selected_pair = available_kv_pairs[selected_idx]
        
        # Add selected pair to the list
        selected_pairs.append(selected_pair)
        
        # Remove the selected pair from available pairs
        available_kv_pairs.pop(selected_idx)
        
        # Update context for next iteration
        # Add key and value tokens to context
        key_tokens = selected_pair.key_tokens
        value_tokens = selected_pair.value_tokens
        
        # Decode for logging
        query_text = tokenizer.batch_decode(query_tokens)
        
        # Update context tokens
        current_context = torch.cat([
            current_context, 
            query_tokens,
            value_tokens
        ], dim=1)
        
        # Update context text
        context_text = tokenizer.batch_decode(current_context)
    
    # Create trajectory
    trajectory = Trajectory(kv_pairs=selected_pairs)
    
    # Compute rewards for the trajectory
    compute_trajectory_rewards(trajectory, adapter_model, base_model, context_tokens)
    
    return trajectory


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using Attention-Guided RL")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to train")
    parser.add_argument("--trajectory-length", type=int, default=3, help="Length of each trajectory (# of KV pairs)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_dir = setup_logging(args)
    
    # Set up models and tokenizer
    logging.info("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Register embedding hooks
    embeddings_dict, hook_remover = register_embedding_hook(adapter_model)
    
    # Make sure hook is removed at the end
    try:
        # Create optimizer
        optimizer = optim.Adam(adapter_model.parameters(), lr=LEARNING_RATE)
        
        # Initialize reward stats
        reward_stats = {"mean": 0.0, "std": 1.0, "count": 0}
        
        # Try to load checkpoint if resume is specified
        start_episode = 0
        if args.resume:
            for episode in range(args.episodes, 0, -1):
                if load_checkpoint(adapter_model, episode):
                    start_episode = episode
                    logging.info(f"Resumed from episode {start_episode}")
                    break
        
        # Set up data loader
        logging.info("Setting up data loader...")
        kv_pair_generator = iter_key_value_pairs(
            batch_size=args.batch_size, 
            embedding_fn=lambda x: extract_embeddings(base_model, x, embeddings_dict)
        )
        
        # Training loop
        logging.info("Starting training...")
        episodes_range = range(start_episode, args.episodes)
        progress_bar = tqdm(episodes_range)
        
        for episode in progress_bar:
            # Get a batch of key-value pairs
            available_kv_pairs = [next(kv_pair_generator) for _ in range(20)]  # Get a pool of KV pairs
            
            # Create initial context (empty for now)
            batch_size = args.batch_size
            context_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=DEVICE)
            
            # Generate a trajectory
            trajectory = generate_trajectory(
                context_tokens,
                adapter_model,
                base_model,
                tokenizer,
                embeddings_dict,
                hook_remover,
                available_kv_pairs,
                batch_size,
                trajectory_length=args.trajectory_length,
            )
            
            # Create a copy of the current model for KL divergence computation
            previous_model = create_model_copy(adapter_model)
            
            # Update reward stats
            if trajectory.avg_reward is not None:
                reward_stats = update_reward_stats(reward_stats, trajectory.avg_reward)
            
            # Perform training step
            loss, num_filtered = train_step(
                [trajectory],
                adapter_model,
                base_model,
                previous_model,
                optimizer,
                reward_stats,
                KL_PENALTY_COEFFICIENT,
            )
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode}/{args.episodes}, "
                f"Loss: {loss:.4f}, "
                f"Reward: {trajectory.avg_reward[0].item():.4f}, "
                f"Reward Threshold: {reward_stats['mean'] + reward_stats['std']:.4f}"
            )
            
            # Log statistics
            if episode % args.log_interval == 0:
                logging.info(
                    f"Episode {episode}/{args.episodes}, "
                    f"Loss: {loss:.4f}, "
                    f"Reward: {trajectory.avg_reward[0].item():.4f}, "
                    f"Reward Mean: {reward_stats['mean']:.4f}, "
                    f"Reward Std: {reward_stats['std']:.4f}, "
                    f"Num Filtered: {num_filtered}"
                )
            
            # Save checkpoint
            if episode % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(adapter_model, episode)
        
        # Save final checkpoint
        save_checkpoint(adapter_model, args.episodes)
        logging.info("Training complete!")
        
    finally:
        # Remove hook
        hook_remover()


if __name__ == "__main__":
    main() 