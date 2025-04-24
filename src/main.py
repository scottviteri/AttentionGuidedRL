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
from typing import List, Optional, Dict, Callable, Any
import numpy as np
from copy import deepcopy

from src.config import (
    DEVICE,
    MODEL_NAME,
    TOKENS_PER_QUERY,
    NUM_KV_PAIRS,
    CHECKPOINT_INTERVAL,
    NUM_EPISODES,
    GENERATION_BATCH_SIZE,
    LEARNING_RATE,
    KL_PENALTY_COEFFICIENT,
    QUERY_PREFIX,
    KEY_PREFIX,
    VALUE_PREFIX,
    INITIAL_PROMPT,
    CHECKPOINT_DIR,
    ENABLE_WANDB,
    LOG_INTERVAL,
)
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path
from src.data import iter_key_value_pairs, QKVStep
from src.embeddings import register_embedding_hook, extract_embeddings, compute_similarity, sample_key_value
from src.training import (
    Trajectory,
    generate_query,
    compute_trajectory_rewards,
    update_reward_stats,
    train_step,
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
    Generate a trajectory by selecting query-key-value steps.
    
    Args:
        context_tokens: Initial context tokens
        adapter_model: The model with LoRA adapter
        base_model: The base model without LoRA
        tokenizer: The tokenizer
        embeddings_dict: Dictionary to store embeddings from hooks
        hook_remover: Function to remove hooks
        available_qkv_steps: List of available QKVStep objects
        batch_size: Batch size
        verbose: Flag to enable verbose logging
        
    Returns:
        Trajectory: The generated trajectory
    """
    # Ensure context_tokens is on the correct device
    device = next(adapter_model.parameters()).device
    current_context = context_tokens.to(device)
    context_text = tokenizer.batch_decode(current_context)
    
    if verbose:
        print("\n=== Starting New Trajectory ===")
        print(f"Initial context: {context_text[0][:50]}...")
        print(f"Available query-key-value steps: {len(available_qkv_steps)}")
        print(f"Query token length: {TOKENS_PER_QUERY} tokens")
    
    # Initialize selected steps list
    selected_steps = []
    
    # Loop until we've selected a fixed number of steps
    for step_idx in range(NUM_KV_PAIRS):
        # Generate query
        query_tokens = generate_query(
            adapter_model,
            tokenizer,
            context_text
        )
        
        # Ensure query_tokens is on the same device as context_tokens
        query_tokens = query_tokens.to(device)
        
        # Decode query for logging
        query_text = tokenizer.batch_decode(query_tokens)
        
        # Extract query embeddings
        query_embeddings = extract_embeddings(adapter_model, query_tokens, embeddings_dict)
        
        # Extract key embeddings from available steps using extract_embeddings
        key_embs = []
        for qkv_step in available_qkv_steps:
            # Move key_tokens to the correct device
            key_tokens = qkv_step.key_tokens.to(device)
            key_emb = extract_embeddings(adapter_model, key_tokens, embeddings_dict)
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
        device_selected_step = QKVStep(
            key_tokens=selected_step.key_tokens.to(device),
            value_tokens=selected_step.value_tokens.to(device),
            key_embedding=selected_step.key_embedding.to(device),
            key_text=selected_step.key_text,
            value_text=selected_step.value_text
        )
        
        # Store query text, tokens and embedding with the selected step for later display
        device_selected_step.query_text = query_text
        device_selected_step.query_tokens = query_tokens
        device_selected_step.query_embedding = query_embeddings
        
        # Add selected step to the list
        selected_steps.append(device_selected_step)
        
        # Remove the selected step from available steps
        available_qkv_steps.pop(selected_idx)
        
        # Update context for next iteration
        # Add query, key and value tokens to context - already on the correct device
        # Create prefix tensors with proper batch dimension
        query_prefix_tokens = tokenizer([QUERY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        key_prefix_tokens = tokenizer([KEY_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        value_prefix_tokens = tokenizer([VALUE_PREFIX] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        current_context = torch.cat([
            current_context,
            query_prefix_tokens,
            query_tokens,
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
    
    # Create trajectory object
    trajectory = Trajectory(qkv_steps=selected_steps)
    
    return trajectory


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using Attention-Guided RL")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to train")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose trajectory logging")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for training")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--training-percentile", type=float, default=90.0,
                        help="Percentile threshold for training (e.g., 90.0 means train on top 10%% of datapoints)")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
    
    # Set up logging
    log_dir = setup_logging(args)
    
    # Set up models and tokenizer
    logging.info("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Log the dynamically calculated token counts
    import src.config as config
    logging.info(f"Token count configuration:")
    logging.info(f"  Query prefix tokens: {config.PREFIX_TOKENS_PER_QUERY}")
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
        
        # Store adapter weights from previous episode for verification
        previous_adapter_weights = {}
        for name, param in adapter_model.named_parameters():
            if 'lora' in name:  # Only track LoRA parameters
                previous_adapter_weights[name] = param.data.clone()
        
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
        
        # Set up data loader
        logging.info("Setting up data loader...")
        kv_pair_generator = iter_key_value_pairs(
            batch_size=args.batch_size, 
            embedding_fn=lambda x: extract_embeddings(adapter_model, x, embeddings_dict)
        )
        
        # Training loop
        logging.info("Starting training...")
        episodes_range = range(start_episode, args.episodes)
        progress_bar = tqdm(episodes_range)
        
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
                base_model,
                tokenizer,
                embeddings_dict,
                hook_remover,
                available_qkv_steps,
                batch_size,
                verbose=args.verbose,
            )
            
            # Compute trajectory rewards
            compute_trajectory_rewards(
                trajectory, 
                adapter_model, 
                base_model, 
                initial_tokens,
                tokenizer=tokenizer,
                verbose=args.verbose
            )
            
            # Create a deep copy of the current adapter model for KL divergence computation
            # This is important to ensure the reference model doesn't change during training
            previous_model = create_model_copy(adapter_model)
            
            # Update reward stats
            if trajectory.avg_reward is not None:
                reward_stats = update_reward_stats(reward_stats, trajectory.avg_reward)
                
                if args.verbose:
                    print(f"\nUpdated reward stats:")
                    print(f"  Mean: {reward_stats['mean']:.4f}")
                    print(f"  Std: {reward_stats['std']:.4f}")
                    print(f"  Count: {reward_stats['count']}")
            
            # Log the training percentile setting
            logging.info(f"Training on top {100-args.training_percentile:.1f}% of datapoints (percentile threshold: {args.training_percentile})")
            
            # Perform training step
            total_loss, num_filtered, policy_loss, kl_loss = train_step(
                trajectory,
                adapter_model,
                base_model,
                previous_model,  # Use the deep copy for KL divergence
                optimizer,
                reward_stats,
                KL_PENALTY_COEFFICIENT,
                verbose=args.verbose,
                percentile=args.training_percentile  # Pass the percentile parameter
            )
            
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
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode}/{args.episodes}, "
                f"Loss: {total_loss:.4f}, "
                f"Filtered: {num_filtered}/{trajectory.avg_reward.shape[0]}, "
                f"Reward: {avg_reward:.4f}, "
                f"Threshold: {reward_stats['mean'] + reward_stats['std']:.4f}"
            )
            
            # Periodically verify weight changes (every 5 episodes)
            if (episode + 1) % 5 == 0:
                # Check that base model weights are unchanged
                base_weights_changed = False
                for name, param in base_model.named_parameters():
                    if 'lora' not in name and name in initial_base_weights:
                        if not torch.allclose(initial_base_weights[name], param.data):
                            base_weights_changed = True
                            logging.warning(f"Base model weights changed for parameter {name}")
                            break
                
                if not base_weights_changed:
                    logging.info("Base model weights verification: UNCHANGED (correct)")
                
                # Check that adapter model weights are changing
                adapter_weights_changed = False
                for name, param in adapter_model.named_parameters():
                    if 'lora' in name and name in previous_adapter_weights:
                        if not torch.allclose(previous_adapter_weights[name], param.data):
                            adapter_weights_changed = True
                            break
                
                if adapter_weights_changed:
                    logging.info("Adapter model weights verification: CHANGED (correct)")
                else:
                    logging.warning("Adapter model weights are NOT changing! This may indicate a training issue.")
                
                # Update previous adapter weights for next check
                for name, param in adapter_model.named_parameters():
                    if 'lora' in name:
                        previous_adapter_weights[name] = param.data.clone()
            
            # Save checkpoint if needed
            if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(adapter_model, "latest")
                if args.verbose:
                    print(f"\nCheckpoint saved at episode {episode}")
                
                # Plot metrics periodically
                plot_metrics(log_dir)
            
            # Log statistics
            if episode % args.log_interval == 0:
                log_dict = {
                    "episode": episode,
                    "total_loss": total_loss,
                    "policy_loss": policy_loss,
                    "kl_loss": kl_loss,
                    "kl_penalty_term": kl_loss * KL_PENALTY_COEFFICIENT,
                    "reward": avg_reward,
                    "reward_mean": reward_stats["mean"],
                    "reward_std": reward_stats["std"],
                    "filtered_ratio": num_filtered / trajectory.avg_reward.shape[0] if trajectory.avg_reward.shape[0] > 0 else 0
                }
                
                logging.info(
                    f"Episode {episode}/{args.episodes}, "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Policy Loss: {policy_loss:.4f}, "
                    f"KL Loss: {kl_loss:.4f}, "
                    f"Filtered: {num_filtered}/{trajectory.avg_reward.shape[0]}, "
                    f"Reward: {avg_reward:.4f}, "
                    f"Reward Mean: {reward_stats['mean']:.4f}, "
                    f"Reward Std: {reward_stats['std']:.4f}"
                )
                
                # Log to wandb if enabled
                if ENABLE_WANDB:
                    wandb.log(log_dict)
            
        # Save final checkpoint
        save_checkpoint(adapter_model, "latest")
        
        # Create final plots
        plot_metrics(log_dir)
        
        logging.info("Training complete!")
        
        # Close wandb if enabled
        if ENABLE_WANDB:
            wandb.finish()
    
    finally:
        # Remove hook
        hook_remover()


def plot_metrics(log_dir, step=None):
    """
    Create and save plots of training metrics.
    
    Args:
        log_dir: Directory where logs and plots are saved
        step: Current training step (optional, not used for filename)
    """
    # Create plots directory
    plots_dir = f"{log_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Make sure all tensors are converted to CPU before plotting
    cpu_training_steps = [step.item() if isinstance(step, torch.Tensor) else step for step in training_steps]
    cpu_total_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in total_losses]
    cpu_policy_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in policy_losses]
    cpu_kl_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in kl_losses]
    cpu_avg_rewards = [reward.item() if isinstance(reward, torch.Tensor) else reward for reward in avg_rewards]
    
    plt.figure(figsize=(12, 8))
    
    # Loss plots in the first subplot
    plt.subplot(2, 1, 1)
    plt.plot(cpu_training_steps, cpu_total_losses, label='Total Loss', color='blue')
    plt.plot(cpu_training_steps, cpu_policy_losses, label='Policy Loss', color='green')
    plt.plot(cpu_training_steps, cpu_kl_losses, label='KL Loss', color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Components During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reward plot in the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(cpu_training_steps, cpu_avg_rewards, label='Avg Reward', color='purple')
    plt.xlabel('Training Step')
    plt.ylabel('Average Reward')
    plt.title('Average Reward During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot to a single file that gets overwritten
    plt.savefig(f"{plots_dir}/training_metrics.png", dpi=150)
    plt.close()
    
    # If wandb is enabled, log the plot
    if ENABLE_WANDB:
        wandb.log({
            "training_metrics_plot": wandb.Image(f"{plots_dir}/training_metrics.png")
        })


if __name__ == "__main__":
    main() 