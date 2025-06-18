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
    NUM_KV_PAIRS,
    CHECKPOINT_INTERVAL,
    NUM_EPISODES,
    LEARNING_RATE,
    KL_PENALTY_COEFFICIENT,
    KEY_PREFIX,
    VALUE_PREFIX,
    INITIAL_PROMPT,
    CHECKPOINT_DIR,
    ENABLE_WANDB,
    LOG_INTERVAL,
)
from src.model import setup_model_and_tokenizer, save_checkpoint, load_checkpoint, create_model_copy, get_checkpoint_path
from src.data import iter_key_value_pairs, iter_key_value_pairs_unified, QKVStep
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
        
        # Store the softmax probabilities for policy gradient
        # We'll compute log probabilities from the softmax distribution
        device_selected_step.similarity_scores = similarity_scores
        device_selected_step.selected_idx = selected_idx
        
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
    
    # Log query mode
    logging.info("Query mode: Vector queries")
    
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
        logging.info(f"Setting up data loader for {args.dataset} dataset...")
        
        # Create a baseline model for both key embeddings and reward computation
        # This is a deep copy of the initial adapter model that won't be updated frequently
        baseline_model = create_model_copy(adapter_model)
        
        # Register embedding hook for the baseline model
        baseline_embeddings_dict, baseline_hook_remover = register_embedding_hook(baseline_model)
        
        # Use baseline model for key embeddings to ensure stability
        kv_pair_generator = iter_key_value_pairs_unified(
            dataset_name=args.dataset,
            batch_size=args.batch_size, 
            embedding_fn=lambda x: extract_embeddings(baseline_model, x, baseline_embeddings_dict)
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
            
            # Compute trajectory rewards using baseline model (not base model)
            # This ensures fair comparison since baseline knows how to use vector queries
            compute_trajectory_rewards(
                trajectory, 
                adapter_model, 
                baseline_model, 
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
            
            # Perform training step
            total_loss, num_filtered, policy_loss, kl_loss = train_step(
                trajectory,
                adapter_model,
                baseline_model,  # Now using baseline instead of base
                previous_model,  # Use the deep copy for KL divergence
                optimizer,
                reward_stats,
                KL_PENALTY_COEFFICIENT,
                verbose=args.verbose,
                tokenizer=tokenizer,
                embeddings_dict=embeddings_dict
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
                f"Reward: {avg_reward:.4f}"
            )
            
            # Update baseline model less frequently (every 50 episodes)
            if (episode + 1) % 50 == 0:
                # Update the baseline model to reflect learning progress
                baseline_model = create_model_copy(adapter_model)
                
                # Re-register the embedding hook for the new baseline
                baseline_hook_remover()  # Remove old hook
                baseline_embeddings_dict, baseline_hook_remover = register_embedding_hook(baseline_model)
                
                # Update the embedding function in the generator
                kv_pair_generator = iter_key_value_pairs_unified(
                    dataset_name=args.dataset,
                    batch_size=args.batch_size, 
                    embedding_fn=lambda x: extract_embeddings(baseline_model, x, baseline_embeddings_dict)
                )
                
                logging.info(f"Updated baseline model at episode {episode + 1}")
            
            # Periodically verify weight changes (every 5 episodes)
            if (episode + 1) % 5 == 0:
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
        # Remove hooks
        hook_remover()
        if 'baseline_hook_remover' in locals():
            baseline_hook_remover()


def plot_metrics(log_dir, step=None):
    """
    Create and save detailed plots of training metrics.
    
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
    
    # Calculate KL penalty term for visualization
    cpu_kl_penalty_terms = [kl * KL_PENALTY_COEFFICIENT for kl in cpu_kl_losses]
    
    # Create a figure with 2x2 subplots for more detailed visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss components plot
    ax1.plot(cpu_training_steps, cpu_total_losses, 'b-', label='Total Loss', linewidth=2)
    ax1.plot(cpu_training_steps, cpu_policy_losses, 'g--', label='Policy Loss', linewidth=1.5)
    ax1.plot(cpu_training_steps, cpu_kl_penalty_terms, 'r:', label=f'KL Penalty (β={KL_PENALTY_COEFFICIENT})', linewidth=1.5)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Components During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward plot
    ax2.plot(cpu_training_steps, cpu_avg_rewards, 'purple', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Reward During Training')
    ax2.grid(True, alpha=0.3)
    
    # Add a trend line if we have enough data points
    if len(cpu_training_steps) > 10:
        z = np.polyfit(cpu_training_steps, cpu_avg_rewards, 1)
        p = np.poly1d(z)
        ax2.plot(cpu_training_steps, p(cpu_training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        ax2.legend()
    
    # 3. Loss breakdown pie chart (for the most recent step)
    if len(cpu_policy_losses) > 0 and len(cpu_kl_penalty_terms) > 0:
        latest_policy_loss = abs(cpu_policy_losses[-1])
        latest_kl_penalty = abs(cpu_kl_penalty_terms[-1])
        
        # Only create pie chart if both components are positive
        if latest_policy_loss > 0 or latest_kl_penalty > 0:
            sizes = [latest_policy_loss, latest_kl_penalty]
            labels = ['Policy Loss', 'KL Penalty']
            colors = ['green', 'red']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Loss Breakdown (Latest Step)')
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Loss Breakdown (Latest Step)')
    
    # 4. KL Divergence over time
    ax4.plot(cpu_training_steps, cpu_kl_losses, 'darkred', linewidth=2)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('KL Divergence')
    ax4.set_title('KL Divergence Between Current and Previous Policy')
    ax4.grid(True, alpha=0.3)
    
    # Add horizontal line for typical KL divergence scale
    if len(cpu_kl_losses) > 0:
        mean_kl = np.mean(cpu_kl_losses)
        ax4.axhline(y=mean_kl, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_kl:.4f}')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot to a single file that gets overwritten
    plt.savefig(f"{plots_dir}/training_metrics.png", dpi=150)
    plt.close()
    
    # If we have enough data, create an additional detailed loss breakdown over time
    if len(cpu_training_steps) > 20:
        plt.figure(figsize=(12, 6))
        
        # Stack plot showing loss composition over time
        plt.stackplot(cpu_training_steps, 
                     cpu_policy_losses, 
                     cpu_kl_penalty_terms,
                     labels=['Policy Loss', 'KL Penalty'],
                     colors=['green', 'red'],
                     alpha=0.7)
        
        plt.plot(cpu_training_steps, cpu_total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss Composition Over Time')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{plots_dir}/loss_breakdown.png", dpi=150)
        plt.close()
    
    # If wandb is enabled, log the plots
    if ENABLE_WANDB:
        wandb_images = {
            "training_metrics_plot": wandb.Image(f"{plots_dir}/training_metrics.png")
        }
        if os.path.exists(f"{plots_dir}/loss_breakdown.png"):
            wandb_images["loss_breakdown_plot"] = wandb.Image(f"{plots_dir}/loss_breakdown.png")
        wandb.log(wandb_images)


if __name__ == "__main__":
    main() 