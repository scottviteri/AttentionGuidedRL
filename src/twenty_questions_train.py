"""
Training script for the twenty questions dataset.

This script adapts the reinforcement learning training procedure
to use the twenty questions dataset instead of Wikipedia articles.
"""

import os
import argparse
import time
import torch
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Set

from transformers import AutoTokenizer

from src.config import (
    DEVICE,
    MODEL_NAME,
    TOKENIZER_NAME,
    CHECKPOINT_DIR,
    LOG_INTERVAL,
    KL_PENALTY_COEFFICIENT,
    LEARNING_RATE,
    GRADIENT_CLIP_NORM,
    ENABLE_WANDB,
)
from src.data import QKVStep
from src.twenty_questions_data import (
    load_twenty_questions_dataset,
    create_twenty_questions_context,
    get_twenty_questions_pool,
    iter_twenty_questions_batches,
)
from src.model import (
    setup_model_and_tokenizer,
    get_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    create_model_copy,
)
from src.embeddings import (
    register_embedding_hook,
    extract_embeddings,
    compute_similarity,
)
from src.training import (
    Trajectory,
    generate_query,
    compute_trajectory_rewards,
    train_step,
    update_reward_stats,
    compute_policy_loss,
)

# Import wandb for logging if enabled
if ENABLE_WANDB:
    import wandb

# Import matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def plot_metrics(training_stats, log_dir, step=None, training_percentile=90.0):
    """
    Generate individual plots for each training metric.
    
    Args:
        training_stats: Dictionary containing training metrics
        log_dir: Directory to save the plots
        step: Optional current step to display in title
        training_percentile: Percentile threshold for training
    """
    # Create plots directory
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if we have enough data to plot
    if len(training_stats["episodes"]) < 2:
        logging.info("Not enough data to generate plots yet")
        return
    
    # Add a step text for titles if specified
    step_text = f" (Episode {step})" if step is not None else ""
    
    # Calculate expected fraction of datapoints that meet threshold (for reference line)
    expected_fraction = (100 - training_percentile) / 100.0
    
    # Define metrics to plot with their properties
    metrics_to_plot = [
        {
            "name": "loss",
            "data": training_stats["losses"],
            "title": f"Total Loss{step_text}",
            "ylabel": "Loss",
            "color": "blue",
            "filename": "loss.png"
        },
        {
            "name": "policy_loss",
            "data": training_stats["policy_losses"],
            "title": f"Policy Loss{step_text}",
            "ylabel": "Policy Loss",
            "color": "green",
            "filename": "policy_loss.png"
        },
        {
            "name": "kl_div",
            "data": training_stats["kl_divs"],
            "title": f"KL Divergence{step_text}",
            "ylabel": "KL Divergence",
            "color": "red",
            "filename": "kl_div.png"
        },
        {
            "name": "reward",
            "data": training_stats["rewards"],
            "title": f"Average Reward{step_text}",
            "ylabel": "Reward",
            "color": "purple",
            "filename": "reward.png"
        },
        {
            "name": "data_coverage",
            "data": training_stats["data_coverage"],
            "title": f"Fraction of Dataset Meeting Threshold{step_text}",
            "ylabel": "Fraction of Total Dataset",
            "color": "orange",
            "filename": "data_coverage.png"
        }
    ]
    
    # Add gradient norm if available
    if "grad_norms" in training_stats and training_stats["grad_norms"]:
        metrics_to_plot.append({
            "name": "grad_norm",
            "data": training_stats["grad_norms"],
            "title": f"Gradient Norm{step_text}",
            "ylabel": "Norm",
            "color": "brown",
            "filename": "grad_norm.png"
        })
    
    # Generate individual plots for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(training_stats["episodes"], metric["data"], color=metric["color"])
        plt.xlabel("Episode")
        plt.ylabel(metric["ylabel"])
        plt.title(metric["title"])
        plt.grid(True, alpha=0.3)
        
        # Add reference line for data coverage plots
        if metric["name"] == "data_coverage":
            plt.axhline(y=expected_fraction, color='r', linestyle='--', 
                      label=f'Expected ({expected_fraction:.1%} with {100-training_percentile:.0f}% selection)')
            plt.legend()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, metric["filename"])
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        logging.debug(f"Saved {metric['name']} plot to {plot_path}")
    
    # Also create a combined plot with all metrics for overview
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
    fig.suptitle(f'Twenty Questions Training Metrics{step_text}', fontsize=16)
    
    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(training_stats["episodes"], metric["data"], color=metric["color"])
        axes[i].set_xlabel("Episode")
        axes[i].set_ylabel(metric["ylabel"])
        axes[i].set_title(metric["title"])
        axes[i].grid(True, alpha=0.3)
        
        # Set y-axis limits and add reference line for data coverage
        if metric["name"] == "data_coverage":
            axes[i].set_ylim(0, 1.0)
            axes[i].axhline(y=expected_fraction, color='r', linestyle='--',
                           label=f'Expected ({expected_fraction:.1%} with {100-training_percentile:.0f}% selection)')
            axes[i].legend()
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the combined plot
    combined_plot_path = os.path.join(plots_dir, "all_metrics.png")
    plt.savefig(combined_plot_path, dpi=100)
    plt.close()
    
    logging.info(f"Training plots saved to {plots_dir}")
    
    # If wandb is enabled, log the plots
    if ENABLE_WANDB:
        for metric in metrics_to_plot:
            plot_path = os.path.join(plots_dir, metric["filename"])
            wandb.log({
                f"{metric['name']}_plot": wandb.Image(plot_path)
            })
        wandb.log({
            "all_metrics_plot": wandb.Image(combined_plot_path)
        })


def setup_logging(run_name=None):
    """
    Set up logging for the training run.
    
    Args:
        run_name: Optional name for this training run
    
    Returns:
        str: Path to the log directory
    """
    # Create log directory with timestamp
    log_dir = os.path.join("logs", 
                          run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
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
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log basic info
    logging.info(f"Starting Twenty Questions training run")
    logging.info(f"Log directory: {log_dir}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Device: {DEVICE}")
    
    return log_dir


def sample_twenty_questions_step(
    model: torch.nn.Module,
    tokenizer: Any,
    embeddings_dict: Dict,
    context: torch.Tensor,
    context_text: str,
    available_pool: List[QKVStep],
    temperature: float = 1.0,
) -> QKVStep:
    """
    Sample a question-answer step from the available pool based on the context embedding similarity.

    Args:
        model: The language model
        tokenizer: The tokenizer
        embeddings_dict: Dictionary to store embeddings from hook
        context: Current context tokens [batch_size, context_length]
        context_text: Current context as text
        available_pool: Pool of available QKVStep objects
        temperature: Temperature for softmax sampling

    Returns:
        QKVStep: The sampled step
    """
    # Instead of generating a query, we'll directly compute the similarity between
    # the context embedding and all available questions
    
    # Extract context embedding
    context_embedding = extract_embeddings(
        model, 
        context.to(DEVICE),
        embeddings_dict
    ).detach()
    
    # Compute similarity with all keys in the pool
    similarities = []
    for step in available_pool:
        key_embedding = step.key_embedding
        
        # Calculate cosine similarity between context and key
        similarity = torch.nn.functional.cosine_similarity(
            context_embedding, key_embedding, dim=-1
        )
        
        similarities.append(similarity.item())
    
    # Apply temperature and convert to probabilities
    similarities = torch.tensor(similarities) / temperature
    probabilities = torch.nn.functional.softmax(similarities, dim=0)
    
    # Sample an index based on the probabilities
    index = torch.multinomial(probabilities, 1).item()
    
    # Get the selected step
    selected_step = available_pool[index]
    
    # The query text is just the selected question
    query_text = selected_step.key_text[0]
    
    # Create query tokens
    query_tokens = tokenizer(
        [query_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).input_ids.to(DEVICE)
    
    # Get query embedding
    query_embedding = extract_embeddings(
        model, 
        query_tokens,
        embeddings_dict
    ).detach()
    
    # Add query information to the step
    selected_step.query_text = [query_text]
    selected_step.query_tokens = query_tokens
    selected_step.query_embedding = query_embedding
    
    # Remove the selected step from the pool to avoid repetition
    available_pool.pop(index)
    
    # Log the selected question
    logging.debug(f"Selected question: {query_text}")
    logging.debug(f"Answer: {selected_step.value_text[0]}")
    
    return selected_step


def build_twenty_questions_trajectory(
    model: torch.nn.Module,
    tokenizer: Any,
    embeddings_dict: Dict,
    object_data: Dict,
    questions: List[str],
    trajectory_length: int = 5,
    temperature: float = 1.0,
) -> Tuple[Trajectory, torch.Tensor]:
    """
    Build a trajectory for a single object from the twenty questions dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        embeddings_dict: Dictionary to store embeddings from hook
        object_data: The object data containing name and answers
        questions: List of all available questions
        trajectory_length: Number of steps in each trajectory
        temperature: Temperature for softmax sampling

    Returns:
        Tuple[Trajectory, torch.Tensor]: The built trajectory and initial context
    """
    # Create initial context for the 20 questions game (without revealing the object)
    context_tokens = create_twenty_questions_context(tokenizer)
    
    # Use a more detailed context description to guide the model
    context_text = (
        "I am thinking of an object. You are playing 20 questions to guess what it is. "
        "You need to ask yes/no questions that will help you identify the object efficiently. "
        "Try to ask questions that will divide the space of possible objects in half with each question."
    )
    
    # Display the object being used
    logging.info(f"\nBuilding trajectory for object: {object_data['object']}")
    
    # Define an embedding extraction function
    extract_fn = lambda x: extract_embeddings(model, x, embeddings_dict)
    
    # Get all question-answer pairs for this object
    available_pool = get_twenty_questions_pool(
        object_data, questions, tokenizer, extract_fn
    )
    
    # Limit trajectory length if fewer questions are available
    trajectory_length = min(trajectory_length, len(available_pool))
    
    # Sample steps to build the trajectory
    qkv_steps = []
    history = []  # Track question-answer history for display only
    
    for step_idx in range(trajectory_length):
        # Break if pool is empty
        if not available_pool:
            break
            
        # Sample a step based on query similarity
        step = sample_twenty_questions_step(
            model, tokenizer, embeddings_dict, context_tokens, context_text,
            available_pool, temperature
        )
        
        qkv_steps.append(step)
        
        # Add to history for tracking (not used for generation)
        history.append(f"Q{step_idx+1}: {step.query_text[0]} A: {step.value_text[0]}")
    
    # Display the full question-answer history
    logging.info("\nFull Q&A history:")
    for entry in history:
        logging.info(f"  {entry}")
    
    # Create the trajectory
    trajectory = Trajectory(qkv_steps=qkv_steps)
    
    return trajectory, context_tokens


def train_twenty_questions(
    adapter_model: torch.nn.Module,
    base_model: torch.nn.Module,
    tokenizer: Any,
    embeddings_dict: Dict,
    hook_remover: Any,
    dataset_path: str = None,
    num_episodes: int = 1000,  # This is now the maximum number of episodes
    trajectory_length: int = None,
    batch_size: int = 1,
    learning_rate: float = LEARNING_RATE,
    kl_penalty_coef: float = KL_PENALTY_COEFFICIENT,
    log_interval: int = LOG_INTERVAL,
    checkpoint_interval: int = 10,
    plot_interval: int = 5,
    resume: bool = False,
    run_name: str = None,
    debug_object_idx: int = None,
    training_percentile: float = 90.0,  # Default to top 10%
) -> Dict[str, float]:
    """
    Train the model using the twenty questions dataset.

    Args:
        adapter_model: The model with LoRA adapter
        base_model: The base model without LoRA
        tokenizer: The tokenizer
        embeddings_dict: Dictionary to store embeddings from hook
        hook_remover: Function to remove the hook 
        dataset_path: Path to the dataset
        num_episodes: Number of episodes to train for
        trajectory_length: Number of steps in each trajectory
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        kl_penalty_coef: Coefficient for KL penalty
        log_interval: Interval for logging statistics
        checkpoint_interval: Interval for saving checkpoints
        plot_interval: Interval for generating plots
        resume: Whether to resume from a checkpoint
        run_name: Optional name for this training run
        debug_object_idx: If set, train repeatedly on this single object index
        training_percentile: Percentile threshold for training (e.g., 90.0 means train on top 10%)

    Returns:
        Dict[str, float]: Training statistics
    """
    # Set up logging
    log_dir = setup_logging(run_name)
    
    # Initialize wandb if enabled
    if ENABLE_WANDB:
        wandb_config = {
            "dataset": dataset_path,
            "learning_rate": learning_rate,
            "episodes": num_episodes,
            "trajectory_length": trajectory_length,
            "batch_size": batch_size,
            "kl_penalty_coef": kl_penalty_coef,
            "debug_object_idx": debug_object_idx,
            "training_percentile": training_percentile,
        }
        wandb.init(
            project="attention-guided-rl-twenty-questions",
            name=run_name,
            config=wandb_config
        )
        logging.info("Weights & Biases logging enabled")
    
    # Log the training percentile setting
    logging.info(f"Training on top {100-training_percentile:.1f}% of datapoints (percentile threshold: {training_percentile})")
    
    # Load the dataset
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_twenty_questions_dataset(dataset_path)
    questions = dataset["questions"]
    data = dataset["data"]
    logging.info(f"Loaded {len(data)} objects and {len(questions)} questions")
    
    # Handle debug mode with a single object
    if debug_object_idx is not None:
        if 0 <= debug_object_idx < len(data):
            logging.info(f"DEBUG MODE: Training only on object at index {debug_object_idx}: '{data[debug_object_idx]['object']}'")
        else:
            logging.error(f"Invalid debug_object_idx {debug_object_idx}. Must be between 0 and {len(data)-1}")
            return {}
    
    # If trajectory_length is None, use all questions
    if trajectory_length is None:
        trajectory_length = len(questions)
        logging.info(f"Using all {trajectory_length} questions for each trajectory")
    
    # Determine the actual number of episodes (minimum of specified episodes and dataset size)
    # In debug mode, we can run all episodes on the same object
    total_episodes = num_episodes if debug_object_idx is not None else min(num_episodes, len(data))
    if debug_object_idx is None:
        logging.info(f"Will train on {total_episodes} objects (limited by dataset size)")
    else:
        logging.info(f"Will train for {total_episodes} episodes on the same object")
    
    # Setup optimization
    optimizer = torch.optim.AdamW(adapter_model.parameters(), lr=learning_rate)
    logging.info(f"Using AdamW optimizer with learning rate {learning_rate}")
    
    # Initialize reward statistics
    reward_stats = {"count": 0, "mean": 0.0, "std": 1.0, "max": -float("inf"), "min": float("inf")}
    
    # Resume from checkpoint if requested
    start_episode = 0
    start_object_idx = 0
    
    if resume:
        checkpoint_path = get_checkpoint_path("latest")
        if os.path.exists(checkpoint_path):
            success = load_checkpoint(adapter_model, "latest")
            if success:
                logging.info(f"Resumed from checkpoint {checkpoint_path}")
                checkpoint_info = checkpoint_path.split("_")
                if "episode" in checkpoint_path:
                    for part in checkpoint_info:
                        if part.isdigit():
                            start_episode = int(part)
                    
                    # The next object index is the same as the episode number
                    # unless we're in debug mode, then always use the debug object
                    start_object_idx = debug_object_idx if debug_object_idx is not None else start_episode
                    
                    logging.info(f"Starting from episode {start_episode}, object index {start_object_idx}")
                else:
                    start_episode = 0
                    start_object_idx = debug_object_idx if debug_object_idx is not None else 0
                    logging.info("Could not determine episode from checkpoint, starting from the beginning")
    
    # Check if we have any episodes left to train
    if debug_object_idx is None and start_object_idx >= len(data):
        logging.info("All objects in the dataset have been processed. Training complete.")
        return {}
    
    # Main training loop
    total_steps = 0
    total_reward = 0.0
    
    # Initialize training statistics with lists for tracking metrics
    training_stats = {
        "episodes": [],
        "rewards": [],
        "losses": [],
        "kl_divs": [],
        "policy_losses": [],
        "data_coverage": [],
        "grad_norms": []  # Track gradient norms
    }
    
    # Set to track which datapoints met the threshold criteria for training
    objects_trained_on = set()
    
    logging.info(f"Starting {'debug' if debug_object_idx is not None else 'deterministic'} training for up to {total_episodes} episodes")
    episode = start_episode
    
    # Process each object in the dataset in order, starting from the resume point
    # In debug mode, we repeatedly use the same object index instead of iterating through the dataset
    for e in range(episode, start_episode + total_episodes):
        if debug_object_idx is not None:
            # In debug mode, always use the specified object
            obj_idx = debug_object_idx
        else:
            # Normal mode: iterate through dataset objects
            obj_idx = e
            # Check if we've reached the end of the dataset
            if obj_idx >= len(data):
                logging.info(f"Reached the end of the dataset ({len(data)} objects). Stopping training.")
                break
            
        # Create a previous model for computing KL divergence
        previous_model = create_model_copy(adapter_model)
        
        # Get the object data
        object_data = data[obj_idx]
        
        # Build a trajectory for this object
        logging.info(f"Episode {e}, processing object {obj_idx+1}/{len(data)}: {object_data['object']}")
        trajectory, context_tokens = build_twenty_questions_trajectory(
            adapter_model, tokenizer, embeddings_dict,
            object_data, questions, trajectory_length
        )
        
        # Compute rewards for the trajectory
        rewards = compute_trajectory_rewards(
            trajectory, adapter_model, base_model, context_tokens, tokenizer
        )
        trajectory.rewards = rewards
        trajectory.avg_reward = rewards.mean(dim=1)
        
        # Update reward statistics
        reward_stats = update_reward_stats(reward_stats, rewards)
        
        # Perform a training step
        loss, steps, policy_loss, kl_div = train_step(
            trajectory, adapter_model, base_model, previous_model,
            optimizer, reward_stats, kl_penalty_coef, e % log_interval == 0,
            percentile=training_percentile  # Pass the percentile parameter
        )
        
        # If steps > 0, it means the datapoint met the threshold criteria and was used for training
        if steps > 0:
            objects_trained_on.add(obj_idx)
        
        # Calculate the fraction of datapoints that have met the threshold for training
        # In debug mode, we're only considering one object
        if debug_object_idx is not None:
            trained_coverage = 1.0 if len(objects_trained_on) > 0 else 0.0
        else:
            trained_coverage = len(objects_trained_on) / len(data)
        
        # Calculate gradient norm
        # Zero gradients first to clear any existing gradients
        optimizer.zero_grad()
        
        # Recompute the loss and backward to get fresh gradients
        # We'll use the same loss calculation as in train_step
        filtered_trajectory = trajectory
        temp_total_loss, _, _ = compute_policy_loss(
            filtered_trajectory,
            adapter_model,
            previous_model,
            kl_penalty_coef,
            verbose=False
        )
        temp_total_loss.backward()
        
        # Compute gradient norm without clipping (to get true norm)
        grad_norm = 0.0
        for param in adapter_model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Clear gradients again
        optimizer.zero_grad()
        
        # Update statistics
        total_steps += steps
        avg_reward = trajectory.avg_reward.mean().item()
        total_reward += avg_reward
        
        # Append metrics to tracking lists - ensure we have Python floats, not tensors
        training_stats["episodes"].append(e)
        training_stats["rewards"].append(float(avg_reward))
        training_stats["losses"].append(float(loss))
        training_stats["policy_losses"].append(float(policy_loss))
        training_stats["kl_divs"].append(float(kl_div) if isinstance(kl_div, float) else float(kl_div.item()))
        training_stats["data_coverage"].append(float(len(objects_trained_on) / len(data)))  # Now using trained_coverage instead of data_coverage
        print(f"DEBUG: Adding {float(len(objects_trained_on) / len(data)):.4f} to data_coverage, objects_trained_on={len(objects_trained_on)}")
        training_stats["grad_norms"].append(float(grad_norm))
        
        # Log progress
        if e % log_interval == 0:
            avg_reward_over_interval = total_reward / log_interval if e > 0 else total_reward
            # Calculate the data processing coverage (different from training coverage)
            if debug_object_idx is not None:
                data_processed_coverage = 1.0 / len(data)  # Only one object out of the dataset
            else:
                data_processed_coverage = (obj_idx + 1) / len(data)
            
            log_message = (
                f"Episode {e}/{start_episode + total_episodes - 1} | "
                f"Object {obj_idx+1}/{len(data)} | "
                f"Loss: {loss:.4f} | "
                f"Policy Loss: {policy_loss:.4f} | "
                f"KL Div: {kl_div:.4f} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"Reward Stats: μ={reward_stats['mean']:.4f}, σ={reward_stats['std']:.4f}"
            )
            
            # Only add data coverage stats in normal mode
            if debug_object_idx is None:
                log_message += f" | Data Processed: {data_processed_coverage:.2%} | Data Trained: {trained_coverage:.2%} ({len(objects_trained_on)}/{len(data)})"
            
            log_message += f" | Steps: {steps}"
            logging.info(log_message)
            
            # Log the current object and a sample of questions from this episode
            object_name = object_data["object"]
            logging.info(f"Current object: {object_name}")
            logging.info("Sample questions and answers:")
            for i, step in enumerate(trajectory.qkv_steps[:3]):  # Show first 3 questions
                logging.info(f"  Q{i+1}: {step.query_text[0]}")
                logging.info(f"     → {step.key_text[0]} {step.value_text[0]}")
            
            # Log to wandb if enabled
            if ENABLE_WANDB:
                wandb_log = {
                    "episode": e,
                    "object_idx": obj_idx,
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "kl_div": kl_div,
                    "avg_reward": avg_reward,
                    "grad_norm": grad_norm,
                    "reward_mean": reward_stats["mean"],
                    "reward_std": reward_stats["std"],
                    "steps": steps,
                }
                
                # Only add coverage metrics in normal mode
                if debug_object_idx is None:
                    wandb_log.update({
                        "data_processed_coverage": data_processed_coverage,
                        "data_trained_coverage": trained_coverage,
                    })
                    
                wandb.log(wandb_log)
            
            total_reward = 0.0
        
        # Generate plots periodically
        if e > 0 and e % plot_interval == 0:
            plot_metrics(training_stats, log_dir, e, training_percentile)
        
        # Save checkpoint
        if e > 0 and e % checkpoint_interval == 0:
            save_checkpoint(adapter_model, e)
            save_checkpoint(adapter_model, "latest")
            logging.info(f"Checkpoint saved at episode {e}")
        
        # In normal mode, increment episode matches obj_idx
        # In debug mode, we manually update episode
        episode = e + 1
    
    # Save final checkpoint
    save_checkpoint(adapter_model, "latest")
    logging.info("Training complete. Final checkpoint saved.")
    
    # Generate final plots
    plot_metrics(training_stats, log_dir, training_percentile=training_percentile)
    
    # Log final data coverage
    if debug_object_idx is None:
        final_processed_coverage = float(min(obj_idx + 1, len(data))) / len(data)
        final_trained_coverage = len(objects_trained_on) / len(data)
        logging.info(f"Final data processed: {final_processed_coverage:.2%} ({min(obj_idx + 1, len(data))}/{len(data)})")
        logging.info(f"Final data trained on: {final_trained_coverage:.2%} ({len(objects_trained_on)}/{len(data)})")
    else:
        # In debug mode, we only care about the single object
        debug_obj_trained = debug_object_idx in objects_trained_on
        logging.info(f"Debug object trained on: {debug_obj_trained}")
    
    # Finish wandb run if enabled
    if ENABLE_WANDB:
        wandb.finish()
    
    return training_stats


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train with twenty questions dataset")
    parser.add_argument("--dataset", type=str, default="data/20q_dataset.json", help="Path to dataset file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--trajectory-length", type=int, default=5, help="Trajectory length")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--kl-penalty", type=float, default=KL_PENALTY_COEFFICIENT, help="KL penalty coefficient")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL, help="Log interval")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--plot-interval", type=int, default=5, help="Plot generation interval")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug-object-idx", type=int, default=None, 
                        help="If set, train repeatedly on this single object index for debugging")
    parser.add_argument("--training-percentile", type=float, default=90.0,
                        help="Percentile threshold for training (e.g., 90.0 means train on top 10%% of datapoints)")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up models and tokenizer
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Register embedding hooks
    embeddings_dict, hook_remover = register_embedding_hook(adapter_model)
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    try:
        # Train the model
        train_twenty_questions(
            adapter_model=adapter_model,
            base_model=base_model,
            tokenizer=tokenizer,
            embeddings_dict=embeddings_dict,
            hook_remover=hook_remover,
            dataset_path=args.dataset,
            num_episodes=args.episodes,
            trajectory_length=args.trajectory_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            kl_penalty_coef=args.kl_penalty,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            plot_interval=args.plot_interval,
            resume=args.resume,
            run_name=args.run_name,
            debug_object_idx=args.debug_object_idx,
            training_percentile=args.training_percentile,
        )
    finally:
        # Make sure to remove the hook
        hook_remover()


if __name__ == "__main__":
    main() 