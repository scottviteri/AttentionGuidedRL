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
from typing import Dict, List, Any, Tuple, Optional

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
)


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
    
    # Print debug info about the selected question
    print(f"Selected question: {query_text}")
    print(f"Answer: {selected_step.value_text[0]}")
    
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
        trajectory_length: Number of steps in the trajectory
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
    
    # Display the object being used (for debugging)
    print(f"\nBuilding trajectory for object: {object_data['object']}")
    
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
    print("\nFull Q&A history:")
    for entry in history:
        print(f"  {entry}")
    print()
    
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
    num_episodes: int = 1000,
    trajectory_length: int = 5,
    batch_size: int = 1,
    learning_rate: float = LEARNING_RATE,
    kl_penalty_coef: float = KL_PENALTY_COEFFICIENT,
    log_interval: int = LOG_INTERVAL,
    checkpoint_interval: int = 10,
    resume: bool = False,
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
        resume: Whether to resume from a checkpoint

    Returns:
        Dict[str, float]: Training statistics
    """
    # Load the dataset
    dataset = load_twenty_questions_dataset(dataset_path)
    questions = dataset["questions"]
    data = dataset["data"]
    
    # Setup optimization
    optimizer = torch.optim.AdamW(adapter_model.parameters(), lr=learning_rate)
    
    # Initialize reward statistics
    reward_stats = {"count": 0, "mean": 0.0, "std": 1.0, "max": -float("inf"), "min": float("inf")}
    
    # Resume from checkpoint if requested
    start_episode = 0
    if resume:
        checkpoint_path = get_checkpoint_path("latest")
        if os.path.exists(checkpoint_path):
            success = load_checkpoint(adapter_model, "latest")
            if success:
                print(f"Resumed from checkpoint {checkpoint_path}")
                start_episode = int(checkpoint_path.split("_")[-1].split(".")[0])
                if "episode" in checkpoint_path:
                    start_episode = int(checkpoint_path.split("episode_")[-1].split(".")[0])
                print(f"Starting from episode {start_episode}")
    
    # Main training loop
    total_steps = 0
    total_reward = 0.0
    training_stats = {
        "rewards": [],
        "losses": [],
        "kl_divs": [],
        "policy_losses": [],
        "entropy": []
    }
    
    print(f"Starting training for {num_episodes} episodes")
    for episode in range(start_episode, start_episode + num_episodes):
        # Create a previous model for computing KL divergence
        previous_model = create_model_copy(adapter_model)
        
        # Sample a random object from the dataset
        obj_idx = random.randint(0, len(data) - 1)
        object_data = data[obj_idx]
        
        # Build a trajectory for this object
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
            optimizer, reward_stats, kl_penalty_coef, episode % log_interval == 0
        )
        
        # Update statistics
        total_steps += steps
        total_reward += trajectory.avg_reward.mean().item()
        training_stats["rewards"].append(trajectory.avg_reward.mean().item())
        training_stats["losses"].append(loss)
        training_stats["kl_divs"].append(kl_div)
        training_stats["policy_losses"].append(policy_loss)
        
        # Log progress
        if episode % log_interval == 0:
            avg_reward = total_reward / log_interval if episode > 0 else total_reward
            print(f"Episode {episode}/{start_episode + num_episodes - 1} | "
                  f"Loss: {loss:.4f} | "
                  f"Policy Loss: {policy_loss:.4f} | "
                  f"KL Div: {kl_div:.4f} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Steps: {steps}")
            
            # Also log the current object and a sample of questions from this episode
            object_name = object_data["object"]
            print(f"Current object: {object_name}")
            print("Sample questions and answers:")
            for i, step in enumerate(trajectory.qkv_steps[:3]):  # Show first 3 questions
                print(f"  Q{i+1}: {step.query_text[0]}")
                print(f"     → {step.key_text[0]} {step.value_text[0]}")
            print()
            
            total_reward = 0.0
        
        # Save checkpoint
        if episode > 0 and episode % checkpoint_interval == 0:
            save_checkpoint(adapter_model, episode)
            save_checkpoint(adapter_model, "latest")
    
    # Save final checkpoint
    save_checkpoint(adapter_model, "latest")
    
    return training_stats


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train with twenty questions dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--trajectory-length", type=int, default=5, help="Trajectory length")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--kl-penalty", type=float, default=KL_PENALTY_COEFFICIENT, help="KL penalty coefficient")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL, help="Log interval")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
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
            resume=args.resume,
        )
    finally:
        # Make sure to remove the hook
        hook_remover()


if __name__ == "__main__":
    main() 