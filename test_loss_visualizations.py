#!/usr/bin/env python3
"""
A test script to demonstrate the loss visualization functionality.
This will run a small number of training steps and generate visualizations.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch.optim as optim

from src.model import setup_model_and_tokenizer, create_model_copy
from src.data import iter_key_value_pairs
from src.embeddings import register_embedding_hook, extract_embeddings
from src.training import compute_trajectory_rewards, train_step, Trajectory, update_reward_stats
from src.main import generate_trajectory
from src.config import INITIAL_PROMPT, NUM_KV_PAIRS, KL_PENALTY_COEFFICIENT

def main():
    """Run a simple test to demonstrate the loss visualization functionality."""
    parser = argparse.ArgumentParser(description="Test loss visualizations")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    print("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Create directories for output
    os.makedirs("test_logs", exist_ok=True)
    os.makedirs("test_logs/plots", exist_ok=True)
    
    # Register embedding hooks
    embeddings_dict, hook_remover = register_embedding_hook(adapter_model)
    
    try:
        # Create optimizer
        optimizer = torch.optim.Adam(adapter_model.parameters(), lr=0.001)
        
        # Initialize reward stats
        reward_stats = {"mean": 0.0, "std": 1.0, "count": 10}  # Pretend we've seen 10 episodes already
        
        # Lists to store metrics
        steps = []
        total_losses = []
        policy_losses = []
        kl_losses = []
        rewards = []
        
        # Set up data loader
        print("Setting up data loader...")
        kv_pair_generator = iter_key_value_pairs(
            batch_size=args.batch_size,
            embedding_fn=lambda x: extract_embeddings(adapter_model, x, embeddings_dict)
        )
        
        # Run for specified number of steps
        for i in range(args.steps):
            print(f"\n\n======== STEP {i+1}/{args.steps} ========")
            
            # Get a batch of key-value pairs
            available_qkv_steps = [next(kv_pair_generator) for _ in range(NUM_KV_PAIRS)]
            
            # Create initial context
            device = next(adapter_model.parameters()).device
            initial_tokens = tokenizer(
                [INITIAL_PROMPT] * args.batch_size,
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
                args.batch_size,
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
            previous_model = create_model_copy(adapter_model)
            
            # Update reward stats
            if trajectory.avg_reward is not None:
                reward_stats = update_reward_stats(reward_stats, trajectory.avg_reward)
            
            # Perform training step
            total_loss, num_filtered, policy_loss, kl_loss = train_step(
                trajectory,
                adapter_model,
                base_model,
                previous_model,
                optimizer,
                reward_stats,
                KL_PENALTY_COEFFICIENT,
                verbose=args.verbose
            )
            
            # Calculate average reward
            if trajectory.avg_reward is not None and trajectory.avg_reward.numel() > 0:
                avg_reward = trajectory.avg_reward.mean().item()
            else:
                avg_reward = 0.0
            
            # Store metrics
            steps.append(i)
            total_losses.append(total_loss)
            policy_losses.append(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss)
            kl_losses.append(kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss)
            rewards.append(avg_reward)
            
            print(f"Step {i+1}, Total Loss: {total_loss:.4f}, Policy Loss: {policy_loss:.4f}, "
                  f"KL Loss: {kl_loss:.4f}, Reward: {avg_reward:.4f}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(steps, total_losses, label='Total Loss', color='blue')
        plt.plot(steps, policy_losses, label='Policy Loss', color='green')
        plt.plot(steps, kl_losses, label='KL Loss', color='red')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reward plot
        plt.subplot(2, 1, 2)
        plt.plot(steps, rewards, label='Avg Reward', color='purple')
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("test_logs/plots/test_visualization.png", dpi=150)
        
        print("\nTest complete! Visualization saved to test_logs/plots/test_visualization.png")
        
    finally:
        # Clean up
        hook_remover()

if __name__ == "__main__":
    main() 