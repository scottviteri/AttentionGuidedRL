#!/usr/bin/env python3
"""
Example script showing how to create custom plots from saved training data.

This demonstrates how to:
1. Load the pickle data
2. Access specific metrics
3. Create custom visualizations
4. Combine multiple runs for comparison
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import glob
import os


def load_plot_data(filename: str) -> Dict:
    """Load plot data from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def example_custom_reward_plot(data: Dict, output_file: str = 'custom_reward_plot.png'):
    """
    Example: Create a custom reward plot with confidence intervals.
    """
    training_steps = data['training_steps']
    avg_rewards = data['avg_rewards']
    reward_variance = data.get('reward_variance', [0] * len(avg_rewards))
    
    # Convert to numpy for easier manipulation
    steps = np.array(training_steps)
    rewards = np.array(avg_rewards)
    variance = np.array(reward_variance)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot rewards with shaded variance
    plt.plot(steps, rewards, 'b-', linewidth=2, label='Average Reward')
    
    # Add shaded region for variance (if available)
    if np.any(variance > 0):
        std_dev = np.sqrt(variance)
        plt.fill_between(steps, rewards - std_dev, rewards + std_dev, 
                        alpha=0.3, color='blue', label='±1 std dev')
    
    # Add smoothed trend
    if len(steps) > 20:
        window = min(20, len(steps) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window//2:-window//2+1]
        plt.plot(smooth_steps, smoothed, 'r--', linewidth=2, 
                label=f'Smoothed (window={window})')
    
    plt.xlabel('Training Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress: Reward Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved custom reward plot to: {output_file}")


def example_compare_multiple_runs(data_files: List[str], output_file: str = 'comparison_plot.png'):
    """
    Example: Compare rewards across multiple training runs.
    """
    plt.figure(figsize=(12, 6))
    
    for i, filename in enumerate(data_files):
        data = load_plot_data(filename)
        training_steps = data['training_steps']
        avg_rewards = data['avg_rewards']
        
        # Extract run name from filename
        run_name = os.path.basename(filename).replace('.pkl', '')
        
        plt.plot(training_steps, avg_rewards, linewidth=2, 
                label=f'Run {i+1}: {run_name}')
    
    plt.xlabel('Training Episode')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Multiple Training Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {output_file}")


def example_loss_analysis(data: Dict, output_file: str = 'loss_analysis.png'):
    """
    Example: Detailed loss component analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    training_steps = data['training_steps']
    total_losses = data['total_losses']
    policy_losses = data['policy_losses']
    kl_losses = data['kl_losses']
    kl_penalty_terms = data.get('kl_penalty_terms', kl_losses)
    
    # Plot 1: Loss components over time
    ax = axes[0, 0]
    ax.plot(training_steps, total_losses, 'b-', label='Total Loss')
    ax.plot(training_steps, policy_losses, 'g--', label='Policy Loss')
    ax.plot(training_steps, kl_penalty_terms, 'r:', label='KL Penalty')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss ratios
    ax = axes[0, 1]
    policy_ratio = np.array(policy_losses) / (np.array(total_losses) + 1e-8)
    kl_ratio = np.array(kl_penalty_terms) / (np.array(total_losses) + 1e-8)
    ax.plot(training_steps, policy_ratio, 'g-', label='Policy Loss Ratio')
    ax.plot(training_steps, kl_ratio, 'r-', label='KL Penalty Ratio')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Ratio to Total Loss')
    ax.set_title('Loss Component Ratios')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss derivatives (rate of change)
    ax = axes[1, 0]
    if len(training_steps) > 1:
        total_loss_diff = np.diff(total_losses)
        policy_loss_diff = np.diff(policy_losses)
        ax.plot(training_steps[1:], total_loss_diff, 'b-', label='Total Loss Change')
        ax.plot(training_steps[1:], policy_loss_diff, 'g--', label='Policy Loss Change')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Change (Δ)')
        ax.set_title('Rate of Loss Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative losses
    ax = axes[1, 1]
    cumulative_total = np.cumsum(total_losses)
    cumulative_policy = np.cumsum(np.abs(policy_losses))
    cumulative_kl = np.cumsum(kl_penalty_terms)
    ax.plot(training_steps, cumulative_total, 'b-', label='Cumulative Total')
    ax.plot(training_steps, cumulative_policy, 'g--', label='Cumulative |Policy|')
    ax.plot(training_steps, cumulative_kl, 'r:', label='Cumulative KL')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cumulative Loss')
    ax.set_title('Cumulative Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved loss analysis to: {output_file}")


def example_advantage_distribution(data: Dict, output_file: str = 'advantage_dist.png'):
    """
    Example: Analyze advantage distribution over training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    training_steps = data['training_steps']
    avg_advantages = data['avg_advantages']
    
    # Plot 1: Advantage over time with zero line
    ax = axes[0]
    ax.plot(training_steps, avg_advantages, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.fill_between(training_steps, 0, avg_advantages, 
                   where=(np.array(avg_advantages) > 0), 
                   color='green', alpha=0.3, label='Positive')
    ax.fill_between(training_steps, 0, avg_advantages, 
                   where=(np.array(avg_advantages) <= 0), 
                   color='red', alpha=0.3, label='Negative')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Advantage')
    ax.set_title('Advantage Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Advantage histogram
    ax = axes[1]
    ax.hist(avg_advantages, bins=30, density=True, alpha=0.7, color='purple')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=np.mean(avg_advantages), color='red', linestyle='--', 
              label=f'Mean: {np.mean(avg_advantages):.3f}')
    ax.set_xlabel('Average Advantage')
    ax.set_ylabel('Density')
    ax.set_title('Advantage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved advantage distribution to: {output_file}")


def main():
    """
    Example usage of custom plotting functions.
    """
    # Example 1: Load and plot single run
    latest_data_file = 'logs/*/plots/plot_data_latest.pkl'
    files = glob.glob(latest_data_file)
    
    if files:
        print(f"Found {len(files)} data file(s)")
        
        # Load the most recent one
        latest_file = sorted(files)[-1]
        print(f"\nLoading data from: {latest_file}")
        data = load_plot_data(latest_file)
        
        # Create custom plots
        example_custom_reward_plot(data)
        example_loss_analysis(data)
        example_advantage_distribution(data)
        
        # Example 2: Compare multiple runs if available
        if len(files) > 1:
            print("\nComparing multiple runs...")
            example_compare_multiple_runs(files[-3:])  # Compare last 3 runs
    else:
        print("No plot data files found!")
        print("Run training with the updated code to generate plot data files.")
        print(f"Looking for files matching: {latest_data_file}")


if __name__ == '__main__':
    main() 