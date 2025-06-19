#!/usr/bin/env python3
"""
Standalone script to generate plots from saved training data.

Usage:
    python generate_plots.py path/to/plot_data.pkl [--output-dir path/to/output]
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def load_plot_data(filename: str) -> Dict[str, Any]:
    """Load plot data from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_plots(data: Dict[str, Any], output_dir: str = None, custom_config: Dict = None):
    """
    Generate all plots from the saved data.
    
    Args:
        data: Dictionary containing all plot data
        output_dir: Directory to save plots (default: same directory as data file)
        custom_config: Optional custom configuration to override defaults
    """
    # Extract config from metadata
    config = data.get('metadata', {}).get('config', {})
    if custom_config:
        config.update(custom_config)
    
    # Get KL penalty coefficient for labels
    KL_PENALTY_COEFFICIENT = config.get('KL_PENALTY_COEFFICIENT', 0.1)
    
    # Create output directory
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all data arrays
    training_steps = data.get('training_steps', [])
    if len(training_steps) == 0:
        print("No data to plot!")
        return
    
    # Get all arrays and ensure consistent length
    min_length = len(training_steps)
    
    # Helper function to safely get data
    def get_data(key, default_value=0.0):
        arr = data.get(key, [])
        if not arr:
            return [default_value] * min_length
        return arr[:min_length]
    
    # Extract all metrics
    total_losses = get_data('total_losses')
    policy_losses = get_data('policy_losses')
    kl_losses = get_data('kl_losses')
    avg_rewards = get_data('avg_rewards')
    adapter_log_probs = get_data('adapter_log_probs')
    baseline_log_probs = get_data('baseline_log_probs')
    base_log_probs = get_data('base_log_probs')
    avg_advantages = get_data('avg_advantages')
    trajectory_log_probs = get_data('trajectory_log_probs')
    wikipedia_order_consistency = get_data('wikipedia_order_consistency', 0.5)
    kl_penalty_terms = get_data('kl_penalty_terms')
    reward_variance = get_data('reward_variance')
    gradient_magnitudes = get_data('gradient_magnitudes')
    step_log_probs = data.get('step_log_probs', [])
    policy_gradients = get_data('policy_gradients')
    clipping_ratios = get_data('clipping_ratios', 1.0)
    kl_from_ref = get_data('kl_from_ref')
    
    # Create comprehensive figure with 3 rows and 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # Plot 1: Loss Components
    axes[0].plot(training_steps, total_losses, 'b-', label='Total Loss', linewidth=2)
    axes[0].plot(training_steps, policy_losses, 'g--', label='Policy Loss', linewidth=1.5)
    axes[0].plot(training_steps, kl_penalty_terms, 'r:', label=f'KL Penalty (β={KL_PENALTY_COEFFICIENT})', linewidth=1.5)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Components')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rewards
    axes[1].plot(training_steps, avg_rewards, 'purple', linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward')
    axes[1].grid(True, alpha=0.3)
    if len(training_steps) > 10:
        z = np.polyfit(training_steps, avg_rewards, 1)
        p = np.poly1d(z)
        axes[1].plot(training_steps, p(training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        axes[1].legend(fontsize=8)
    
    # Plot 3: Log Probabilities
    axes[2].plot(training_steps, adapter_log_probs, 'darkgreen', label='Adapter Model', linewidth=2)
    axes[2].plot(training_steps, baseline_log_probs, 'orange', label='Baseline Model', linewidth=2)
    axes[2].plot(training_steps, base_log_probs, 'blue', label='Base Model', linewidth=2)
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Avg Log Prob (per token)')
    axes[2].set_title('Model Log Probabilities')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Advantages
    axes[3].plot(training_steps, avg_advantages, 'brown', linewidth=2, label='Avg Advantage')
    axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[3].set_xlabel('Training Step')
    axes[3].set_ylabel('Average Advantage')
    axes[3].set_title('Advantage Statistics')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)
    if len(training_steps) > 10:
        z = np.polyfit(training_steps, avg_advantages, 1)
        p = np.poly1d(z)
        axes[3].plot(training_steps, p(training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        axes[3].legend(fontsize=8)
    
    # Plot 5: Trajectory-Level Log Probabilities
    axes[4].plot(training_steps, trajectory_log_probs, 'darkblue', linewidth=2)
    axes[4].set_xlabel('Training Step')
    axes[4].set_ylabel('Trajectory Log Prob')
    axes[4].set_title('Trajectory-Level Log Probabilities')
    axes[4].grid(True, alpha=0.3)
    if len(training_steps) > 10:
        z = np.polyfit(training_steps, trajectory_log_probs, 1)
        p = np.poly1d(z)
        axes[4].plot(training_steps, p(training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        axes[4].legend(fontsize=8)
    
    # Plot 6: Gradient Norm
    axes[5].plot(training_steps, gradient_magnitudes, 'red', linewidth=2)
    axes[5].set_xlabel('Training Step')
    axes[5].set_ylabel('Gradient Norm')
    axes[5].set_title('Gradient Norm Over Time')
    axes[5].grid(True, alpha=0.3)
    axes[5].set_yscale('log')
    
    # Plot 7: Wikipedia Order Consistency
    axes[6].plot(training_steps, wikipedia_order_consistency, 'teal', linewidth=2)
    axes[6].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    axes[6].axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect Order (1.0)')
    axes[6].axhline(y=0.0, color='red', linestyle='--', alpha=0.3, label='Reverse Order (0.0)')
    axes[6].set_xlabel('Training Step')
    axes[6].set_ylabel('Order Consistency')
    axes[6].set_title('Wikipedia Key Selection Order')
    axes[6].set_ylim(-0.1, 1.1)
    axes[6].legend(fontsize=7)
    axes[6].grid(True, alpha=0.3)
    
    # Plot 8: Policy Gradients
    axes[7].plot(training_steps, policy_gradients, 'darkgreen', linewidth=2)
    axes[7].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[7].set_xlabel('Training Step')
    axes[7].set_ylabel('Policy Gradient')
    axes[7].set_title('Policy Gradient (positive = reinforce)')
    axes[7].grid(True, alpha=0.3)
    if len(policy_gradients) > 10:
        mean_grad = np.mean(policy_gradients)
        axes[7].text(0.02, 0.98, f'Mean: {mean_grad:.4f}', transform=axes[7].transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Plot 9: KL Divergence from Reference
    axes[8].plot(training_steps, kl_from_ref, 'darkred', linewidth=2)
    axes[8].set_xlabel('Training Step')
    axes[8].set_ylabel('KL Divergence')
    axes[8].set_title('KL Divergence from Reference Model (π_ref)')
    axes[8].grid(True, alpha=0.3)
    if len(kl_from_ref) > 0 and any(kl_from_ref):
        mean_kl = np.mean(kl_from_ref)
        axes[8].axhline(y=mean_kl, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_kl:.4f}')
        axes[8].legend(fontsize=8)
    
    # Plot 10: Reward Variance
    axes[9].plot(training_steps, reward_variance, 'magenta', linewidth=2)
    axes[9].set_xlabel('Training Step')
    axes[9].set_ylabel('Reward Variance')
    axes[9].set_title('Reward Variance Within Trajectory')
    axes[9].grid(True, alpha=0.3)
    
    # Plot 11: Step-Indexed Log Probabilities
    if len(step_log_probs) > 0 and any(len(ep) > 0 for ep in step_log_probs):
        # Get NUM_KV_PAIRS from config or infer from data
        NUM_KV_PAIRS = config.get('NUM_KV_PAIRS', 15)
        step_indices = list(range(NUM_KV_PAIRS))
        
        # Divide episodes into thirds
        total_episodes = len(step_log_probs)
        first_third_end = total_episodes // 3
        second_third_end = 2 * total_episodes // 3
        
        def compute_avg_for_period(start_idx, end_idx):
            avg_log_probs_by_step = []
            for step_idx in step_indices:
                step_log_probs_period = []
                for episode_idx in range(start_idx, min(end_idx, len(step_log_probs))):
                    episode_log_probs = step_log_probs[episode_idx]
                    if step_idx < len(episode_log_probs):
                        step_log_probs_period.append(episode_log_probs[step_idx])
                
                if step_log_probs_period:
                    avg_log_prob = sum(step_log_probs_period) / len(step_log_probs_period)
                    avg_log_probs_by_step.append(avg_log_prob)
                else:
                    avg_log_probs_by_step.append(0.0)
            return avg_log_probs_by_step
        
        if total_episodes >= 9:
            first_third = compute_avg_for_period(0, first_third_end)
            second_third = compute_avg_for_period(first_third_end, second_third_end)
            third_third = compute_avg_for_period(second_third_end, total_episodes)
            
            axes[10].plot(step_indices, first_third, 'lightcoral', linewidth=2, marker='o', 
                         markersize=3, label=f'Early (eps 0-{first_third_end})', alpha=0.8)
            axes[10].plot(step_indices, second_third, 'gold', linewidth=2, marker='s', 
                         markersize=3, label=f'Mid (eps {first_third_end}-{second_third_end})', alpha=0.8)
            axes[10].plot(step_indices, third_third, 'mediumseagreen', linewidth=2, marker='^', 
                         markersize=3, label=f'Late (eps {second_third_end}+)', alpha=0.8)
            axes[10].legend(fontsize=7)
        else:
            overall_avg = compute_avg_for_period(0, total_episodes)
            axes[10].plot(step_indices, overall_avg, 'darkviolet', linewidth=2, marker='o', 
                         markersize=4, label='Overall Average')
            axes[10].legend(fontsize=8)
        
        axes[10].set_xlabel('Step Index')
        axes[10].set_ylabel('Avg Log Prob of Selected Action')
        axes[10].set_title('Log Prob by Step Index (training progression)')
        axes[10].grid(True, alpha=0.3)
        axes[10].set_xticks(step_indices)
    else:
        axes[10].text(0.5, 0.5, 'No step log prob data\navailable yet', 
                     ha='center', va='center', transform=axes[10].transAxes, fontsize=10)
        axes[10].set_title('Log Prob by Step Index')
    
    # Plot 12: PPO Clipping Ratio
    axes[11].plot(training_steps, clipping_ratios, 'navy', linewidth=2)
    axes[11].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Lower clip (0.8)')
    axes[11].axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Upper clip (1.2)')
    axes[11].axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, label='No change (1.0)')
    axes[11].set_xlabel('Training Step')
    axes[11].set_ylabel('Average Clipping Ratio')
    axes[11].set_title('PPO Clipping Ratio (π_θ / π_old)')
    axes[11].legend(fontsize=7)
    axes[11].grid(True, alpha=0.3)
    axes[11].set_ylim(0.5, 1.5)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")
    
    # Create additional loss breakdown plot if enough data
    if len(training_steps) > 20:
        plt.figure(figsize=(12, 6))
        plt.stackplot(training_steps, policy_losses, kl_penalty_terms,
                     labels=['Policy Loss', 'KL Penalty'],
                     colors=['green', 'red'], alpha=0.7)
        plt.plot(training_steps, total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss Composition Over Time')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        breakdown_path = os.path.join(output_dir, 'loss_breakdown.png')
        plt.savefig(breakdown_path, dpi=150)
        plt.close()
        print(f"Saved loss breakdown to: {breakdown_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from saved training data')
    parser.add_argument('data_file', help='Path to the pickle file containing plot data')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Directory to save plots (default: current directory)')
    parser.add_argument('--kl-coef', type=float, default=None,
                       help='Override KL penalty coefficient for labels')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data_file}")
    data = load_plot_data(args.data_file)
    
    # Print some info about the data
    metadata = data.get('metadata', {})
    print(f"Data from episode: {metadata.get('episode', 'unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"Number of training steps: {len(data.get('training_steps', []))}")
    
    # Prepare custom config if needed
    custom_config = {}
    if args.kl_coef is not None:
        custom_config['KL_PENALTY_COEFFICIENT'] = args.kl_coef
    
    # Generate plots
    generate_plots(data, args.output_dir, custom_config)


if __name__ == '__main__':
    main() 