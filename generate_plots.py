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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (same as main training)
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def load_plot_data(filename: str) -> Dict[str, Any]:
    """Load plot data from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_plots(data: Dict[str, Any], output_dir: Optional[str] = None, custom_config: Optional[Dict[str, Any]] = None):
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
    
    print(f"Plotting {len(training_steps)} training steps")
    
    # Check for missing required keys
    required_keys = [
        'total_losses', 'policy_losses', 'kl_losses', 'avg_rewards',
        'adapter_log_probs', 'baseline_log_probs', 'base_log_probs',
        'avg_advantages', 'trajectory_log_probs', 'wikipedia_order_consistency',
        'kl_penalty_terms', 'reward_variance', 'gradient_magnitudes',
        'step_log_probs', 'policy_gradients', 'clipping_ratios',
        'kl_from_ref', 'batch_selection_entropy',
        'lora_layer_gradients', 'advantage_distributions', 'similarity_score_stats'
    ]
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"Error: Missing required keys: {missing_keys}")
        return
    
    # Extract all metrics (arrays must have consistent lengths)
    min_length = len(training_steps)
    
    # Core metrics
    total_losses = data['total_losses'][:min_length]
    policy_losses = data['policy_losses'][:min_length]
    kl_losses = data['kl_losses'][:min_length]
    avg_rewards = data['avg_rewards'][:min_length]
    adapter_log_probs = data['adapter_log_probs'][:min_length]
    baseline_log_probs = data['baseline_log_probs'][:min_length]
    base_log_probs = data['base_log_probs'][:min_length]
    avg_advantages = data['avg_advantages'][:min_length]
    trajectory_log_probs = data['trajectory_log_probs'][:min_length]
    wikipedia_order_consistency = data['wikipedia_order_consistency'][:min_length]
    kl_penalty_terms = data['kl_penalty_terms'][:min_length]
    reward_variance = data['reward_variance'][:min_length]
    gradient_magnitudes = data['gradient_magnitudes'][:min_length]
    step_log_probs = data['step_log_probs']
    policy_gradients = data['policy_gradients'][:min_length]
    clipping_ratios = data['clipping_ratios'][:min_length]
    kl_from_ref = data['kl_from_ref'][:min_length]
    batch_selection_entropy = data['batch_selection_entropy'][:min_length]
    
    # Enhanced debugging metrics (required)
    lora_layer_gradients = data['lora_layer_gradients']
    advantage_distributions = data['advantage_distributions'][:min_length]
    similarity_score_stats = data['similarity_score_stats'][:min_length]
    
    # Create comprehensive figure with 3 rows and 4 columns (removing 2 redundant plots)
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
    
    # Plot 2: Rewards with Baseline Update Markers
    axes[1].plot(training_steps, avg_rewards, 'purple', linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward (Baseline Updates Marked)')
    axes[1].grid(True, alpha=0.3)
    if len(training_steps) > 10:
        z = np.polyfit(training_steps, avg_rewards, 1)
        p = np.poly1d(z)
        axes[1].plot(training_steps, p(training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        axes[1].legend(fontsize=8)
    
    # Add baseline update markers
    BASELINE_UPDATE_FREQUENCY = config['BASELINE_UPDATE_FREQUENCY']
    for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
        if episode <= max(training_steps):
            axes[1].axvline(x=episode, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Plot 3: Model Log Probabilities
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
    
    # Plot 5: Gradient Norm
    axes[4].plot(training_steps, gradient_magnitudes, 'red', linewidth=2)
    axes[4].set_xlabel('Training Step')
    axes[4].set_ylabel('Gradient Norm')
    axes[4].set_title('Gradient Norm Over Time')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_yscale('log')
    
    # Plot 6: Wikipedia Order Consistency
    axes[5].plot(training_steps, wikipedia_order_consistency, 'teal', linewidth=2)
    axes[5].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    axes[5].axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect Order (1.0)')
    axes[5].axhline(y=0.0, color='red', linestyle='--', alpha=0.3, label='Reverse Order (0.0)')
    axes[5].set_xlabel('Training Step')
    axes[5].set_ylabel('Order Consistency')
    axes[5].set_title('Wikipedia Key Selection Order')
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].legend(fontsize=7)
    axes[5].grid(True, alpha=0.3)
    
    # Plot 7: KL Divergence from Reference
    axes[6].plot(training_steps, kl_from_ref, 'darkred', linewidth=2)
    axes[6].set_xlabel('Training Step')
    axes[6].set_ylabel('KL Divergence')
    axes[6].set_title('KL Divergence from Reference Model (π_ref)')
    axes[6].grid(True, alpha=0.3)
    mean_kl = np.mean(kl_from_ref)
    axes[6].axhline(y=mean_kl, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_kl:.4f}')
    axes[6].legend(fontsize=8)
    
    # Plot 8: Reward Variance
    axes[7].plot(training_steps, reward_variance, 'magenta', linewidth=2)
    axes[7].set_xlabel('Training Step')
    axes[7].set_ylabel('Reward Variance')
    axes[7].set_title('Reward Variance Within Trajectory')
    axes[7].grid(True, alpha=0.3)
    
    # Plot 9: Step-Indexed Log Probabilities
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
            
            axes[8].plot(step_indices, first_third, 'lightcoral', linewidth=2, marker='o', 
                         markersize=3, label=f'Early (eps 0-{first_third_end})', alpha=0.8)
            axes[8].plot(step_indices, second_third, 'gold', linewidth=2, marker='s', 
                         markersize=3, label=f'Mid (eps {first_third_end}-{second_third_end})', alpha=0.8)
            axes[8].plot(step_indices, third_third, 'mediumseagreen', linewidth=2, marker='^', 
                         markersize=3, label=f'Late (eps {second_third_end}+)', alpha=0.8)
            axes[8].legend(fontsize=7)
        else:
            overall_avg = compute_avg_for_period(0, total_episodes)
            axes[8].plot(step_indices, overall_avg, 'darkviolet', linewidth=2, marker='o', 
                         markersize=4, label='Overall Average')
            axes[8].legend(fontsize=8)
        
        axes[8].set_xlabel('Step Index')
        axes[8].set_ylabel('Avg Log Prob of Selected Action')
        axes[8].set_title('Log Prob by Step Index (training progression)')
        axes[8].grid(True, alpha=0.3)
        axes[8].set_xticks(step_indices)
    else:
        axes[8].text(0.5, 0.5, 'No step log prob data\navailable yet', 
                     ha='center', va='center', transform=axes[8].transAxes, fontsize=10)
        axes[8].set_title('Log Prob by Step Index')
    
    # Plot 10: PPO Clipping Ratio
    axes[9].plot(training_steps, clipping_ratios, 'navy', linewidth=2)
    axes[9].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Lower clip (0.8)')
    axes[9].axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Upper clip (1.2)')
    axes[9].axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, label='No change (1.0)')
    axes[9].set_xlabel('Training Step')
    axes[9].set_ylabel('Average Clipping Ratio')
    axes[9].set_title('PPO Clipping Ratio (π_θ / π_old)')
    axes[9].legend(fontsize=7)
    axes[9].grid(True, alpha=0.3)
    axes[9].set_ylim(0.5, 1.5)
    
    # Plot 11: LoRA Layer Gradient Flow
    for layer_idx, grad_history in lora_layer_gradients.items():
        # Ensure gradient history matches training_steps length
        grad_data = grad_history[:min_length] + [0.0] * max(0, min_length - len(grad_history))
        color = 'red' if layer_idx == max(lora_layer_gradients.keys()) else f'C{layer_idx % 10}'
        linewidth = 3 if layer_idx == max(lora_layer_gradients.keys()) else 1.5
        alpha = 1.0 if layer_idx == max(lora_layer_gradients.keys()) else 0.7
        label = f'Layer {layer_idx}' + (' (Query Layer)' if layer_idx == max(lora_layer_gradients.keys()) else '')
        axes[10].plot(training_steps, grad_data, color=color, linewidth=linewidth, alpha=alpha, label=label)
    axes[10].set_xlabel('Training Step')
    axes[10].set_ylabel('Gradient Magnitude')
    axes[10].set_title('LoRA Layer Gradient Flow')
    axes[10].set_yscale('log')
    axes[10].legend(fontsize=6, loc='upper right')
    axes[10].grid(True, alpha=0.3)
    
    # Plot 12: Advantage Distribution Analysis
    positive_percentages = [d['positive_percentage'] for d in advantage_distributions]
    negative_percentages = [d['negative_percentage'] for d in advantage_distributions]
    
    axes[11].plot(training_steps, positive_percentages, 'green', linewidth=2, label='Positive Advantages')
    axes[11].plot(training_steps, negative_percentages, 'red', linewidth=2, label='Negative Advantages')
    axes[11].axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='50% (Balanced)')
    axes[11].set_xlabel('Training Step')
    axes[11].set_ylabel('Percentage (%)')
    axes[11].set_title('Advantage Distribution (Step-Level Learning Signal)')
    axes[11].set_ylim(0, 100)
    axes[11].legend(fontsize=7)
    axes[11].grid(True, alpha=0.3)
    
    # Add baseline update markers
    BASELINE_UPDATE_FREQUENCY = config['BASELINE_UPDATE_FREQUENCY']
    for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
        if episode <= max(training_steps):
            axes[11].axvline(x=episode, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    
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

    # Create additional similarity score analysis plot
    if len(training_steps) > 5:
        plt.figure(figsize=(15, 10))
        
        # Extract similarity metrics
        similarity_means = [s['mean'] for s in similarity_score_stats]
        similarity_stds = [s['std'] for s in similarity_score_stats]
        similarity_entropies = [s['entropy'] for s in similarity_score_stats]
        similarity_maxs = [s['max'] for s in similarity_score_stats]
        similarity_mins = [s['min'] for s in similarity_score_stats]
        
        # Create 2x2 subplot for detailed similarity analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Subplot 1: Mean similarity scores
        ax1.plot(training_steps, similarity_means, 'blue', linewidth=2, label='Mean Similarity')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Similarity Score')
        ax1.set_title('Mean Query-Key Similarity Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Similarity entropy (measure of specificity)
        ax2.plot(training_steps, similarity_entropies, 'green', linewidth=2, label='Similarity Entropy')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Entropy (nats)')
        ax2.set_title('Query Specificity (Lower = More Specific)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Similarity range (max - min)
        similarity_ranges = [mx - mn for mx, mn in zip(similarity_maxs, similarity_mins)]
        ax3.plot(training_steps, similarity_ranges, 'orange', linewidth=2, label='Range (Max - Min)')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Similarity Range')
        ax3.set_title('Query Discrimination Ability')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Standard deviation
        ax4.plot(training_steps, similarity_stds, 'red', linewidth=2, label='Similarity Std Dev')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Similarity Score Variability')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add baseline update markers to all subplots
        BASELINE_UPDATE_FREQUENCY = config['BASELINE_UPDATE_FREQUENCY']
        for ax in [ax1, ax2, ax3, ax4]:
            for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
                if episode <= max(training_steps):
                    ax.axvline(x=episode, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        similarity_path = os.path.join(output_dir, 'similarity_analysis.png')
        plt.savefig(similarity_path, dpi=150)
        plt.close()
        print(f"Saved similarity analysis to: {similarity_path}")

    # Create step-level learning dynamics analysis
    if len(training_steps) > 10:
        plt.figure(figsize=(12, 8))
        
        # Extract step-level learning metrics over time
        positive_advantages = [d['positive_percentage'] for d in advantage_distributions]
        advantage_means = [d['mean'] for d in advantage_distributions]
        advantage_stds = [d['std'] for d in advantage_distributions]
        
        # Create 2x2 subplot for step-level analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Subplot 1: Learning signal strength (positive advantage percentage)
        ax1.plot(training_steps, positive_advantages, 'green', linewidth=2, label='Positive Advantage %')
        ax1.axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='50% (Balanced)')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Step-Level Learning Signal Strength')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Advantage magnitude evolution
        ax2.plot(training_steps, advantage_means, 'blue', linewidth=2, label='Mean Advantage')
        ax2.axhline(y=0.0, color='gray', linestyle='-', alpha=0.5, label='Zero Advantage')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Advantage')
        ax2.set_title('Advantage Magnitude Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Advantage variability (indicates learning consistency)
        ax3.plot(training_steps, advantage_stds, 'orange', linewidth=2, label='Advantage Std Dev')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Learning Signal Variability')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Combined learning health score
        # Normalize metrics to 0-1 scale for combination
        norm_positive = np.array(positive_advantages) / 100.0  # Already 0-100, normalize to 0-1
        norm_mean_adv = np.array(advantage_means)
        norm_mean_adv = (norm_mean_adv - np.min(norm_mean_adv)) / (np.max(norm_mean_adv) - np.min(norm_mean_adv) + 1e-8)
        
        # Health score: high positive percentage + positive trend in advantages
        health_score = 0.6 * norm_positive + 0.4 * norm_mean_adv
        ax4.plot(training_steps, health_score * 100, 'purple', linewidth=2, label='Learning Health Score')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Health Score (%)')
        ax4.set_title('Overall Step-Level Learning Health')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add baseline update markers to all subplots
        BASELINE_UPDATE_FREQUENCY = config['BASELINE_UPDATE_FREQUENCY']
        for ax in [ax1, ax2, ax3, ax4]:
            for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
                if episode <= max(training_steps):
                    ax.axvline(x=episode, color='red', linestyle=':', alpha=0.4, linewidth=1)
        
        plt.tight_layout()
        step_analysis_path = os.path.join(output_dir, 'step_learning_analysis.png')
        plt.savefig(step_analysis_path, dpi=150)
        plt.close()
        print(f"Saved step-level learning analysis to: {step_analysis_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from saved training data')
    parser.add_argument('data_file', help='Path to the pickle file containing plot data')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Directory to save plots (default: same directory as input file)')
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
    
    # If no output directory specified, use the directory of the input file
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.data_file))
    
    # Prepare custom config if needed
    custom_config = {}
    if args.kl_coef is not None:
        custom_config['KL_PENALTY_COEFFICIENT'] = args.kl_coef
    
    # Generate plots
    generate_plots(data, args.output_dir, custom_config)


if __name__ == '__main__':
    main() 