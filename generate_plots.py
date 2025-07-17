#!/usr/bin/env python3
"""
Standalone script to generate plots from saved training data.

Usage:
    # Use explicit path to plot_data.pkl
    python generate_plots.py path/to/plot_data.pkl [--output-dir path/to/output] [--smooth-window N]
    
    # Use nth most recent log folder (automatically finds logs/YYYYMMDD-HHMMSS/plots/plot_data.pkl)
    python generate_plots.py --recent N [--output-dir path/to/output] [--smooth-window N] [--logs-dir path/to/logs]
    
Examples:
    # Generate plots from most recent training run
    python generate_plots.py --recent 0
    
    # Generate plots from second most recent run with smoothing
    python generate_plots.py --recent 1 --smooth-window 5
    
    # Generate plots from third most recent run, save to custom directory
    python generate_plots.py --recent 2 --output-dir ./custom_plots/
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (same as main training)
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import glob
from datetime import datetime


def find_nth_most_recent_log_folder(n: int = 0, logs_dir: str = "logs") -> Optional[str]:
    """
    Find the nth most recent log folder based on date in folder names.
    
    Args:
        n: Index of folder to select (0 = most recent, 1 = second most recent, etc.)
        logs_dir: Base logs directory path
        
    Returns:
        Path to the nth most recent log folder, or None if not found
    """
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' does not exist")
        return None
    
    # Get all date-based folders (format: YYYYMMDD-HHMMSS)
    date_folders = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path) and len(item) == 15 and item[8] == '-':
            try:
                # Validate date format by parsing it
                datetime.strptime(item, "%Y%m%d-%H%M%S")
                date_folders.append(item)
            except ValueError:
                continue  # Skip folders that don't match the expected date format
    
    if not date_folders:
        print(f"Error: No date-based log folders found in '{logs_dir}'")
        return None
    
    # Sort by date (most recent first)
    date_folders.sort(reverse=True)
    
    if n >= len(date_folders):
        print(f"Error: Requested folder index {n} but only {len(date_folders)} folders available")
        print(f"Available folders: {date_folders}")
        return None
    
    selected_folder = date_folders[n]
    full_path = os.path.join(logs_dir, selected_folder)
    
    # Check if plot_data.pkl exists
    plot_data_path = os.path.join(full_path, "plots", "plot_data.pkl")
    if not os.path.exists(plot_data_path):
        print(f"Error: plot_data.pkl not found in {full_path}/plots/")
        return None
    
    print(f"Selected folder: {selected_folder} (index {n})")
    print(f"Plot data path: {plot_data_path}")
    return full_path


def smooth_data(data: List[float], window_size: int) -> List[float]:
    """
    Apply moving average smoothing to data.
    
    Args:
        data: List of values to smooth
        window_size: Size of the smoothing window
        
    Returns:
        List of smoothed values (same length as input)
    """
    if window_size <= 1 or len(data) <= 1:
        return data
    
    # Ensure window size doesn't exceed data length
    window_size = min(window_size, len(data))
    
    smoothed = []
    for i in range(len(data)):
        # Calculate the window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        
        # Take the mean of values in the window
        window_values = data[start_idx:end_idx]
        smoothed.append(np.mean(window_values))
    
    return smoothed


def load_plot_data(filename: str) -> Dict[str, Any]:
    """Load plot data from pickle file with better error handling."""
    import os
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Plot data file not found: {filename}")
    
    # Check file size
    file_size = os.path.getsize(filename)
    if file_size == 0:
        raise ValueError(f"Plot data file is empty: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except pickle.UnpicklingError as e:
        # More helpful error message for corrupted files
        raise ValueError(
            f"Plot data file appears to be corrupted: {filename}\n"
            f"Original error: {e}\n"
            f"File size: {file_size} bytes\n"
            f"You may need to re-run training to regenerate this file."
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to load plot data from {filename}: {e}"
        ) from e
    
    return data


def generate_plots(data: Dict[str, Any], output_dir: Optional[str] = None, custom_config: Optional[Dict[str, Any]] = None, smooth_window: int = 1):
    """
    Generate all plots from the saved data.
    
    Args:
        data: Dictionary containing all plot data
        output_dir: Directory to save plots (default: same directory as data file)
        custom_config: Optional custom configuration to override defaults
        smooth_window: Size of smoothing window (1 = no smoothing)
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
    if smooth_window > 1:
        print(f"Applying smoothing with window size: {smooth_window}")
    
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
    
    # Core metrics - apply smoothing
    total_losses = smooth_data(data['total_losses'][:min_length], smooth_window)
    policy_losses = smooth_data(data['policy_losses'][:min_length], smooth_window)
    kl_losses = smooth_data(data['kl_losses'][:min_length], smooth_window)
    avg_rewards = smooth_data(data['avg_rewards'][:min_length], smooth_window)
    adapter_log_probs = smooth_data(data['adapter_log_probs'][:min_length], smooth_window)
    baseline_log_probs = smooth_data(data['baseline_log_probs'][:min_length], smooth_window)
    base_log_probs = smooth_data(data['base_log_probs'][:min_length], smooth_window)
    avg_advantages = smooth_data(data['avg_advantages'][:min_length], smooth_window)
    trajectory_log_probs = smooth_data(data['trajectory_log_probs'][:min_length], smooth_window)
    wikipedia_order_consistency = smooth_data(data['wikipedia_order_consistency'][:min_length], smooth_window)
    kl_penalty_terms = smooth_data(data['kl_penalty_terms'][:min_length], smooth_window)
    reward_variance = smooth_data(data['reward_variance'][:min_length], smooth_window)
    gradient_magnitudes = smooth_data(data['gradient_magnitudes'][:min_length], smooth_window)
    step_log_probs = data['step_log_probs']
    policy_gradients = smooth_data(data['policy_gradients'][:min_length], smooth_window)
    clipping_ratios = smooth_data(data['clipping_ratios'][:min_length], smooth_window)
    kl_from_ref = smooth_data(data['kl_from_ref'][:min_length], smooth_window)
    batch_selection_entropy = smooth_data(data['batch_selection_entropy'][:min_length], smooth_window)
    
    # Enhanced debugging metrics (required)
    lora_layer_gradients = data['lora_layer_gradients']
    advantage_distributions = data['advantage_distributions'][:min_length]
    similarity_score_stats = data['similarity_score_stats'][:min_length]
    
    # Create comprehensive figure with 3 rows and 4 columns (removing 2 redundant plots)
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    # Determine title suffix for smoothing
    title_suffix = f" (smoothed, window={smooth_window})" if smooth_window > 1 else ""
    
    # Plot 1: Loss Components
    axes[0].plot(training_steps, total_losses, 'b-', label='Total Loss', linewidth=2)
    axes[0].plot(training_steps, policy_losses, 'g--', label='Policy Loss', linewidth=1.5)
    axes[0].plot(training_steps, kl_penalty_terms, 'r:', label=f'KL Penalty (β={KL_PENALTY_COEFFICIENT})', linewidth=1.5)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Components{title_suffix}')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rewards with Baseline Update Markers
    axes[1].plot(training_steps, avg_rewards, 'purple', linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title(f'Average Reward (Baseline Updates Marked){title_suffix}')
    axes[1].grid(True, alpha=0.3)
    if len(training_steps) > 10:
        z = np.polyfit(training_steps, avg_rewards, 1)
        p = np.poly1d(z)
        axes[1].plot(training_steps, p(training_steps), "k--", alpha=0.5, label=f'Trend (slope={z[0]:.2e})')
        axes[1].legend(fontsize=8)
    
    # Add baseline update markers
    BASELINE_UPDATE_FREQUENCY = config.get('BASELINE_UPDATE_FREQUENCY', 10)
    for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
        if episode <= max(training_steps):
            axes[1].axvline(x=episode, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Plot 3: Model Log Probabilities
    axes[2].plot(training_steps, adapter_log_probs, 'darkgreen', label='Adapter Model', linewidth=2)
    axes[2].plot(training_steps, baseline_log_probs, 'orange', label='Baseline Model', linewidth=2)
    axes[2].plot(training_steps, base_log_probs, 'blue', label='Base Model', linewidth=2)
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Avg Log Prob (per token)')
    axes[2].set_title(f'Model Log Probabilities{title_suffix}')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Advantages
    axes[3].plot(training_steps, avg_advantages, 'brown', linewidth=2, label='Avg Advantage')
    axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[3].set_xlabel('Training Step')
    axes[3].set_ylabel('Average Advantage')
    axes[3].set_title(f'Advantage Statistics{title_suffix}')
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
    axes[4].set_title(f'Gradient Norm Over Time{title_suffix}')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_yscale('log')
    
    # Plot 6: Wikipedia Order Consistency
    axes[5].plot(training_steps, wikipedia_order_consistency, 'teal', linewidth=2)
    axes[5].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    axes[5].axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect Order (1.0)')
    axes[5].axhline(y=0.0, color='red', linestyle='--', alpha=0.3, label='Reverse Order (0.0)')
    axes[5].set_xlabel('Training Step')
    axes[5].set_ylabel('Order Consistency')
    axes[5].set_title(f'Wikipedia Key Selection Order{title_suffix}')
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].legend(fontsize=7)
    axes[5].grid(True, alpha=0.3)
    
    # Plot 7: KL Divergence from Reference
    axes[6].plot(training_steps, kl_from_ref, 'darkred', linewidth=2)
    axes[6].set_xlabel('Training Step')
    axes[6].set_ylabel('KL Divergence')
    axes[6].set_title(f'KL Divergence from Reference Model (π_ref){title_suffix}')
    axes[6].grid(True, alpha=0.3)
    mean_kl = np.mean(kl_from_ref)
    axes[6].axhline(y=mean_kl, color='gray', linestyle='--', alpha=0.5, label=f'Mean: {mean_kl:.4f}')
    axes[6].legend(fontsize=8)
    
    # Plot 8: Reward Variance
    axes[7].plot(training_steps, reward_variance, 'magenta', linewidth=2)
    axes[7].set_xlabel('Training Step')
    axes[7].set_ylabel('Reward Variance')
    axes[7].set_title(f'Reward Variance Within Trajectory{title_suffix}')
    axes[7].grid(True, alpha=0.3)
    
    # Plot 9: Log Probabilities by Step Index (Early/Mid/Late training periods)
    if step_log_probs and len(step_log_probs) > 0:
        max_steps = max(len(episode_steps) for episode_steps in step_log_probs if episode_steps)
        if max_steps > 0:
            step_indices = list(range(max_steps))
            
            def compute_avg_for_period(start_episode, end_episode):
                """Compute average log probs by step index for a period."""
                relevant_episodes = step_log_probs[start_episode:end_episode]
                step_averages = []
                for step_idx in range(max_steps):
                    step_values = []
                    for episode_steps in relevant_episodes:
                        if step_idx < len(episode_steps):
                            step_values.append(episode_steps[step_idx])
                    step_averages.append(np.mean(step_values) if step_values else 0.0)
                return step_averages
            
            total_episodes = len(step_log_probs)
            first_third_end = total_episodes // 3
            second_third_end = 2 * total_episodes // 3
            
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
            axes[8].set_title(f'Log Prob by Step Index (training progression){title_suffix}')
            axes[8].grid(True, alpha=0.3)
            axes[8].set_xticks(step_indices)
        else:
            axes[8].text(0.5, 0.5, 'No step log prob data\navailable yet', 
                         ha='center', va='center', transform=axes[8].transAxes, fontsize=10)
            axes[8].set_title('Log Prob by Step Index')
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
    axes[9].set_title(f'PPO Clipping Ratio (π_θ / π_old){title_suffix}')
    axes[9].legend(fontsize=7)
    axes[9].grid(True, alpha=0.3)
    axes[9].set_ylim(0.5, 1.5)
    
    # Plot 11: LoRA Layer Gradient Flow
    if lora_layer_gradients:
        for layer_idx, grad_history in lora_layer_gradients.items():
            # Ensure gradient history matches training_steps length
            grad_data = grad_history[:min_length] + [0.0] * max(0, min_length - len(grad_history))
            # Apply smoothing to gradient data
            grad_data_smoothed = smooth_data(grad_data, smooth_window)
            color = 'red' if layer_idx == max(lora_layer_gradients.keys()) else f'C{layer_idx % 10}'
            linewidth = 3 if layer_idx == max(lora_layer_gradients.keys()) else 1.5
            alpha = 1.0 if layer_idx == max(lora_layer_gradients.keys()) else 0.7
            label = f'Layer {layer_idx}' + (' (Query Layer)' if layer_idx == max(lora_layer_gradients.keys()) else '')
            axes[10].plot(training_steps, grad_data_smoothed, color=color, linewidth=linewidth, alpha=alpha, label=label)
        axes[10].legend(fontsize=6, loc='upper right')
    else:
        # Show message when no LoRA gradient data is available
        axes[10].text(0.5, 0.5, 'No LoRA gradient data available', 
                     transform=axes[10].transAxes, ha='center', va='center',
                     fontsize=12, style='italic', alpha=0.6)
    axes[10].set_xlabel('Training Step')
    axes[10].set_ylabel('Gradient Magnitude')
    axes[10].set_title(f'LoRA Layer Gradient Flow{title_suffix}')
    axes[10].set_yscale('log')
    axes[10].grid(True, alpha=0.3)
    
    # Plot 12: Advantage Distribution Analysis
    positive_percentages = smooth_data([d['positive_percentage'] for d in advantage_distributions], smooth_window)
    negative_percentages = smooth_data([d['negative_percentage'] for d in advantage_distributions], smooth_window)
    
    axes[11].plot(training_steps, positive_percentages, 'green', linewidth=2, label='Positive Advantages')
    axes[11].plot(training_steps, negative_percentages, 'red', linewidth=2, label='Negative Advantages')
    axes[11].axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='50% (Balanced)')
    axes[11].set_xlabel('Training Step')
    axes[11].set_ylabel('Percentage (%)')
    axes[11].set_title(f'Advantage Distribution (Step-Level Learning Signal){title_suffix}')
    axes[11].set_ylim(0, 100)
    axes[11].legend(fontsize=7)
    axes[11].grid(True, alpha=0.3)
    
    # Add baseline update markers
    BASELINE_UPDATE_FREQUENCY = config.get('BASELINE_UPDATE_FREQUENCY', 10)
    for episode in range(BASELINE_UPDATE_FREQUENCY, max(training_steps) + 1, BASELINE_UPDATE_FREQUENCY):
        if episode <= max(training_steps):
            axes[11].axvline(x=episode, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the plot (always use the same filename, whether smoothed or not)
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
        title = 'Loss Composition Over Time'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        plt.title(title)
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
        
        # Extract similarity metrics and apply smoothing
        similarity_means = smooth_data([s['mean'] for s in similarity_score_stats], smooth_window)
        similarity_stds = smooth_data([s['std'] for s in similarity_score_stats], smooth_window)
        similarity_entropies = smooth_data([s['entropy'] for s in similarity_score_stats], smooth_window)
        similarity_maxs = smooth_data([s['max'] for s in similarity_score_stats], smooth_window)
        similarity_mins = smooth_data([s['min'] for s in similarity_score_stats], smooth_window)
        
        # Create 2x2 subplot for detailed similarity analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Subplot 1: Mean similarity scores
        ax1.plot(training_steps, similarity_means, 'blue', linewidth=2, label='Mean Similarity')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Similarity Score')
        title = 'Mean Query-Key Similarity Evolution'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Similarity entropy (measure of specificity)
        ax2.plot(training_steps, similarity_entropies, 'green', linewidth=2, label='Similarity Entropy')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Entropy (nats)')
        title = 'Query Specificity (Lower = More Specific)'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax2.set_title(title)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Similarity range (max - min)
        similarity_ranges = [mx - mn for mx, mn in zip(similarity_maxs, similarity_mins)]
        ax3.plot(training_steps, similarity_ranges, 'orange', linewidth=2, label='Range (Max - Min)')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Similarity Range')
        title = 'Query Discrimination Ability'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax3.set_title(title)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Standard deviation
        ax4.plot(training_steps, similarity_stds, 'red', linewidth=2, label='Similarity Std Dev')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Standard Deviation')
        title = 'Similarity Score Variability'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax4.set_title(title)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add baseline update markers to all subplots
        BASELINE_UPDATE_FREQUENCY = config.get('BASELINE_UPDATE_FREQUENCY', 10)
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
        
        # Extract step-level learning metrics over time and apply smoothing
        positive_advantages = smooth_data([d['positive_percentage'] for d in advantage_distributions], smooth_window)
        advantage_means = smooth_data([d['mean'] for d in advantage_distributions], smooth_window)
        advantage_stds = smooth_data([d['std'] for d in advantage_distributions], smooth_window)
        
        # Create 2x2 subplot for step-level analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Subplot 1: Learning signal strength (positive advantage percentage)
        ax1.plot(training_steps, positive_advantages, 'green', linewidth=2, label='Positive Advantage %')
        ax1.axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='50% (Balanced)')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Percentage (%)')
        title = 'Step-Level Learning Signal Strength'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax1.set_title(title)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Advantage magnitude evolution
        ax2.plot(training_steps, advantage_means, 'blue', linewidth=2, label='Mean Advantage')
        ax2.axhline(y=0.0, color='gray', linestyle='-', alpha=0.5, label='Zero Advantage')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Advantage')
        title = 'Advantage Magnitude Over Time'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax2.set_title(title)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Subplot 3: Advantage variability (indicates learning consistency)
        ax3.plot(training_steps, advantage_stds, 'orange', linewidth=2, label='Advantage Std Dev')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Standard Deviation')
        title = 'Learning Signal Variability'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax3.set_title(title)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot 4: Combined learning health score
        # Normalize metrics to 0-1 scale for combination
        norm_positive = np.array(positive_advantages) / 100.0
        norm_mean_adv = np.array(advantage_means)
        norm_mean_adv = (norm_mean_adv - np.min(norm_mean_adv)) / (np.max(norm_mean_adv) - np.min(norm_mean_adv) + 1e-8)
        
        # Health score: high positive percentage + positive trend in advantages
        health_score = 0.6 * norm_positive + 0.4 * norm_mean_adv
        ax4.plot(training_steps, health_score * 100, 'purple', linewidth=2, label='Learning Health Score')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Health Score (%)')
        title = 'Overall Step-Level Learning Health'
        if smooth_window > 1:
            title += f' (smoothed, window={smooth_window})'
        ax4.set_title(title)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add baseline update markers to all subplots
        BASELINE_UPDATE_FREQUENCY = config.get('BASELINE_UPDATE_FREQUENCY', 10)
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
    parser.add_argument('data_file', nargs='?', help='Path to the pickle file containing plot data')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Directory to save plots (default: same directory as input file)')
    parser.add_argument('--kl-coef', type=float, default=None,
                       help='Override KL penalty coefficient for labels')
    parser.add_argument('--smooth-window', '-s', type=int, default=1,
                       help='Smoothing window size (default: 1, no smoothing)')
    parser.add_argument('--recent', type=int, metavar='N',
                       help='Select the nth most recent log folder (0=most recent, 1=second most recent, etc.)')
    parser.add_argument('--logs-dir', default='logs',
                       help='Base logs directory (default: logs)')
    
    args = parser.parse_args()
    
    # Validate smoothing window
    if args.smooth_window < 1:
        print("Error: Smoothing window size must be at least 1")
        return
    
    # Determine data file path
    if args.recent is not None:
        # Use --recent flag to find log folder
        log_folder = find_nth_most_recent_log_folder(args.recent, args.logs_dir)
        if log_folder is None:
            return
        data_file = os.path.join(log_folder, "plots", "plot_data.pkl")
        
        # If no output directory specified, use the plots directory of the selected folder
        if args.output_dir is None:
            args.output_dir = os.path.join(log_folder, "plots")
    else:
        # Use explicit data_file argument
        if args.data_file is None:
            print("Error: Must specify either data_file argument or --recent flag")
            parser.print_help()
            return
        data_file = args.data_file
        
        # If no output directory specified, use the directory of the input file
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(data_file))
    
    # Load data
    print(f"Loading data from: {data_file}")
    data = load_plot_data(data_file)
    
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
    generate_plots(data, args.output_dir, custom_config, args.smooth_window)


if __name__ == "__main__":
    main() 