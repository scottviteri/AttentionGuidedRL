
"""
Plotting data structures and utilities for the Attention-Guided RL project.

This module provides a clean, type-safe way to collect and manage all plotting metrics
without relying on global variables.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch
from datetime import datetime


@dataclass(frozen=True)
class PlotData:
    """
    Frozen dataclass containing all metrics for plotting and logging.
    
    This replaces the previous global variable approach with a clean,
    type-safe data structure.
    """
    # Core training metrics
    training_steps: List[int] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    kl_losses: List[float] = field(default_factory=list)
    avg_rewards: List[float] = field(default_factory=list)
    
    # Model log probabilities
    adapter_log_probs: List[float] = field(default_factory=list)
    baseline_log_probs: List[float] = field(default_factory=list)
    base_log_probs: List[float] = field(default_factory=list)  # Reference model
    
    # Advanced training metrics
    avg_advantages: List[float] = field(default_factory=list)
    trajectory_log_probs: List[float] = field(default_factory=list)
    wikipedia_order_consistency: List[float] = field(default_factory=list)
    entropy_values: List[float] = field(default_factory=list)
    kl_penalty_terms: List[float] = field(default_factory=list)
    reward_variance: List[float] = field(default_factory=list)
    gradient_magnitudes: List[float] = field(default_factory=list)
    step_log_probs: List[List[float]] = field(default_factory=list)  # List of lists
    clipping_ratios: List[float] = field(default_factory=list)
    batch_selection_entropy: List[float] = field(default_factory=list)
    trajectory_samples: List[Dict[str, Any]] = field(default_factory=list)
    kl_from_ref: List[float] = field(default_factory=list)
    kl_keys_from_ref: List[float] = field(default_factory=list)
    kl_values_from_ref: List[float] = field(default_factory=list)
    policy_gradients: List[float] = field(default_factory=list)  # Policy gradients (before negation)
    
    # Enhanced debugging metrics
    lora_layer_gradients: Dict[int, List[float]] = field(default_factory=dict)
    advantage_distributions: List[Dict[str, float]] = field(default_factory=list)
    similarity_score_stats: List[Dict[str, float]] = field(default_factory=list)
    
    # Chain rule specific metrics
    policy_term_values: List[float] = field(default_factory=list)
    reward_term_values: List[float] = field(default_factory=list)
    total_returns_mean: List[float] = field(default_factory=list)
    total_returns_std: List[float] = field(default_factory=list)
    policy_term_variance: List[float] = field(default_factory=list)
    reward_term_variance: List[float] = field(default_factory=list)
    reward_gradient_norm: List[float] = field(default_factory=list)
    policy_reward_ratio: List[float] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_episode_data(
        self,
        episode: int,
        total_loss: float,
        policy_loss: float,
        kl_loss: float,
        avg_reward: float,
        adapter_log_prob: float,
        baseline_log_prob: float,
        base_log_prob: float,
        avg_advantage: float,
        trajectory_log_prob: float,
        wikipedia_order_consistency: float,
        entropy_value: float,
        kl_penalty_term: float,
        reward_variance: float,
        gradient_magnitude: float,
        step_log_probs_episode: List[float],
        clipping_ratio: float,
        batch_selection_entropy: float,
        kl_from_ref_value: float,
        kl_keys_from_ref_value: float = 0.0,
        kl_values_from_ref_value: float = 0.0,
        lora_layer_gradients_episode: Optional[Dict[int, float]] = None,
        advantage_distribution: Optional[Dict[str, float]] = None,
        similarity_score_stats: Optional[Dict[str, float]] = None,
        policy_gradient: float = 0.0,
        # New chain rule metrics
        policy_term_value: float = 0.0,
        reward_term_value: float = 0.0,
        total_returns_mean_val: float = 0.0,
        total_returns_std_val: float = 0.0,
        policy_term_variance_val: float = 0.0,
        reward_term_variance_val: float = 0.0,
        reward_gradient_norm_val: float = 0.0,
        policy_reward_ratio_val: float = 0.0,
        trajectory_sample: Optional[Dict[str, Any]] = None
    ) -> 'PlotData':
        """
        Add data for a single episode and return a new PlotData instance.
        
        This method follows the pure functional approach preferred by the user,
        taking the current frozen instance and returning a new updated instance.
        """
        # Provide defaults for optional dicts
        if advantage_distribution is None:
            advantage_distribution = {
                'positive_percentage': 0.0,
                'negative_percentage': 0.0,
                'zero_percentage': 100.0,
                'mean': 0.0,
                'std': 0.0,
            }
        if similarity_score_stats is None:
            similarity_score_stats = {'mean': 0.0, 'std': 0.0, 'entropy': 0.0, 'max': 0.0, 'min': 0.0}
        if lora_layer_gradients_episode is None:
            lora_layer_gradients_episode = {}

        # Create new lists by copying existing data and appending new values
        new_training_steps = self.training_steps + [episode]
        new_total_losses = self.total_losses + [total_loss]
        new_policy_losses = self.policy_losses + [policy_loss]
        new_kl_losses = self.kl_losses + [kl_loss]
        new_avg_rewards = self.avg_rewards + [avg_reward]
        new_adapter_log_probs = self.adapter_log_probs + [adapter_log_prob]
        new_baseline_log_probs = self.baseline_log_probs + [baseline_log_prob]
        new_base_log_probs = self.base_log_probs + [base_log_prob]
        new_avg_advantages = self.avg_advantages + [avg_advantage]
        new_trajectory_log_probs = self.trajectory_log_probs + [trajectory_log_prob]
        new_wikipedia_order_consistency = self.wikipedia_order_consistency + [wikipedia_order_consistency]
        new_entropy_values = self.entropy_values + [entropy_value]
        new_kl_penalty_terms = self.kl_penalty_terms + [kl_penalty_term]
        new_reward_variance = self.reward_variance + [reward_variance]
        new_gradient_magnitudes = self.gradient_magnitudes + [gradient_magnitude]
        new_step_log_probs = self.step_log_probs + [step_log_probs_episode]
        new_clipping_ratios = self.clipping_ratios + [clipping_ratio]
        new_batch_selection_entropy = self.batch_selection_entropy + [batch_selection_entropy]
        new_kl_from_ref = self.kl_from_ref + [kl_from_ref_value]
        new_kl_keys_from_ref = self.kl_keys_from_ref + [kl_keys_from_ref_value]
        new_kl_values_from_ref = self.kl_values_from_ref + [kl_values_from_ref_value]
        new_policy_gradients = self.policy_gradients + [policy_gradient]
        new_advantage_distributions = self.advantage_distributions + [advantage_distribution]
        new_similarity_score_stats = self.similarity_score_stats + [similarity_score_stats]
        
        # Add new chain rule metrics
        new_policy_term_values = self.policy_term_values + [policy_term_value]
        new_reward_term_values = self.reward_term_values + [reward_term_value]
        new_total_returns_mean = self.total_returns_mean + [total_returns_mean_val]
        new_total_returns_std = self.total_returns_std + [total_returns_std_val]
        new_policy_term_variance = self.policy_term_variance + [policy_term_variance_val]
        new_reward_term_variance = self.reward_term_variance + [reward_term_variance_val]
        new_reward_gradient_norm = self.reward_gradient_norm + [reward_gradient_norm_val]
        new_policy_reward_ratio = self.policy_reward_ratio + [policy_reward_ratio_val]
        
        # Handle trajectory samples (optional)
        new_trajectory_samples = self.trajectory_samples.copy()
        if trajectory_sample is not None:
            new_trajectory_samples.append(trajectory_sample)
        
        # Update LoRA layer gradients
        new_lora_layer_gradients = {}
        for layer_idx, gradients_list in self.lora_layer_gradients.items():
            new_lora_layer_gradients[layer_idx] = gradients_list.copy()
        
        for layer_idx, gradient_value in lora_layer_gradients_episode.items():
            if layer_idx not in new_lora_layer_gradients:
                new_lora_layer_gradients[layer_idx] = []
            new_lora_layer_gradients[layer_idx].append(gradient_value)
        
        # Return new instance with updated data
        return PlotData(
            training_steps=new_training_steps,
            total_losses=new_total_losses,
            policy_losses=new_policy_losses,
            kl_losses=new_kl_losses,
            avg_rewards=new_avg_rewards,
            adapter_log_probs=new_adapter_log_probs,
            baseline_log_probs=new_baseline_log_probs,
            base_log_probs=new_base_log_probs,
            avg_advantages=new_avg_advantages,
            trajectory_log_probs=new_trajectory_log_probs,
            wikipedia_order_consistency=new_wikipedia_order_consistency,
            entropy_values=new_entropy_values,
            kl_penalty_terms=new_kl_penalty_terms,
            reward_variance=new_reward_variance,
            gradient_magnitudes=new_gradient_magnitudes,
            step_log_probs=new_step_log_probs,
            clipping_ratios=new_clipping_ratios,
            batch_selection_entropy=new_batch_selection_entropy,
            trajectory_samples=new_trajectory_samples,
            kl_from_ref=new_kl_from_ref,
            kl_keys_from_ref=new_kl_keys_from_ref,
            kl_values_from_ref=new_kl_values_from_ref,
            policy_gradients=new_policy_gradients,
            lora_layer_gradients=new_lora_layer_gradients,
            advantage_distributions=new_advantage_distributions,
            similarity_score_stats=new_similarity_score_stats,
            # New chain rule metrics
            policy_term_values=new_policy_term_values,
            reward_term_values=new_reward_term_values,
            total_returns_mean=new_total_returns_mean,
            total_returns_std=new_total_returns_std,
            policy_term_variance=new_policy_term_variance,
            reward_term_variance=new_reward_term_variance,
            reward_gradient_norm=new_reward_gradient_norm,
            policy_reward_ratio=new_policy_reward_ratio,
            metadata=self.metadata  # Metadata is updated separately
        )
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'PlotData':
        """
        Return a new PlotData instance with updated metadata.
        """
        return PlotData(
            training_steps=self.training_steps,
            total_losses=self.total_losses,
            policy_losses=self.policy_losses,
            kl_losses=self.kl_losses,
            avg_rewards=self.avg_rewards,
            adapter_log_probs=self.adapter_log_probs,
            baseline_log_probs=self.baseline_log_probs,
            base_log_probs=self.base_log_probs,
            avg_advantages=self.avg_advantages,
            trajectory_log_probs=self.trajectory_log_probs,
            wikipedia_order_consistency=self.wikipedia_order_consistency,
            entropy_values=self.entropy_values,
            kl_penalty_terms=self.kl_penalty_terms,
            reward_variance=self.reward_variance,
            gradient_magnitudes=self.gradient_magnitudes,
            step_log_probs=self.step_log_probs,
            clipping_ratios=self.clipping_ratios,
            batch_selection_entropy=self.batch_selection_entropy,
            trajectory_samples=self.trajectory_samples,
            kl_from_ref=self.kl_from_ref,
            kl_keys_from_ref=self.kl_keys_from_ref,
            kl_values_from_ref=self.kl_values_from_ref,
            policy_gradients=self.policy_gradients,
            lora_layer_gradients=self.lora_layer_gradients,
            advantage_distributions=self.advantage_distributions,
            similarity_score_stats=self.similarity_score_stats,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with the existing plotting system.
        """
        return {
            'training_steps': self.training_steps,
            'total_losses': self.total_losses,
            'policy_losses': self.policy_losses,
            'kl_losses': self.kl_losses,
            'avg_rewards': self.avg_rewards,
            'adapter_log_probs': self.adapter_log_probs,
            'baseline_log_probs': self.baseline_log_probs,
            'base_log_probs': self.base_log_probs,
            'avg_advantages': self.avg_advantages,
            'trajectory_log_probs': self.trajectory_log_probs,
            'wikipedia_order_consistency': self.wikipedia_order_consistency,
            'entropy_values': self.entropy_values,
            'kl_penalty_terms': self.kl_penalty_terms,
            'reward_variance': self.reward_variance,
            'gradient_magnitudes': self.gradient_magnitudes,
            'step_log_probs': self.step_log_probs,
            'clipping_ratios': self.clipping_ratios,
            'batch_selection_entropy': self.batch_selection_entropy,
            'trajectory_samples': self.trajectory_samples,
            'kl_from_ref': self.kl_from_ref,
            'kl_keys_from_ref': self.kl_keys_from_ref,
            'kl_values_from_ref': self.kl_values_from_ref,
            'policy_gradients': self.policy_gradients,
            'lora_layer_gradients': self.lora_layer_gradients,
            'advantage_distributions': self.advantage_distributions,
            'similarity_score_stats': self.similarity_score_stats,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlotData':
        """
        Create PlotData instance from dictionary (for loading from pickle).
        """
        return cls(
            training_steps=data.get('training_steps', []),
            total_losses=data.get('total_losses', []),
            policy_losses=data.get('policy_losses', []),
            kl_losses=data.get('kl_losses', []),
            avg_rewards=data.get('avg_rewards', []),
            adapter_log_probs=data.get('adapter_log_probs', []),
            baseline_log_probs=data.get('baseline_log_probs', []),
            base_log_probs=data.get('base_log_probs', []),
            avg_advantages=data.get('avg_advantages', []),
            trajectory_log_probs=data.get('trajectory_log_probs', []),
            wikipedia_order_consistency=data.get('wikipedia_order_consistency', []),
            entropy_values=data.get('entropy_values', []),
            kl_penalty_terms=data.get('kl_penalty_terms', []),
            reward_variance=data.get('reward_variance', []),
            gradient_magnitudes=data.get('gradient_magnitudes', []),
            step_log_probs=data.get('step_log_probs', []),
            clipping_ratios=data.get('clipping_ratios', []),
            batch_selection_entropy=data.get('batch_selection_entropy', []),
            trajectory_samples=data.get('trajectory_samples', []),
            kl_from_ref=data.get('kl_from_ref', []),
            kl_keys_from_ref=data.get('kl_keys_from_ref', []),
            kl_values_from_ref=data.get('kl_values_from_ref', []),
            policy_gradients=data.get('policy_gradients', []),
            lora_layer_gradients=data.get('lora_layer_gradients', {}),
            advantage_distributions=data.get('advantage_distributions', []),
            similarity_score_stats=data.get('similarity_score_stats', []),
            metadata=data.get('metadata', {}),
        )


def save_plot_data(plot_data: PlotData, log_dir: str) -> None:
    """
    Save plotting data to a single pickle file using atomic writes to prevent corruption.
    Also generates text-based analysis for LM consumption.
    
    Args:
        plot_data: PlotData instance containing all metrics
        log_dir: Directory where logs are saved
    """
    import pickle
    import os
    import tempfile
    import shutil
    
    # Create plots directory
    plots_dir = f"{log_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use atomic write: write to temp file, then rename to final location
    filename = f"{plots_dir}/plot_data.pkl"
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(mode='wb', dir=plots_dir, delete=False) as tmp_file:
        try:
            plot_data_dict = plot_data.to_dict()
            pickle.dump(plot_data_dict, tmp_file)
            tmp_file.flush()  # Ensure data is written to disk
            os.fsync(tmp_file.fileno())  # Force write to storage
            temp_filename = tmp_file.name
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            raise e
    
    # Atomically move temp file to final location
    try:
        shutil.move(temp_filename, filename)
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_filename)
        except:
            pass
        raise e
    
    # Generate text analysis for LM consumption
    try:
        from src.text_analysis import save_text_analysis
        text_path, json_path = save_text_analysis(filename, plots_dir)
        print(f"Generated text analysis: {os.path.basename(text_path)}")
        print(f"Generated JSON analysis: {os.path.basename(json_path)}")
    except Exception as e:
        print(f"Warning: Failed to generate text analysis: {e}")
        # Don't fail the main saving process if text analysis fails


def load_plot_data(filepath: str) -> PlotData:
    """
    Load plotting data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        PlotData instance
    """
    import pickle
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plot data file not found: {filepath}")
    
    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"Plot data file is empty: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except pickle.UnpicklingError as e:
        # More helpful error message for corrupted files
        raise ValueError(
            f"Plot data file appears to be corrupted: {filepath}\n"
            f"Original error: {e}\n"
            f"File size: {file_size} bytes\n"
            f"You may need to re-run training to regenerate this file."
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to load plot data from {filepath}: {e}"
        ) from e
    
    return PlotData.from_dict(data)


def create_metadata(episode: int, config_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata dictionary for the plotting data.
    
    Args:
        episode: Current episode number
        config_values: Configuration values to include
        
    Returns:
        Metadata dictionary
    """
    return {
        'episode': episode,
        'timestamp': datetime.now().isoformat(),
        'config': config_values
    } 