
"""
Text-based analysis generation from plotting data for LM consumption.

This module generates structured text summaries of training metrics that are
optimized for language model analysis rather than visual plotting.
"""

import pickle
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import os


def compute_trend(values: List[float], window: int = 50) -> Dict[str, float]:
    """
    Compute trend statistics for a metric over recent episodes.
    
    Args:
        values: List of metric values
        window: Number of recent episodes to analyze
        
    Returns:
        Dict with trend statistics
    """
    if len(values) < 2:
        return {"slope": 0.0, "r_squared": 0.0, "recent_mean": 0.0, "overall_mean": 0.0}
    
    # Use recent window or all data if less available
    recent_values = values[-window:] if len(values) >= window else values
    x = np.arange(len(recent_values))
    y = np.array(recent_values)
    
    # Linear regression
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
    else:
        slope = 0.0
        r_squared = 0.0
    
    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "recent_mean": float(np.mean(recent_values)),
        "overall_mean": float(np.mean(values)),
        "recent_std": float(np.std(recent_values)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def analyze_advantage_distribution(advantage_distributions: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze advantage distribution patterns over training."""
    if not advantage_distributions:
        return {}
    
    # Extract time series
    positive_pcts = [d['positive_percentage'] for d in advantage_distributions]
    negative_pcts = [d['negative_percentage'] for d in advantage_distributions]
    means = [d['mean'] for d in advantage_distributions]
    stds = [d['std'] for d in advantage_distributions]
    
    return {
        "positive_percentage": compute_trend(positive_pcts),
        "negative_percentage": compute_trend(negative_pcts),
        "advantage_means": compute_trend(means),
        "advantage_stds": compute_trend(stds),
        "current_positive_pct": positive_pcts[-1] if positive_pcts else 0.0,
        "learning_signal_strength": "strong" if positive_pcts[-1] > 60 else "weak" if positive_pcts[-1] < 40 else "moderate",
        "learning_consistency": "high" if stds[-1] < np.mean(stds) * 0.8 else "low"
    }


def analyze_similarity_scores(similarity_stats: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze query-key similarity evolution."""
    if not similarity_stats:
        return {}
    
    means = [s['mean'] for s in similarity_stats]
    stds = [s['std'] for s in similarity_stats]
    entropies = [s['entropy'] for s in similarity_stats]
    ranges = [s['max'] - s['min'] for s in similarity_stats]
    
    return {
        "similarity_means": compute_trend(means),
        "similarity_stds": compute_trend(stds),
        "similarity_entropies": compute_trend(entropies),
        "similarity_ranges": compute_trend(ranges),
        "discrimination_ability": "high" if ranges[-1] > np.mean(ranges) * 1.2 else "low",
        "query_specificity": "high" if entropies[-1] > np.mean(entropies) * 1.1 else "low"
    }


def analyze_learning_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute overall learning health metrics."""
    
    # Core metrics
    rewards = data.get('avg_rewards', [])
    advantages = data.get('avg_advantages', [])
    policy_losses = data.get('policy_losses', [])
    kl_losses = data.get('kl_losses', [])
    gradient_mags = data.get('gradient_magnitudes', [])
    
    # Model performance
    adapter_logprobs = data.get('adapter_log_probs', [])
    baseline_logprobs = data.get('baseline_log_probs', [])
    
    health_metrics = {}
    
    if rewards:
        reward_trend = compute_trend(rewards)
        health_metrics["reward_health"] = {
            "trend": reward_trend,
            "improving": reward_trend["slope"] > 0.001,
            "recent_performance": "good" if reward_trend["recent_mean"] > reward_trend["overall_mean"] else "poor"
        }
    
    if advantages:
        adv_trend = compute_trend(advantages)
        health_metrics["advantage_health"] = {
            "trend": adv_trend,
            "positive": adv_trend["recent_mean"] > 0,
            "improving": adv_trend["slope"] > 0
        }
    
    if policy_losses:
        loss_trend = compute_trend(policy_losses)
        health_metrics["loss_health"] = {
            "trend": loss_trend,
            "decreasing": loss_trend["slope"] < 0,
            "stable": abs(loss_trend["slope"]) < 0.01
        }
    
    if gradient_mags:
        grad_trend = compute_trend(gradient_mags)
        health_metrics["gradient_health"] = {
            "trend": grad_trend,
            "magnitude": "healthy" if 1e-5 < grad_trend["recent_mean"] < 1e-1 else "concerning",
            "stable": grad_trend["recent_std"] < grad_trend["recent_mean"]
        }
    
    # Compute overall health score (0-100)
    score_components = []
    if "reward_health" in health_metrics and health_metrics["reward_health"]["improving"]:
        score_components.append(25)
    if "advantage_health" in health_metrics and health_metrics["advantage_health"]["positive"]:
        score_components.append(25)
    if "loss_health" in health_metrics and (health_metrics["loss_health"]["decreasing"] or health_metrics["loss_health"]["stable"]):
        score_components.append(25)
    if "gradient_health" in health_metrics and health_metrics["gradient_health"]["magnitude"] == "healthy":
        score_components.append(25)
    
    health_metrics["overall_health_score"] = sum(score_components)
    health_metrics["health_status"] = (
        "excellent" if health_metrics["overall_health_score"] >= 75 else
        "good" if health_metrics["overall_health_score"] >= 50 else
        "concerning" if health_metrics["overall_health_score"] >= 25 else
        "poor"
    )
    
    return health_metrics


def detect_training_phases(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect distinct training phases based on metric patterns."""
    training_steps = data.get('training_steps', [])
    if len(training_steps) < 100:  # Need sufficient data
        return []
    
    # Look for baseline update points
    baseline_update_freq = data.get('metadata', {}).get('config', {}).get('BASELINE_UPDATE_FREQUENCY', 10)
    update_points = [i for i, step in enumerate(training_steps) if step % baseline_update_freq == 0]
    
    phases = []
    for i, update_idx in enumerate(update_points[:-1]):
        start_idx = update_idx
        end_idx = update_points[i + 1] if i + 1 < len(update_points) else len(training_steps)
        
        # Analyze this phase
        phase_rewards = data['avg_rewards'][start_idx:end_idx] if 'avg_rewards' in data else []
        phase_advantages = data['avg_advantages'][start_idx:end_idx] if 'avg_advantages' in data else []
        
        if phase_rewards and phase_advantages:
            phase_data = {
                "phase_number": i + 1,
                "episodes": (training_steps[start_idx], training_steps[end_idx - 1]),
                "reward_stats": {
                    "mean": np.mean(phase_rewards),
                    "trend": np.polyfit(range(len(phase_rewards)), phase_rewards, 1)[0] if len(phase_rewards) > 1 else 0
                },
                "advantage_stats": {
                    "mean": np.mean(phase_advantages),
                    "positive_ratio": np.mean([1 if a > 0 else 0 for a in phase_advantages])
                }
            }
            phases.append(phase_data)
    
    return phases


def generate_text_summary(data: Dict[str, Any]) -> str:
    """Generate a comprehensive text summary of training progress."""
    
    # Extract basic info
    training_steps = data.get('training_steps', [])
    metadata = data.get('metadata', {})
    config = metadata.get('config', {})
    
    if not training_steps:
        return "No training data available for analysis."
    
    total_episodes = len(training_steps)
    latest_episode = training_steps[-1] if training_steps else 0
    
    # Analyze different aspects
    advantage_analysis = analyze_advantage_distribution(data.get('advantage_distributions', []))
    similarity_analysis = analyze_similarity_scores(data.get('similarity_score_stats', []))
    health_analysis = analyze_learning_health(data)
    training_phases = detect_training_phases(data)
    
    # Build summary
    summary = f"""
TRAINING ANALYSIS REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Run: {metadata.get('timestamp', 'Unknown')}
Episodes Analyzed: {total_episodes} (up to episode {latest_episode})

LEARNING HEALTH OVERVIEW
-----------------------
Overall Status: {health_analysis.get('health_status', 'unknown').upper()}
Health Score: {health_analysis.get('overall_health_score', 0)}/100

Key Metrics:
"""
    
    # Add reward analysis
    if 'reward_health' in health_analysis:
        rh = health_analysis['reward_health']
        trend_dir = "↗" if rh['trend']['slope'] > 0.001 else "↘" if rh['trend']['slope'] < -0.001 else "→"
        summary += f"• Rewards: {rh['recent_performance']} performance, trending {trend_dir} (slope: {rh['trend']['slope']:.4f})\n"
        summary += f"  Current avg: {rh['trend']['recent_mean']:.4f}, Overall avg: {rh['trend']['overall_mean']:.4f}\n"
    
    # Add advantage analysis
    if advantage_analysis:
        summary += f"• Learning Signal: {advantage_analysis['learning_signal_strength']} ({advantage_analysis['current_positive_pct']:.1f}% positive advantages)\n"
        summary += f"  Consistency: {advantage_analysis['learning_consistency']} (std: {advantage_analysis['advantage_stds']['recent_mean']:.4f})\n"
    
    # Add gradient health
    if 'gradient_health' in health_analysis:
        gh = health_analysis['gradient_health']
        summary += f"• Gradients: {gh['magnitude']} magnitude ({gh['trend']['recent_mean']:.2e})\n"
    
    # Step-level learning analysis
    if advantage_analysis:
        summary += f"""
STEP-LEVEL LEARNING ANALYSIS
----------------------------
Positive Advantage Trend: slope={advantage_analysis['positive_percentage']['slope']:.2f}%, R²={advantage_analysis['positive_percentage']['r_squared']:.3f}
Recent Performance: {advantage_analysis['current_positive_pct']:.1f}% steps above average
Learning Signal Strength: {advantage_analysis['learning_signal_strength'].upper()}

Advantage Distribution Evolution:
• Mean advantage: {advantage_analysis['advantage_means']['recent_mean']:.4f} (trend: {advantage_analysis['advantage_means']['slope']:.4f})
• Variability: {advantage_analysis['advantage_stds']['recent_mean']:.4f} (consistency: {advantage_analysis['learning_consistency']})
"""
    
    # Query-key similarity analysis
    if similarity_analysis:
        summary += f"""
QUERY-KEY SIMILARITY ANALYSIS  
-----------------------------
Discrimination Ability: {similarity_analysis['discrimination_ability'].upper()}
Query Specificity: {similarity_analysis['query_specificity'].upper()}

Recent Similarity Stats:
• Mean similarity: {similarity_analysis['similarity_means']['recent_mean']:.4f}
• Similarity range: {similarity_analysis['similarity_ranges']['recent_mean']:.4f}
• Entropy: {similarity_analysis['similarity_entropies']['recent_mean']:.4f}
"""
    
    # Training phases
    if training_phases:
        summary += f"""
TRAINING PHASE ANALYSIS
----------------------
Detected {len(training_phases)} training phases:
"""
        for phase in training_phases[-3:]:  # Show last 3 phases
            summary += f"Phase {phase['phase_number']} (episodes {phase['episodes'][0]}-{phase['episodes'][1]}):\n"
            summary += f"  • Avg reward: {phase['reward_stats']['mean']:.4f}, trend: {phase['reward_stats']['trend']:.4f}\n"
            summary += f"  • Positive advantages: {phase['advantage_stats']['positive_ratio']:.1%}\n"
    
    # Configuration highlights
    summary += f"""
CONFIGURATION HIGHLIGHTS
-----------------------
• KL Penalty Coefficient: {config.get('KL_PENALTY_COEFFICIENT', 'unknown')}
• Gamma (discount): {config.get('CONFIG.gamma', 'unknown')}
• Temperature: {config.get('CONFIG.temperature', 'unknown')}

DATA SUMMARY
-----------
• Total training episodes: {total_episodes}
• Latest episode: {latest_episode}
• Advantage distribution samples: {len(data.get('advantage_distributions', []))}
• Similarity score samples: {len(data.get('similarity_score_stats', []))}
"""
    
    return summary


def generate_json_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured JSON analysis for programmatic consumption."""
    
    training_steps = data.get('training_steps', [])
    if not training_steps:
        return {"error": "No training data available"}
    
    return {
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "training_timestamp": data.get('metadata', {}).get('timestamp'),
            "total_episodes": len(training_steps),
            "latest_episode": training_steps[-1] if training_steps else 0
        },
        "learning_health": analyze_learning_health(data),
        "advantage_analysis": analyze_advantage_distribution(data.get('advantage_distributions', [])),
        "similarity_analysis": analyze_similarity_scores(data.get('similarity_score_stats', [])),
        "training_phases": detect_training_phases(data),
        "recent_metrics": {
            "avg_reward": data['avg_rewards'][-1] if data.get('avg_rewards') else None,
            "avg_advantage": data['avg_advantages'][-1] if data.get('avg_advantages') else None,
            "policy_loss": data['policy_losses'][-1] if data.get('policy_losses') else None,
            "kl_loss": data['kl_losses'][-1] if data.get('kl_losses') else None,
            "gradient_magnitude": data['gradient_magnitudes'][-1] if data.get('gradient_magnitudes') else None
        }
    }


def save_text_analysis(plot_data_path: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Load plot data and save both text and JSON analyses.
    
    Args:
        plot_data_path: Path to the plot_data.pkl file
        output_dir: Directory to save analysis files (default: same as pkl file)
        
    Returns:
        Tuple of (text_analysis_path, json_analysis_path)
    """
    # Load data
    with open(plot_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(plot_data_path)
    
    # Generate analyses
    text_summary = generate_text_summary(data)
    json_analysis = generate_json_analysis(data)
    
    # Save files
    base_name = os.path.splitext(os.path.basename(plot_data_path))[0]
    
    text_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
    json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
    
    with open(text_path, 'w') as f:
        f.write(text_summary)
    
    with open(json_path, 'w') as f:
        json.dump(json_analysis, f, indent=2)
    
    return text_path, json_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python text_analysis.py <plot_data.pkl>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    text_path, json_path = save_text_analysis(pkl_path)
    print(f"Generated text analysis: {text_path}")
    print(f"Generated JSON analysis: {json_path}") 