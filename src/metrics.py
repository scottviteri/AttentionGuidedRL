from collections import Counter
from typing import Dict, List
import math
import torch


def compute_wikipedia_order_consistency(trajectory) -> float:
    """
    Compute how consistently the model selects keys in their original order.
    Returns a score in [0.0, 1.0].
    """
    if not trajectory.qkv_steps:
        raise ValueError("Trajectory must contain qkv_steps")

    first_step = trajectory.qkv_steps[0]
    if not isinstance(first_step.selected_idx, torch.Tensor):
        raise TypeError("selected_idx must be a torch.Tensor")
    if first_step.selected_idx.numel() == 0:
        raise ValueError("selected_idx tensor is empty")

    batch_size = trajectory.qkv_steps[0].selected_idx.shape[0]
    all_batch_consistency_scores: List[float] = []

    def edit_distance(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    for b in range(batch_size):
        selected_indices_for_batch_item: List[int] = []
        for step in trajectory.qkv_steps:
            if hasattr(step, 'selected_idx') and isinstance(step.selected_idx, torch.Tensor):
                selected_indices_for_batch_item.append(step.selected_idx[b].item())

        if len(selected_indices_for_batch_item) < 2:
            all_batch_consistency_scores.append(0.5)
            continue

        n = len(selected_indices_for_batch_item)
        perfect_sequence = list(range(n))
        distance = edit_distance(selected_indices_for_batch_item, perfect_sequence)
        max_distance = n
        if max_distance == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (distance / max_distance)
        all_batch_consistency_scores.append(max(0.0, min(1.0, consistency_score)))

    return sum(all_batch_consistency_scores) / len(all_batch_consistency_scores) if all_batch_consistency_scores else 0.5


def compute_batch_selection_entropy(trajectory) -> float:
    """
    Compute the entropy of key selection orders within a batch.
    Returns a normalized entropy in [0.0, 1.0].
    """
    if not trajectory.qkv_steps:
        raise ValueError("Trajectory must contain qkv_steps")

    if not isinstance(trajectory.qkv_steps[0].selected_idx, torch.Tensor):
        raise TypeError("selected_idx must be a torch.Tensor")

    if trajectory.qkv_steps[0].selected_idx.numel() == 0:
        raise ValueError("selected_idx tensor is empty")

    batch_size = trajectory.qkv_steps[0].selected_idx.shape[0]
    if batch_size <= 1:
        return 0.0

    all_batch_sequences: List[tuple] = []
    for b in range(batch_size):
        sequence: List[int] = []
        for step in trajectory.qkv_steps:
            if hasattr(step, 'selected_idx') and isinstance(step.selected_idx, torch.Tensor):
                sequence.append(step.selected_idx[b].item())
        all_batch_sequences.append(tuple(sequence))

    if not all_batch_sequences:
        return 0.0

    sequence_counts = Counter(all_batch_sequences)
    total_sequences = len(all_batch_sequences)
    entropy_val = 0.0
    for count in sequence_counts.values():
        if count > 0:
            p = count / total_sequences
            entropy_val -= p * math.log2(p)

    max_entropy = math.log2(batch_size)
    normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0
    return normalized_entropy


def compute_advantage_distribution(advantages: torch.Tensor) -> Dict[str, float]:
    """
    Distribution of positive/negative/zero advantages and basic stats.
    """
    total_advantages = advantages.numel()
    positive_count = (advantages > 0).sum().item()
    negative_count = (advantages < 0).sum().item()
    zero_count = (advantages == 0).sum().item()

    return {
        'positive_percentage': positive_count / total_advantages * 100 if total_advantages else 0.0,
        'negative_percentage': negative_count / total_advantages * 100 if total_advantages else 0.0,
        'zero_percentage': zero_count / total_advantages * 100 if total_advantages else 0.0,
        'mean': advantages.mean().item() if total_advantages else 0.0,
        'std': advantages.std().item() if total_advantages else 0.0,
    }


def compute_similarity_score_stats(trajectory) -> Dict[str, float]:
    """
    Aggregate similarity score statistics across trajectory steps.
    """
    import torch
    all_similarities = []
    for step in trajectory.qkv_steps:
        if hasattr(step, 'similarity_scores') and step.similarity_scores is not None:
            similarities = step.similarity_scores
            probs = torch.exp(similarities)
            all_similarities.append({
                'mean': probs.mean().item(),
                'std': probs.std().item(),
                'entropy': -(probs * similarities).sum(dim=-1).mean().item(),
                'max': probs.max().item(),
                'min': probs.min().item()
            })

    if not all_similarities:
        return {'mean': 0.0, 'std': 0.0, 'entropy': 0.0, 'max': 0.0, 'min': 0.0}

    return {
        'mean': sum(s['mean'] for s in all_similarities) / len(all_similarities),
        'std': sum(s['std'] for s in all_similarities) / len(all_similarities),
        'entropy': sum(s['entropy'] for s in all_similarities) / len(all_similarities),
        'max': sum(s['max'] for s in all_similarities) / len(all_similarities),
        'min': sum(s['min'] for s in all_similarities) / len(all_similarities),
    }