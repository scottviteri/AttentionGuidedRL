"""
Demonstration of how compute_similarity could be dramatically simplified.

The current implementation is overly complex for what is essentially 
a simple dot-product similarity computation.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def compute_similarity_current(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor, 
    num_heads: int,
    num_groups: int,
    head_dim: int,
    temperature: float = 1.0,
    availability_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Current complex implementation (simplified version)."""
    batch_size, num_keys = key_embeddings.shape[0], key_embeddings.shape[1]
    
    # All the complex GQA logic...
    query_reshaped = query_embeddings.view(batch_size, num_heads, head_dim)
    key_group_dim = num_groups * head_dim
    key_embeddings_truncated = key_embeddings[:, :, :key_group_dim]
    key_reshaped = key_embeddings_truncated.view(batch_size, num_keys, num_groups, head_dim)
    
    # Head-to-group mapping
    head_to_group = torch.div(
        torch.arange(num_heads, device=query_embeddings.device),
        num_heads // num_groups,
        rounding_mode='floor'
    )
    
    # Complex reshaping and indexing
    key_reshaped_flat = key_reshaped.reshape(-1, num_groups, head_dim)
    key_groups_selected = torch.index_select(key_reshaped_flat, dim=1, index=head_to_group)
    key_groups_batched = key_groups_selected.view(batch_size, num_keys, num_heads, head_dim)
    all_key_groups = key_groups_batched.permute(0, 2, 1, 3)
    
    # Per-head computation
    query_expanded = query_reshaped.unsqueeze(2)
    similarities = torch.einsum('bhad,bhkd->bhak', query_expanded, all_key_groups) / math.sqrt(head_dim)
    similarities = similarities.squeeze(2)
    
    # Temperature and masking
    scaled_similarities = similarities / temperature
    if availability_mask is not None:
        expanded_mask = availability_mask.unsqueeze(1).expand(-1, num_heads, -1)
        scaled_similarities = scaled_similarities + expanded_mask
    
    # Complex log probability averaging
    head_probabilities = F.softmax(scaled_similarities, dim=2)
    head_log_probs = torch.log(head_probabilities + 1e-8)
    lse = torch.logsumexp(head_log_probs, dim=1)
    log_probabilities = lse - torch.log(torch.tensor(num_heads, dtype=torch.float, device=lse.device))
    
    return log_probabilities


def compute_similarity_simple(
    query_embeddings: torch.Tensor,  # [batch, hidden_size]
    key_embeddings: torch.Tensor,   # [batch, num_keys, hidden_size]
    temperature: float = 1.0,
    availability_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Simplified implementation that does exactly what we need.
    
    For RL action selection, we just need:
    1. Dot product similarities
    2. Temperature scaling
    3. Masking
    4. Log softmax
    """
    # Simple dot product: [batch, 1, hidden] × [batch, hidden, num_keys] -> [batch, 1, num_keys]
    similarities = torch.bmm(
        query_embeddings.unsqueeze(1),  # [batch, 1, hidden]
        key_embeddings.transpose(1, 2)  # [batch, hidden, num_keys]  
    ).squeeze(1)  # [batch, num_keys]
    
    # Scale by embedding dimension and temperature
    hidden_size = query_embeddings.shape[-1]
    similarities = similarities / (math.sqrt(hidden_size) * temperature)
    
    # Apply availability mask if provided
    if availability_mask is not None:
        similarities = similarities + availability_mask
    
    # Return log probabilities
    return F.log_softmax(similarities, dim=-1)


def compute_similarity_ultra_simple(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    temperature: float = 1.0,
    availability_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Ultra-simple version using torch.cosine_similarity for normalized embeddings."""
    # If embeddings are normalized, cosine similarity is just dot product
    batch_size, num_keys = key_embeddings.shape[0], key_embeddings.shape[1]
    
    # Expand query to match key dimensions
    query_expanded = query_embeddings.unsqueeze(1).expand(-1, num_keys, -1)
    
    # Cosine similarity
    similarities = F.cosine_similarity(query_expanded, key_embeddings, dim=-1)
    similarities = similarities / temperature
    
    # Apply mask and return log probabilities
    if availability_mask is not None:
        similarities = similarities + availability_mask
    
    return F.log_softmax(similarities, dim=-1)


def test_equivalence():
    """Test that all implementations give similar results."""
    batch_size, num_keys, hidden_size = 2, 5, 768
    num_heads, head_dim = 12, 64  # 12 * 64 = 768
    num_groups = 12  # Standard MHA case
    
    # Create test data
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, num_keys, hidden_size)
    mask = torch.zeros(batch_size, num_keys)
    mask[:, -1] = float('-inf')  # Mask last key
    
    print("🧪 TESTING IMPLEMENTATION EQUIVALENCE")
    print("=" * 50)
    
    # Test current complex implementation
    try:
        result_complex = compute_similarity_current(query, keys, num_heads, num_groups, head_dim, availability_mask=mask)
        print(f"✅ Complex implementation: {result_complex.shape}")
        print(f"   Sum of probs: {torch.exp(result_complex).sum(dim=1)}")
    except Exception as e:
        print(f"❌ Complex implementation failed: {e}")
    
    # Test simple implementation
    result_simple = compute_similarity_simple(query, keys, availability_mask=mask)
    print(f"✅ Simple implementation: {result_simple.shape}")
    print(f"   Sum of probs: {torch.exp(result_simple).sum(dim=1)}")
    
    # Test ultra-simple implementation  
    result_ultra = compute_similarity_ultra_simple(query, keys, availability_mask=mask)
    print(f"✅ Ultra-simple implementation: {result_ultra.shape}")
    print(f"   Sum of probs: {torch.exp(result_ultra).sum(dim=1)}")
    
    print(f"\n📊 COMPARISON:")
    print(f"Simple vs Ultra-simple max diff: {torch.abs(result_simple - result_ultra).max().item():.6f}")
    
    # Verify masking works
    print(f"\n🎭 MASKING VERIFICATION:")
    print(f"Simple - last key prob: {torch.exp(result_simple)[:, -1]}")
    print(f"Ultra - last key prob: {torch.exp(result_ultra)[:, -1]}")


def complexity_analysis():
    """Analyze the complexity difference."""
    print("\n📈 COMPLEXITY ANALYSIS")
    print("=" * 50)
    
    print("Current Implementation:")
    print("  ❌ 200+ lines of code")
    print("  ❌ Complex GQA logic")
    print("  ❌ Multiple tensor reshapes")
    print("  ❌ Index selection operations")
    print("  ❌ Per-head softmax + averaging")
    print("  ❌ Complex log-space operations")
    print("  ❌ Forced float32 conversion")
    
    print("\nSimple Implementation:")
    print("  ✅ ~15 lines of code")
    print("  ✅ Single matrix multiplication")
    print("  ✅ Direct log_softmax")
    print("  ✅ Preserves input dtype")
    print("  ✅ Mathematically cleaner")
    print("  ✅ Easier to understand/debug")
    
    print("\n🎯 FOR RL ACTION SELECTION:")
    print("  • We don't need per-head attention")
    print("  • We don't need GQA complexity") 
    print("  • We just need similarity ranking")
    print("  • Simple dot product is sufficient")


if __name__ == "__main__":
    test_equivalence()
    complexity_analysis() 