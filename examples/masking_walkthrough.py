"""
Detailed walkthrough of the masking logic in AttentionGuidedRL.

This demonstrates step-by-step how availability masks prevent reselection of keys
and ensure valid probability distributions.
"""

import torch
import numpy as np
from src.main import _build_available_mask
from src.embeddings import compute_similarity, get_attention_params
from transformers import GPT2LMHeadModel


def masking_logic_walkthrough():
    """
    Complete walkthrough of masking logic with concrete examples.
    """
    print("=" * 80)
    print("MASKING LOGIC WALKTHROUGH")
    print("=" * 80)
    
    # Setup - simulate a trajectory with 3 timesteps and 5 available keys
    batch_size = 2
    num_keys = 5
    num_timesteps = 3
    
    print(f"\n📋 SCENARIO SETUP:")
    print(f"   Batch size: {batch_size}")
    print(f"   Total keys available: {num_keys}")
    print(f"   Trajectory length: {num_timesteps} steps")
    print(f"   Goal: Select {num_timesteps} different keys per trajectory")
    
    # Initialize available indices (all keys available at start)
    available_indices_per_batch = [list(range(num_keys)) for _ in range(batch_size)]
    device = torch.device("cpu")
    
    print(f"\n🎯 INITIAL STATE:")
    for b in range(batch_size):
        print(f"   Batch {b} available keys: {available_indices_per_batch[b]}")
    
    # Setup model for similarity computation
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    num_heads, num_groups, head_dim = get_attention_params(model)
    hidden_size = num_heads * head_dim
    
    # Create dummy query and key embeddings
    query_embeddings = torch.randn(batch_size, hidden_size)
    key_embeddings = torch.randn(batch_size, num_keys, hidden_size)
    
    print(f"\n🔧 MODEL SETUP:")
    print(f"   Attention heads: {num_heads}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Query shape: {query_embeddings.shape}")
    print(f"   Key embeddings shape: {key_embeddings.shape}")
    
    # Simulate trajectory generation step by step
    selected_keys_per_batch = [[] for _ in range(batch_size)]
    
    for timestep in range(num_timesteps):
        print(f"\n" + "="*50)
        print(f"TIMESTEP {timestep + 1}")
        print("="*50)
        
        print(f"\n📊 STEP 1: Build Availability Mask")
        print("   Mathematical: mask[i,j] = 0 if key j available for batch i, -inf otherwise")
        print("   Code: _build_available_mask(available_indices_per_batch, num_keys, device)")
        
        # Build the mask
        availability_mask = _build_available_mask(available_indices_per_batch, num_keys, device)
        
        print(f"\n   Available indices per batch:")
        for b in range(batch_size):
            print(f"     Batch {b}: {available_indices_per_batch[b]}")
        
        print(f"\n   Availability mask shape: {availability_mask.shape}")
        print(f"   Availability mask values:")
        for b in range(batch_size):
            mask_str = []
            for k in range(num_keys):
                if availability_mask[b, k] == 0.0:
                    mask_str.append("  0.0")
                else:
                    mask_str.append(" -inf")
            print(f"     Batch {b}: [{', '.join(mask_str)}]")
        
        print(f"\n⚡ STEP 2: Compute Similarities with Masking")
        print("   Mathematical: Apply mask BEFORE softmax to ensure valid probabilities")
        print("   Code: compute_similarity(..., availability_mask=availability_mask)")
        
        # Compute similarities with masking
        log_probs = compute_similarity(
            query_embeddings, key_embeddings, num_heads, num_groups, head_dim,
            availability_mask=availability_mask
        )
        
        # Convert to probabilities for display
        probs = torch.exp(log_probs)
        
        print(f"\n   Log probabilities shape: {log_probs.shape}")
        print(f"   Log probabilities:")
        for b in range(batch_size):
            log_prob_str = [f"{log_probs[b, k]:.3f}" for k in range(num_keys)]
            print(f"     Batch {b}: [{', '.join(log_prob_str)}]")
        
        print(f"\n   Probabilities (exp of log probs):")
        for b in range(batch_size):
            prob_str = [f"{probs[b, k]:.3f}" for k in range(num_keys)]
            print(f"     Batch {b}: [{', '.join(prob_str)}]")
        
        print(f"\n   Probability sums (should be 1.0):")
        for b in range(batch_size):
            prob_sum = probs[b].sum().item()
            print(f"     Batch {b}: {prob_sum:.6f}")
        
        # Verify masking worked correctly
        print(f"\n✅ MASKING VERIFICATION:")
        for b in range(batch_size):
            for k in range(num_keys):
                if k not in available_indices_per_batch[b]:
                    prob_val = probs[b, k].item()
                    is_masked = prob_val < 1e-6
                    print(f"     Batch {b}, Key {k}: {'✓ Masked' if is_masked else '✗ NOT MASKED'} (prob={prob_val:.2e})")
        
        print(f"\n🎲 STEP 3: Sample Keys (Simulated)")
        print("   Real code: sample_key_value(similarity_scores, available_indices_per_batch, batch_size)")
        print("   For demo: Select highest probability available key")
        
        # Simulate sampling (for demo, just pick highest probability available key)
        selected_indices = []
        for b in range(batch_size):
            available_keys = available_indices_per_batch[b]
            available_probs = probs[b, available_keys]
            best_local_idx = torch.argmax(available_probs).item()
            selected_key = available_keys[best_local_idx]
            selected_indices.append(selected_key)
            selected_keys_per_batch[b].append(selected_key)
        
        print(f"\n   Selected keys:")
        for b in range(batch_size):
            selected_key = selected_indices[b]
            prob = probs[b, selected_key].item()
            print(f"     Batch {b}: Key {selected_key} (probability: {prob:.3f})")
        
        print(f"\n🔄 STEP 4: Update Available Indices")
        print("   Code: available_indices_per_batch[b].remove(selected_idx)")
        
        # Update available indices (remove selected keys)
        for b, selected_key in enumerate(selected_indices):
            if selected_key in available_indices_per_batch[b]:
                available_indices_per_batch[b].remove(selected_key)
        
        print(f"\n   Updated available indices:")
        for b in range(batch_size):
            print(f"     Batch {b}: {available_indices_per_batch[b]}")
    
    print(f"\n" + "="*80)
    print("TRAJECTORY SUMMARY")
    print("="*80)
    
    print(f"\n📈 Final selected keys per batch:")
    for b in range(batch_size):
        print(f"   Batch {b}: {selected_keys_per_batch[b]}")
    
    # Verify no duplicates
    print(f"\n✅ Duplicate verification:")
    for b in range(batch_size):
        keys = selected_keys_per_batch[b]
        has_duplicates = len(keys) != len(set(keys))
        print(f"   Batch {b}: {'✗ Has duplicates' if has_duplicates else '✓ No duplicates'}")


def demonstrate_masking_math():
    """
    Show the mathematical difference between correct and incorrect masking.
    """
    print(f"\n" + "="*80)
    print("MATHEMATICAL COMPARISON: Correct vs Incorrect Masking")
    print("="*80)
    
    # Simple example
    raw_scores = torch.tensor([[2.0, 1.0, 3.0]])  # Raw similarity scores
    mask = torch.tensor([[0.0, -float('inf'), 0.0]])  # Key 1 is unavailable
    
    print(f"\n📊 INPUT:")
    print(f"   Raw similarity scores: {raw_scores[0].tolist()}")
    print(f"   Availability mask: [0.0, -inf, 0.0] (key 1 unavailable)")
    
    print(f"\n❌ INCORRECT APPROACH (Old Implementation):")
    print("   1. Apply softmax to raw scores")
    wrong_probs = torch.softmax(raw_scores, dim=1)
    print(f"      softmax([2.0, 1.0, 3.0]) = {wrong_probs[0].tolist()}")
    
    print("   2. Add mask to log probabilities")
    wrong_log_probs = torch.log(wrong_probs) + mask
    wrong_final_probs = torch.exp(wrong_log_probs)
    print(f"      log_probs + mask = {wrong_log_probs[0].tolist()}")
    print(f"      exp(result) = {wrong_final_probs[0].tolist()}")
    print(f"      Sum: {wrong_final_probs[0].sum().item():.6f} ≠ 1.0 ❌")
    
    print(f"\n✅ CORRECT APPROACH (Fixed Implementation):")
    print("   1. Add mask to raw scores BEFORE softmax")
    masked_scores = raw_scores + mask
    print(f"      raw_scores + mask = {masked_scores[0].tolist()}")
    
    print("   2. Apply softmax to masked scores")
    correct_probs = torch.softmax(masked_scores, dim=1)
    print(f"      softmax(masked_scores) = {correct_probs[0].tolist()}")
    print(f"      Sum: {correct_probs[0].sum().item():.6f} = 1.0 ✅")
    
    print(f"\n🔍 KEY INSIGHT:")
    print("   Masking BEFORE softmax ensures exp(-inf) = 0 and remaining probabilities sum to 1")
    print("   Masking AFTER softmax breaks the probability distribution")


def trace_code_execution():
    """
    Trace through the actual code execution path.
    """
    print(f"\n" + "="*80)
    print("CODE EXECUTION TRACE")
    print("="*80)
    
    print(f"\n📍 MASK CREATION (src/main.py:96-102):")
    print("   def _build_available_mask(available_indices_per_batch, num_keys, device):")
    print("     Line 99:  mask = torch.full((batch_size, num_keys), float('-inf'), device=device)")
    print("     Line 100-101: for b, avail in enumerate(available_indices_per_batch):")
    print("                     mask[b, avail] = 0.0")
    print("     Line 102: return mask")
    
    print(f"\n📍 MASK USAGE (src/main.py:166-169):")
    print("     Line 166: available_mask = _build_available_mask(available_indices_per_batch, num_keys, device)")
    print("     Line 167: available_mask = available_mask.clamp(min=-1e9)  # Prevent -inf for numerical stability")
    print("     Line 168-169: similarity_scores = compute_similarity(..., availability_mask=available_mask)")
    
    print(f"\n📍 MASK APPLICATION (src/embeddings.py:465-470):")
    print("     Line 465: if availability_mask is not None:")
    print("     Line 467:   expanded_mask = availability_mask.unsqueeze(1).expand(-1, num_heads, -1)")
    print("     Line 468:   scaled_similarities = scaled_similarities + expanded_mask")
    print("     Line 470: head_probabilities = F.softmax(scaled_similarities, dim=2)")
    
    print(f"\n📍 INDEX UPDATE (src/main.py:230-232):")
    print("     Line 230-232: for b, idx in enumerate(selected_indices):")
    print("                     if idx in available_indices_per_batch[b]:")
    print("                       available_indices_per_batch[b].remove(idx)")
    
    print(f"\n✅ EXECUTION FLOW:")
    print("   1. Start with all keys available")
    print("   2. Build mask: 0.0 for available, -inf for unavailable")
    print("   3. Apply mask BEFORE softmax in attention computation")
    print("   4. Sample from valid probability distribution")
    print("   5. Remove selected key from available list")
    print("   6. Repeat for next timestep")


if __name__ == "__main__":
    masking_logic_walkthrough()
    demonstrate_masking_math()
    trace_code_execution() 