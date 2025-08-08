"""
Detailed walkthrough of GAE (Generalized Advantage Estimation) computation.

This file explains step-by-step how GAE works mathematically and how it's implemented
in our codebase, using concrete numerical examples.
"""

import torch
import numpy as np
from src.training import compute_returns, compute_advantages


def gae_mathematical_walkthrough():
    """
    Step-by-step walkthrough of GAE computation with concrete example.
    """
    print("=" * 80)
    print("GAE (Generalized Advantage Estimation) Mathematical Walkthrough")
    print("=" * 80)
    
    # Simple example with 2 batch items, 4 timesteps
    batch_size = 2
    num_steps = 4
    
    # Example rewards
    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # Trajectory 1
        [0.5, 1.5, 2.5, 3.5]   # Trajectory 2
    ])
    
    gamma = 0.9
    gae_lambda = 0.95
    
    print(f"\n📊 INPUT DATA:")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards:\n{rewards}")
    print(f"Gamma (discount factor): {gamma}")
    print(f"Lambda (GAE parameter): {gae_lambda}")
    
    print(f"\n🔢 STEP 1: Compute Returns (Rewards-to-go)")
    print("=" * 50)
    
    # Step 1: Compute returns
    returns = compute_returns(rewards, gamma)
    
    print("Returns formula: R_t = r_t + γ * R_{t+1}")
    print("Working backwards from the end:")
    
    # Manual computation for verification
    manual_returns = torch.zeros_like(rewards)
    manual_returns[:, -1] = rewards[:, -1]  # Last timestep
    
    print(f"\nFor Trajectory 1:")
    print(f"R_3 = r_3 = {rewards[0, 3]:.1f}")
    manual_returns[0, 2] = rewards[0, 2] + gamma * manual_returns[0, 3]
    print(f"R_2 = r_2 + γ*R_3 = {rewards[0, 2]:.1f} + {gamma}*{manual_returns[0, 3]:.1f} = {manual_returns[0, 2]:.2f}")
    manual_returns[0, 1] = rewards[0, 1] + gamma * manual_returns[0, 2]
    print(f"R_1 = r_1 + γ*R_2 = {rewards[0, 1]:.1f} + {gamma}*{manual_returns[0, 2]:.2f} = {manual_returns[0, 1]:.3f}")
    manual_returns[0, 0] = rewards[0, 0] + gamma * manual_returns[0, 1]
    print(f"R_0 = r_0 + γ*R_1 = {rewards[0, 0]:.1f} + {gamma}*{manual_returns[0, 1]:.3f} = {manual_returns[0, 0]:.4f}")
    
    # Complete manual computation for trajectory 2
    manual_returns[1, 2] = rewards[1, 2] + gamma * manual_returns[1, 3]
    manual_returns[1, 1] = rewards[1, 1] + gamma * manual_returns[1, 2]
    manual_returns[1, 0] = rewards[1, 0] + gamma * manual_returns[1, 1]
    
    print(f"\nComputed returns:\n{returns}")
    print(f"Manual verification:\n{manual_returns}")
    print(f"✅ Match: {torch.allclose(returns, manual_returns, atol=1e-4)}")
    
    print(f"\n🎯 STEP 2: Compute Baseline (Value Function Estimate)")
    print("=" * 50)
    
    # Step 2: Compute baseline as batch mean of returns
    baseline = returns.mean(dim=0, keepdim=True)
    
    print("Baseline formula: V̂(s_t) = E[R_t] ≈ (1/batch_size) * Σ R_t^(i)")
    print("This uses the batch average of returns as our value function estimate.")
    print(f"\nBaseline (per timestep):\n{baseline}")
    print(f"Shape: {baseline.shape} (broadcast across batch dimension)")
    
    print(f"\n⚡ STEP 3: Compute GAE Advantages")
    print("=" * 50)
    
    print("GAE Formula: A^GAE_t = δ_t + γλ * A^GAE_{t+1}")
    print("where δ_t = r_t + γ*V̂(s_{t+1}) - V̂(s_t)")
    print("\nWorking backwards from the last timestep:")
    
    # Manual GAE computation
    advantages_manual = torch.zeros_like(rewards)
    
    # Last timestep (T-1 = 3)
    t = 3
    advantages_manual[:, t] = returns[:, t] - baseline[:, t]
    print(f"\nTimestep {t} (last):")
    print(f"A_3 = R_3 - V̂_3 = R_3 - baseline_3")
    for b in range(batch_size):
        print(f"  Batch {b}: A_3^({b}) = {returns[b, t]:.2f} - {baseline[0, t]:.2f} = {advantages_manual[b, t]:.3f}")
    
    # Work backwards through remaining timesteps
    for t in reversed(range(num_steps - 1)):
        print(f"\nTimestep {t}:")
        
        # Compute temporal difference δ_t
        delta_t = rewards[:, t] + gamma * baseline[:, t + 1] - baseline[:, t]
        print(f"δ_{t} = r_{t} + γ*V̂_{t+1} - V̂_{t}")
        
        for b in range(batch_size):
            delta_val = rewards[b, t] + gamma * baseline[0, t + 1] - baseline[0, t]
            print(f"  Batch {b}: δ_{t}^({b}) = {rewards[b, t]:.1f} + {gamma}*{baseline[0, t + 1]:.2f} - {baseline[0, t]:.2f} = {delta_val:.3f}")
        
        # Apply GAE recurrence
        advantages_manual[:, t] = delta_t + gamma * gae_lambda * advantages_manual[:, t + 1]
        print(f"A_{t} = δ_{t} + γλ*A_{t+1}")
        
        for b in range(batch_size):
            gae_term = gamma * gae_lambda * advantages_manual[b, t + 1]
            total = delta_t[b] + gae_term
            print(f"  Batch {b}: A_{t}^({b}) = {delta_t[b]:.3f} + {gamma}*{gae_lambda}*{advantages_manual[b, t + 1]:.3f} = {total:.4f}")
    
    print(f"\n🧮 STEP 4: Compare with Implementation")
    print("=" * 50)
    
    # Use our implementation
    advantages_computed, returns_computed = compute_advantages(
        rewards, gamma, gae_lambda, use_grpo_baseline=True
    )
    
    print(f"Manual GAE computation:\n{advantages_manual}")
    print(f"Implementation result:\n{advantages_computed}")
    print(f"✅ Match: {torch.allclose(advantages_manual, advantages_computed, atol=1e-4)}")
    
    print(f"\n📈 STEP 5: Verify GAE Properties")
    print("=" * 50)
    
    # Check centering property
    mean_advantages = advantages_computed.mean(dim=0)
    print(f"Mean advantages per timestep: {mean_advantages}")
    print(f"Should be close to zero (properly centered): {torch.allclose(mean_advantages, torch.zeros_like(mean_advantages), atol=1e-6)}")
    
    # Check variance reduction compared to Monte Carlo
    advantages_mc, _ = compute_advantages(rewards, gamma, gae_lambda=1.0, use_grpo_baseline=True)
    gae_variance = advantages_computed.var()
    mc_variance = advantages_mc.var()
    print(f"\nVariance comparison:")
    print(f"GAE (λ={gae_lambda}) variance: {gae_variance:.4f}")
    print(f"Monte Carlo (λ=1.0) variance: {mc_variance:.4f}")
    print(f"Variance reduction: {((mc_variance - gae_variance) / mc_variance * 100):.1f}%")
    
    print(f"\n🎓 STEP 6: Interpretation")
    print("=" * 50)
    
    print("Key insights about GAE:")
    print("1. λ=0: Only uses 1-step TD error (high bias, low variance)")
    print("2. λ=1: Uses full Monte Carlo returns (low bias, high variance)")
    print("3. 0<λ<1: Exponentially weighted mixture (balanced bias-variance)")
    print("4. Baseline centering ensures advantages have zero mean across batch")
    print("5. Using batch mean of returns as value function is a simple but effective choice")
    
    print(f"\nIn our implementation:")
    print(f"- We use λ={gae_lambda}, which gives moderate variance reduction")
    print(f"- Baseline = batch mean of returns at each timestep")
    print(f"- This works well for episodic tasks where we have full trajectories")


def lambda_comparison():
    """
    Show how different λ values affect GAE computation.
    """
    print(f"\n" + "=" * 80)
    print("GAE Lambda Parameter Comparison")
    print("=" * 80)
    
    # Same example data
    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5]
    ])
    
    gamma = 0.9
    lambdas = [0.0, 0.5, 0.95, 1.0]
    
    for lam in lambdas:
        advantages, _ = compute_advantages(rewards, gamma, gae_lambda=lam, use_grpo_baseline=True)
        variance = advantages.var()
        
        print(f"\nλ = {lam}:")
        print(f"  Advantages:\n  {advantages}")
        print(f"  Variance: {variance:.4f}")
        
        if lam == 0.0:
            print("  → Pure TD(0): Only immediate temporal difference")
        elif lam == 1.0:
            print("  → Pure Monte Carlo: Full trajectory returns")
        else:
            print(f"  → Mixed: {lam*100:.0f}% future weighting")


if __name__ == "__main__":
    gae_mathematical_walkthrough()
    lambda_comparison() 