#!/usr/bin/env python3
"""
Demonstration of GRPO baseline gaming vulnerability.
"""

import torch

def demonstrate_grpo_gaming():
    print("=== GRPO Baseline Gaming Demonstration ===\n")
    
    # Scenario 1: Normal performance
    print("Scenario 1: Normal diverse performance")
    returns_normal = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 trajectories in batch
    ]).T  # Shape: [5, 1] (5 trajectories, 1 timestep)
    
    baseline_normal = returns_normal.mean(dim=0, keepdim=True)
    advantages_normal = returns_normal - baseline_normal
    
    print(f"Returns: {returns_normal.flatten().tolist()}")
    print(f"Baseline: {baseline_normal.item():.2f}")
    print(f"Advantages: {advantages_normal.flatten().tolist()}")
    print(f"Positive advantages: {(advantages_normal > 0).sum().item()}/5")
    print()
    
    # Scenario 2: Gaming by deliberately bad performance
    print("Scenario 2: Gaming with deliberately bad trajectories")
    returns_gamed = torch.tensor([
        [-10.0, -8.0, 4.0, 5.0, 6.0]  # Deliberately tank first 2 trajectories
    ]).T
    
    baseline_gamed = returns_gamed.mean(dim=0, keepdim=True) 
    advantages_gamed = returns_gamed - baseline_gamed
    
    print(f"Returns: {returns_gamed.flatten().tolist()}")
    print(f"Baseline: {baseline_gamed.item():.2f}")
    print(f"Advantages: {advantages_gamed.flatten().tolist()}")
    print(f"Positive advantages: {(advantages_gamed > 0).sum().item()}/5")
    print()
    
    print("🚨 GAMING EFFECT:")
    print(f"- Normal: {(advantages_normal > 0).sum().item()}/5 positive advantages")
    print(f"- Gamed: {(advantages_gamed > 0).sum().item()}/5 positive advantages")
    print("- By doing badly on some trajectories, model gets more positive advantages!")
    print()
    
    print("💡 POTENTIAL MITIGATIONS:")
    print("1. Use fixed baseline (like original base model)")
    print("2. Use running average baseline across episodes")
    print("3. Clip advantage values to prevent extreme gaming")
    print("4. Use smaller batch sizes (harder to game)")

if __name__ == "__main__":
    demonstrate_grpo_gaming() 