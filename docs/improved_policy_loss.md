# Improved Policy Loss Implementation

## Overview

This document describes the improvements made to the policy loss computation in the Attention-Guided RL system. The changes transform the original REINFORCE-style loss into a more sophisticated policy gradient formulation with:

1. **Proper return computation** with discount factor γ
2. **Advantage estimation** for variance reduction
3. **Entropy regularization** for exploration
4. **Improved KL divergence** computation

## Key Components

### 1. Return Computation

Instead of using average rewards across the trajectory, we now compute discounted returns (rewards-to-go):

```
returns[t] = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...
```

This gives each timestep credit for all future rewards, discounted by γ (default: 0.99).

### 2. Advantage Estimation (GRPO-style)

Advantages measure how much better an action was compared to the expected value:

```python
advantage[t] = returns[t] - baseline[t]
```

We implement **GRPO (Group Relative Policy Optimization)** style baseline:
- At each timestep t, the baseline is the mean return across the batch
- This naturally handles the fact that earlier steps have higher returns due to more future rewards
- Advantages sum to zero at each timestep by construction

```
baseline[t] = mean(returns[:, t])  # Average return at timestep t across batch
advantages[:, t] = returns[:, t] - baseline[t]
```

Benefits of GRPO baseline:
- No additional parameters to train (unlike value functions)
- Automatically adapts to step-dependent return magnitudes
- Reduces variance while maintaining unbiased gradients
- Simple and effective for language model fine-tuning

### 3. Entropy Regularization

To encourage exploration:
- **Token queries**: Use categorical entropy from the softmax distribution
- **Vector queries**: Use Gaussian entropy: $H = 0.5 \\times d \\times (1 + \\log(2\\pi)) + d \\times \\log(\\sigma)$

### 4. Loss Formulation

The total loss is now:

$$
L = L_{\\text{policy}} + \\beta \\times L_{\\text{KL}} - \\alpha \\times H_{\\text{entropy}}
$$

Where:
- $L_{\\text{policy}} = -\\mathbb{E}[\\log \\pi(a|s) \\times A(s,a)]$ (policy gradient with advantages)
- $L_{\\text{KL}}$ = KL divergence between current and previous policy
- $H_{\\text{entropy}}$ = Entropy of the policy distribution
- $\\beta$ = KL penalty coefficient (default: 0.1)
- $\\alpha$ = Entropy coefficient (default: 0.01)

## Configuration Parameters

New parameters in `src/config.py`:

```
GAMMA = 0.99          # Discount factor for returns
GAE_LAMBDA = 0.95     # GAE lambda (for future use)
ENTROPY_COEF = 0.01   # Entropy bonus coefficient
POLICY_SIGMA = 0.1    # Std dev for Gaussian policy
USE_GRPO_BASELINE = True  # Use GRPO-style baseline
```

## Implementation Details

### For Token-based Queries

1. Generate query tokens from language model
2. Compute log probabilities under current and previous policies
3. Apply policy gradient loss with advantages
4. Add categorical entropy bonus

### For Vector-based Queries

1. Generate query vectors from $\\mathcal{N}(\\mu(s), \\sigma^2 I)$
2. Store log prob of sampled vector under the Gaussian
3. Apply policy gradient loss with advantages
4. Add Gaussian entropy bonus
5. Approximate KL divergence using L2 distance of means

## Benefits

1. **Reduced variance**: Advantages normalize rewards and remove baseline
2. **Better exploration**: Entropy bonus prevents premature convergence
3. **Stable learning**: KL constraint prevents large policy updates
4. **Principled credit assignment**: Proper discounting of future rewards

## Future Improvements

1. **Value function baseline**: Train a value head to predict returns
2. **GAE implementation**: Use TD($\\lambda$) for advantage estimation
3. **PPO clipping**: Add probability ratio clipping for stability
4. **Better KL computation**: Store or regenerate previous policy means 