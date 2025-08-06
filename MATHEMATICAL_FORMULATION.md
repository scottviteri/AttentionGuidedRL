# Mathematical Formulation of Attention-Guided Reinforcement Learning

**AttentionGuidedRL Repository Mathematical Documentation**

---

## Abstract

This document provides a comprehensive mathematical formulation of the optimization objective implemented in the Attention-Guided Reinforcement Learning (AttentionGuidedRL) repository. The system trains a language model to autonomously guide its own training by sequencing key-value pairs from Wikipedia articles using vanilla policy gradient (REINFORCE) with multi-head attention mechanisms and GRPO baseline estimation.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [Policy Definition](#policy-definition)
4. [Reward Function](#reward-function)
5. [Optimization Objective](#optimization-objective)
6. [Advantage Estimation with GAE](#advantage-estimation-with-gae)
7. [Loss Function and Training Algorithm](#loss-function-and-training-algorithm)
8. [Training Algorithm](#training-algorithm)
9. [Implementation Details](#implementation-details)
10. [Key Mathematical Properties](#key-mathematical-properties)
11. [Conclusion](#conclusion)

---

## Introduction

The AttentionGuidedRL system implements a novel approach where a base language model (Llama-3.2-3B or GPT-2) learns to sequence its own training data through reinforcement learning. The model generates vector queries using attention mechanisms and selects key-value pairs to maximize learning progress, measured by conditional log probabilities of value tokens. The system uses an **advantage-based policy gradient approach with GRPO baseline estimation** and θ-dependent reward gradients for stable training.

---

## Problem Formulation

### State Space and Action Space

**Definition 1 (State Space):** At timestep $t$, the state $s_t$ consists of:

$$s_t = (c_t, \mathcal{K}_t^{\text{available}})$$

where:
- $c_t \in \mathbb{Z}^{L_t}$ is the context sequence of length $L_t$
- $\mathcal{K}_t^{\text{available}} \subseteq \{1, 2, \ldots, K\}$ is the set of available key indices

**Definition 2 (Action Space):** The action space $\mathcal{A}_t$ at timestep $t$ is the set of available key indices:

$$\mathcal{A}_t = \mathcal{K}_t^{\text{available}}$$

### Trajectory and Episode Structure

A trajectory $\tau$ consists of $T$ steps:

$$\tau = \{(s_1, a_1, r_1), (s_2, a_2, r_2), \ldots, (s_T, a_T, r_T)\}$$

where $T$ is the number of key-value pairs per episode (NUM_KV_PAIRS in the code).

---

## Policy Definition

### Vector Query Generation

The policy $\pi_\theta$ operates through a vector query mechanism parameterized by LoRA adapter weights $\theta$.

**Definition 3 (Query Vector Generation):** Given context $c_t$, the query vector is generated as:

$$q_t = \text{Embed}_\theta(c_t \oplus [\text{"Query"}])_{-1}$$

where:
- $\text{Embed}_\theta(\cdot)$ extracts embeddings from the second-to-last attention layer
- $\oplus$ denotes sequence concatenation
- $(\cdot)_{-1}$ extracts the embedding of the last token (the "Query" token)
- $q_t \in \mathbb{R}^{d_{\text{model}}}$ where $d_{\text{model}}$ is the model dimension

This focused extraction ensures that the model learns to encode its query intent specifically into the "Query" token, rather than diluting the signal across the entire context.

### Multi-Head Attention Similarity

The system handles both standard Multi-Head Attention (MHA) and Grouped Query Attention (GQA).

**Definition 4 (Head-wise Similarity Computation):** For each attention head $h \in \{1, 2, \ldots, H\}$ and key $k$:

$$\text{sim}_{t,k}^{(h)} = \frac{(q_t^{(h)})^T k_k^{(\text{group}(h))}}{\sqrt{d_{\text{head}}}} \cdot \frac{1}{\tau}$$

where:
- $q_t^{(h)} \in \mathbb{R}^{d_{\text{head}}}$ is the query for head $h$
- $k_k^{(\text{group}(h))} \in \mathbb{R}^{d_{\text{head}}}$ is the key for the corresponding group
- $\text{group}(h) = \lfloor h \cdot G / H \rfloor$ maps heads to groups (for GQA)
- $G$ is the number of key-value groups
- $H$ is the number of query heads
- $d_{\text{head}} = d_{\text{model}} / H$ is the head dimension
- $\tau = 1.0$ is the temperature parameter

**Definition 5 (Per-Head Probability Distribution):** For each head $h$, the probability distribution over keys is:

$$p_{t,k}^{(h)} = \frac{\exp(\text{sim}_{t,k}^{(h)})}{\sum_{k' \in \mathcal{K}_t^{\text{available}}} \exp(\text{sim}_{t,k'}^{(h)})}$$

**Definition 6 (Policy Distribution):** The final policy distribution is obtained by averaging over heads:

$$\pi_\theta(k | s_t) = \frac{1}{H} \sum_{h=1}^{H} p_{t,k}^{(h)}$$

---

## Reward Function

**Definition 7 (θ-Dependent Step Reward):** The reward at step $t$ depends on the current model parameters $\theta$:

$$r_{\theta,t} = \log p_\theta(v_t | c_t, k_t)$$

where:
- $v_t$ are the value tokens at step $t$
- $k_t$ are the selected key tokens at step $t$  
- $p_\theta(\cdot)$ is the current adapter model probability (with LoRA parameters $\theta$)
- The reward is computed using the **current model**, making it θ-dependent

**Definition 8 (Conditional Log Probability):** The conditional log probability is computed as:

$$\log p_\theta(v_t | c_t, k_t) = \frac{1}{|v_t|} \sum_{i=1}^{|v_t|} \log p_\theta(v_{t,i} | c_t, k_t, v_{t,<i})$$

where $v_{t,i}$ is the $i$-th token in the value sequence and $v_{t,<i}$ represents the preceding tokens.

**Implementation Note:** For computational efficiency, rewards are computed once during trajectory generation and reused during the chain rule update. This preserves the same gradient flow while reducing memory usage.

**Configuration Option:** The system supports two reward modes:
- `subtract_base_model_logprobs=True`: $r_{\theta,t} = \log p_\theta(v_t | c_t, k_t) - \log p_{\text{ref}}(v_t | c_t, k_t)$
- `subtract_base_model_logprobs=False`: $r_{\theta,t} = \log p_\theta(v_t | c_t, k_t)$ (default)

---

## Optimization Objective

### Primary Objective Function

**Theorem 1 (Policy Gradient with θ-Dependent Rewards):** The system maximizes the following objective function:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R_\theta(\tau) \right]$$

where $R_\theta(\tau) = \sum_{t=1}^T \gamma^{t-1} r_{\theta,t}$ and the instantaneous reward depends on the current model parameters:

$$r_{\theta,t} = \frac{1}{|v_t|} \sum_{i=1}^{|v_t|} \log p_\theta(v_{t,i} | c_t, k_t, v_{t,<i})$$

**Key Insight:** Since rewards depend on $\theta$, the gradient derivation uses the chain rule combined with **advantage-based policy gradients**:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T A_t \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) + \lambda \sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t} \right]$$

where:
- $A_t$ are **normalized advantages** computed using GRPO baseline: $A_t = \frac{r_{\theta,t} - b_{\text{batch}} - \mu_A}{\sigma_A + \epsilon}$
- $b_{\text{batch}} = \frac{1}{B} \sum_{i=1}^B \sum_{t=1}^T r_{\theta,t}^{(i)}$ is the GRPO batch baseline
- $\mu_A, \sigma_A$ are the mean and standard deviation of advantages for normalization
- $\lambda = 0.1$ is a scaling factor for the reward gradient term
- $\nabla_\theta r_{\theta,t}$ is the reward gradient term (recomputed efficiently)

**Chain Rule Justification:** Since the reward function $r_{\theta,t}$ depends on the same parameters $\theta$ as the policy, the chain rule for $\frac{d}{d\theta}\mathbb{E}_{\tau \sim \pi_\theta}[R_\theta(\tau)]$ requires both terms:

1. **Policy gradient term:** How changing $\theta$ affects the probability of sampling high-reward trajectories
2. **Reward gradient term:** How changing $\theta$ affects the reward evaluation of sampled trajectories

**Mathematical Equivalence:** In expectation, this chain rule approach produces the same gradient direction as standard REINFORCE with advantages:
$$\mathbb{E}_{\tau \sim \pi_\theta}[R_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=1}^T R_t(\tau) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)]$$

where $R_t(\tau)$ are the returns-to-go. The reward gradient term $\nabla_\theta R_\theta(\tau)$ provides additional signal that standard REINFORCE lacks.

**Key Insight:** This chain rule approach uses the **total trajectory return** $R_\theta(\tau)$ to weight each action's gradient contribution, rather than step-specific advantages. This creates a unified learning signal where all actions in successful trajectories receive positive gradients.

### Total Return Computation

**Definition 9 (Total Trajectory Return):** The total discounted return for trajectory $\tau$ is:

$$R_\theta(\tau) = \sum_{t=1}^T \gamma^{t-1} r_{\theta,t}$$

where $r_{\theta,t}$ is the θ-dependent reward at step $t$.

**Key Property:** Unlike advantage-based methods, this approach assigns the **same weight** (the total return) to all actions within a trajectory, emphasizing trajectory-level success rather than step-specific credit assignment.

**Note:** The system implements the chain rule approach directly and does not use advantage estimation or GAE. The advantages computation exists in the codebase for compatibility and monitoring purposes but is not used in the core training loop.

---

## Loss Function and Training Algorithm

### Chain Rule Policy Gradient Loss

**Theorem 2 (Policy Gradient with θ-Dependent Reward Chain Rule):** The loss function implements the chain rule derivation for θ-dependent rewards:

$$\mathcal{L}(\theta) = \mathbb{E}_{\text{batch}} \left[ -R_\theta(\tau) \cdot \sum_{t=1}^T \log \pi_\theta(a_t | s_t) - \sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t} \right]$$

**Decomposition into Two Terms:**

1. **Policy Gradient Term**: $-R_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)$
   - Standard REINFORCE scaled by total trajectory return
   - Encourages actions that led to high-reward trajectories

2. **Reward Gradient Term**: $-\nabla_\theta R_\theta(\tau) = -\sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t}$
   - Direct optimization of reward model parameters
   - Natural regularization by encouraging the model to assign higher rewards

**Key Properties:**
1. **Self-Consistent Rewards**: Both policy and reward computation use current model $\pi_\theta$
2. **Natural Regularization**: Reward gradient term provides stability without artificial KL penalty
3. **Unified Objective**: Both action selection and reward assignment optimized jointly
4. **Mathematical Rigor**: Proper chain rule application for θ-dependent rewards

### Trajectory Sampling 

**Definition 12 (Current Policy Sampling):** All trajectories are sampled using the current policy:

$$\tau \sim \pi_\theta$$

- Trajectories are generated using the current adapter model (with LoRA weights)
- This ensures truly on-policy data for policy gradient updates
- No old model copies or memory-efficient state management needed
- Simpler implementation with lower memory footprint

**Mathematical Formulation:** The chain rule policy gradient becomes:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R_\theta(\tau) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t) + \sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t} \right]$$

where $\nabla_\theta \log \pi_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$ is the sum of individual action log-gradients.

### Chain Rule Implementation Details

**Definition 13 (Chain Rule Gradient Decomposition):** The θ-dependent objective decomposes into two complementary terms:

1. **Policy Term**: $R_\theta(\tau) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$
   - Uses total trajectory return to weight all action gradients equally
   - Successful trajectories reinforce all actions that led to success
   - Simpler credit assignment: trajectory-level rather than step-level

2. **Reward Term**: $\sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t}$
   - Direct optimization of the reward function itself
   - Encourages the model to assign higher probabilities to selected value tokens
   - Provides natural regularization without artificial penalties

**Key Property:** Both terms optimize the same parameters θ, creating a **self-consistent learning signal** where the model learns to both select good actions AND accurately evaluate their outcomes.

**Comparison with Standard REINFORCE:**

- **Standard REINFORCE**: $\mathcal{L} = -\sum_{t=1}^T A_t(\tau) \log \pi_\theta(a_t | s_t)$ (step-specific advantages)
- **Chain Rule Approach**: $\mathcal{L} = -R_\theta(\tau) \sum_{t=1}^T \log \pi_\theta(a_t | s_t) - \sum_{t=1}^T \gamma^{t-1} \nabla_\theta r_{\theta,t}$ (total return + reward gradient)

**Advantages of Chain Rule Approach:**
1. **Unified Objective**: Policy and reward optimization happen simultaneously
2. **Natural Regularization**: Reward term prevents divergence without KL penalties
3. **Self-Consistency**: No mismatch between action sampling and reward evaluation
4. **Simplified Training**: No need for advantage computation or baseline management

### Implementation Details: Math-Code Correspondence

**Code Implementation Overview:**
```python
# 1. Generate complete trajectories using current policy
trajectory = generate_trajectory(adapter_model, ...)

# 2. Compute total trajectory returns R_θ(τ)
total_returns = trajectory.rewards.sum(dim=1)  # Sum over time steps for each trajectory

# 3. Apply θ-dependent reward chain rule loss
policy_term = 0.0
reward_term = 0.0

for t, qkv_step in enumerate(trajectory.qkv_steps):
    # Policy gradient term: R_θ(τ) * ∇log π_θ(a_t|s_t)
    current_action_log_probs = compute_action_log_probs(qkv_step)
    policy_gradient_t = total_returns * current_action_log_probs
    policy_term += policy_gradient_t.sum()  # Sum over batch
    
    # Reward gradient term: ∇_θ r_{θ,t} (via pre-computed rewards with autograd)
    step_reward = trajectory.rewards[:, t]  # Pre-computed during trajectory generation
    gamma_t = gamma ** t
    reward_term += gamma_t * step_reward.sum()  # Sum over batch

# Total loss: negative for gradient ascent → descent  
total_loss = -(policy_term + reward_term)
```

**Mathematical Correspondence:**
- **Line `total_returns`** ↔ $R_\theta(\tau)$ - total discounted return for each trajectory in batch
- **Line `total_returns * current_action_log_probs`** ↔ $R_\theta(\tau) \cdot \log \pi_\theta(a_t | s_t)$ - policy gradient term
- **Line `reward_t`** ↔ $r_{\theta,t}$ - instantaneous θ-dependent reward
- **Line `gamma_t * reward_t.sum()`** ↔ $\gamma^{t-1} \nabla_\theta r_{\theta,t}$ - discounted reward gradient
- **Line `policy_term + reward_term`** ↔ complete chain rule decomposition $\nabla_\theta \mathcal{J}(\theta)$

**Key Insight:** Both the policy (action selection) and reward (outcome evaluation) are optimized using the same current model parameters $\theta$. This creates a self-consistent learning signal where the model learns to both select good actions AND accurately evaluate their outcomes.

**Stability Analysis:** The reward gradient term $\nabla_\theta R_\theta(\tau)$ provides natural regularization by encouraging the model to assign higher probabilities to the value tokens it selected. This prevents divergence without requiring artificial KL penalties against a mismatched reference policy.

---

## Training Algorithm

**Algorithm 1: Attention-Guided RL Training (Chain Rule Policy Gradient)**

```
Input: Base model, tokenizer, Wikipedia dataset
Input: Hyperparameters γ, learning rate, batch size

1. Initialize LoRA adapter parameters θ
2. Initialize optimizer (AdamW with learning rate 5×10⁻⁴)

3. For episode = 1 to NUM_EPISODES:
   a. Sample Wikipedia article and extract key-value pairs
   b. Generate trajectory τ using current policy π_θ
   c. Compute rewards {r_{θ,t}}ᵀₜ₌₁ using current model θ
   d. Compute total returns R_θ(τ) = Σₜγᵗ⁻¹r_{θ,t} for each trajectory
   e. Compute chain rule loss: ℒ(θ) = -(R_θ(τ)·∇log π_θ(τ) + ∇R_θ(τ))
   f. Update parameters: θ ← θ - ∇_θ ℒ(θ)
   g. Save checkpoint periodically
```

---

## Implementation Details

### Model Architecture
- **Base Models**: Llama-3.2-3B or GPT-2
- **LoRA Configuration**: rank=8, α=16, dropout=0.0
- **Precision**: bfloat16 for computation, float32 for similarity calculations
- **Context Window**: Up to model maximum (2048 for GPT-2, 4096+ for Llama)

### Training Configuration
- **Optimizer**: AdamW with learning rate $5 \times 10^{-4}$
- **Batch Size**: 4 trajectories per update
- **Gradient Clipping**: Norm clipped to 1.0
- **Episodes**: 10,000 total training episodes
- **Checkpoint Frequency**: Every 100 episodes

### Data Processing
- **Dataset**: Wikipedia 2022-03-01 English subset
- **Token Counts**: 10 tokens per key, 10 tokens per value
- **Trajectory Length**: Up to 10 key-value pairs per episode
- **Prefixes**: "Key: " and "Value: " for context building

---

## Key Mathematical Properties

### Convergence Properties

**Lemma 1 (Policy Improvement):** Under the chain rule policy gradient objective, the policy improvement follows standard REINFORCE convergence guarantees. The additional reward gradient term provides natural regularization without requiring explicit baseline estimation.

### Attention Mechanism Properties

**Lemma 2 (Multi-Head Averaging):** The averaging operation over attention heads in Equation (6) preserves the probability distribution property:

$$\sum_{k \in \mathcal{K}_t^{\text{available}}} \pi_\theta(k | s_t) = 1$$

### Chain Rule Properties

**Theorem 3 (Self-Consistent Learning):** The chain rule approach ensures self-consistent parameter updates:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau) + \nabla_\theta R_\theta(\tau)]$$

Both terms optimize the same parameters $\theta$, creating a unified learning signal where the model learns to both select high-reward actions AND accurately evaluate their outcomes.

---

## Conclusion

This mathematical formulation describes a sophisticated reinforcement learning system that combines:

1. **Vector-based policy**: Using attention layer embeddings for query generation
2. **Multi-head attention similarity**: Handling both MHA and GQA architectures  
3. **θ-dependent reward chain rule**: Proper gradient computation for model-dependent rewards
4. **Total return weighting**: All actions in successful trajectories receive positive gradients
5. **Self-consistent training**: Both policy and rewards computed using current model θ
6. **Self-supervised learning**: Where the model learns to sequence its own training data

The key innovation is the **θ-dependent reward formulation** with proper chain rule application:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau) + \nabla_\theta R_\theta(\tau) \right]$$

This creates two complementary learning signals:
- **Policy gradient term**: Encourages actions that led to high-reward trajectories
- **Reward gradient term**: Directly optimizes the model's ability to generate high rewards

**Advantages of the θ-Dependent Approach:**
- **Self-Consistent**: No mismatch between policy and reward model (both use current θ)
- **Natural Regularization**: Reward gradient term provides stability without artificial KL penalties
- **Mathematically Rigorous**: Proper chain rule application for θ-dependent objectives
- **Unified Learning**: Action selection and outcome evaluation optimized jointly
- **Memory Efficient**: No need for reference models or old model copies

The system represents a novel approach to active learning where the model's attention mechanisms directly drive the selection of training examples, creating a feedback loop between the model's internal representations and its learning curriculum.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv preprint arXiv:1707.06347.

2. Shazeer, N. (2019). *Fast transformer decoding: One write-head is all you need*. arXiv preprint arXiv:1911.02150.

3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv preprint arXiv:2106.09685.

---

*This document was automatically generated from the AttentionGuidedRL repository codebase analysis.* 