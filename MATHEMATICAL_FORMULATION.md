# Mathematical Formulation of Attention-Guided Reinforcement Learning

**AttentionGuidedRL Repository Mathematical Documentation**

---

## Abstract

This document provides a comprehensive mathematical formulation of the optimization objective implemented in the Attention-Guided Reinforcement Learning (AttentionGuidedRL) repository. The system trains a language model to autonomously guide its own training by sequencing key-value pairs from Wikipedia articles using reinforcement learning with policy gradients, multi-head attention mechanisms, and proximal policy optimization (PPO).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [Policy Definition](#policy-definition)
4. [Reward Function](#reward-function)
5. [Optimization Objective](#optimization-objective)
6. [Loss Function and Training Algorithm](#loss-function-and-training-algorithm)
7. [Training Algorithm](#training-algorithm)
8. [Implementation Details](#implementation-details)
9. [Key Mathematical Properties](#key-mathematical-properties)
10. [Conclusion](#conclusion)

---

## Introduction

The AttentionGuidedRL system implements a novel approach where a base language model (Llama-3.2-3B or GPT-2) learns to sequence its own training data through reinforcement learning. The model generates vector queries using attention mechanisms and selects key-value pairs to maximize learning progress, measured by conditional log probabilities of value tokens.

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

$$q_t = \text{AvgPool}(\text{Embed}_\theta(c_t \oplus [\text{"Query"}]))$$

where:
- $\text{Embed}_\theta(\cdot)$ extracts embeddings from the second-to-last attention layer
- $\oplus$ denotes sequence concatenation
- $\text{AvgPool}$ averages over the sequence dimension
- $q_t \in \mathbb{R}^{d_{\text{model}}}$ where $d_{\text{model}}$ is the model dimension

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

**Definition 7 (Step Reward):** The reward at step $t$ is defined as the conditional log probability of value tokens:

$$r_t = \begin{cases}
\log p_\theta(v_t | c_t, k_t) - \log p_{\text{ref}}(v_t | c_t, k_t) & \text{if SUBTRACT\_BASE\_MODEL\_LOGPROBS} \\
\log p_\theta(v_t | c_t, k_t) & \text{otherwise}
\end{cases}$$

where:
- $v_t$ are the value tokens at step $t$
- $k_t$ are the selected key tokens at step $t$
- $p_\theta(\cdot)$ is the adapter model probability
- $p_{\text{ref}}(\cdot)$ is the reference model probability (base model without LoRA)

**Definition 8 (Conditional Log Probability):** The conditional log probability is computed as:

$$\log p_\theta(v_t | c_t, k_t) = \frac{1}{|v_t|} \sum_{i=1}^{|v_t|} \log p_\theta(v_{t,i} | c_t, k_t, v_{t,<i})$$

where $v_{t,i}$ is the $i$-th token in the value sequence and $v_{t,<i}$ represents the preceding tokens.

---

## Optimization Objective

### Primary Objective Function

**Theorem 1 (Trajectory Sampling with Step-Level Credit Assignment):** The system maximizes the following objective function:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T A_t(\tau) \cdot \log \pi_\theta(a_t | s_t) \right] - \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$$

where:
- $\tau = (s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ is a complete trajectory sampled from policy $\pi_\theta$
- $A_t(\tau) = R_t(\tau) - b_t$ is the **trajectory-dependent advantage** at step $t$
- $R_t(\tau) = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ are the discounted returns from step $t$ forward in trajectory $\tau$
- $b_t = \mathbb{E}_{\text{batch}}[R_t(\tau)]$ is the GRPO baseline (batch average of returns at step $t$)
- $\gamma = 0.99$ is the discount factor
- $\beta = 0.1$ is the KL penalty coefficient
- $\pi_{\text{ref}}$ is the fixed reference policy (base model without LoRA)

**Key Insight:** We sample complete trajectories first, then assign step-specific credit. Each action $a_t$ receives weight $A_t(\tau)$ that depends on the **actual future rewards** observed in that trajectory, not a trajectory-average.

### Return and Advantage Computation

**Definition 9 (Discounted Returns):** The return at timestep $t$ is:

$$R_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$$

**Definition 10 (GRPO Advantage Estimation):** The system uses Group Relative Policy Optimization (GRPO) for advantage estimation:

$$A_t(\tau^{(i)}) = R_t(\tau^{(i)}) - b_t$$
$$b_t = \frac{1}{B} \sum_{i=1}^B R_t(\tau^{(i)})$$

where:
- $\tau^{(i)}$ is the $i$-th trajectory in the batch
- $b_t$ is the per-timestep baseline computed as the batch average of returns at step $t$
- $B$ is the batch size
- Each trajectory gets its own advantage based on its actual returns vs. the batch average

---

## Loss Function and Training Algorithm

### PPO Loss Implementation

**Theorem 2 (Trajectory-Level PPO with Step-Specific Credit Assignment):** The actual loss function implements PPO over complete trajectories with step-specific advantages:

$$\mathcal{L}(\theta) = \mathbb{E}_{\text{batch}} \left[ -\sum_{t=1}^T \min\left( \rho_t(\theta, \tau) \cdot A_t(\tau), \text{clip}(\rho_t(\theta, \tau), 1-\epsilon, 1+\epsilon) \cdot A_t(\tau) \right) \right] + \beta \cdot D_{KL}(\pi_\theta || \pi_{\text{ref}})$$

where:
- $\tau$ is a complete trajectory sampled from $\pi_\theta$
- $A_t(\tau) = R_t(\tau) - b_t$ is the **trajectory-dependent advantage** at step $t$
- $\rho_t(\theta, \tau) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)}$ is the probability ratio for action $a_t$ in trajectory $\tau$
- $\epsilon = 0.2$ is the PPO clipping parameter
- $\text{clip}(x, a, b) = \max(a, \min(x, b))$
- KL divergence is computed against the fixed reference policy $\pi_{\text{ref}}$

**Process Supervision Interpretation:** Each action $a_t$ in trajectory $\tau$ receives credit proportional to $A_t(\tau)$ - the actual rewards-to-go observed from that step forward in that specific trajectory, enabling precise temporal credit assignment.

### Step-Level Credit Assignment Details

**Definition 12 (Trajectory-Dependent Gradient Decomposition):** The policy gradient decomposes into trajectory-dependent step contributions:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T A_t(\tau) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

This formulation ensures:

1. **Proper Temporal Credit**: Action $a_1$ in trajectory $\tau$ affects actual rewards $\{r_1, r_2, \ldots, r_T\}$ observed in $\tau$, so $A_1(\tau) = \sum_{t'=1}^T \gamma^{t'-1} r_{t'} - b_1$
2. **Diminishing Influence**: Action $a_T$ in trajectory $\tau$ only affects the final reward $r_T$ in $\tau$, so $A_T(\tau) = r_T - b_T$  
3. **Trajectory-Specific Assessment**: Each action's contribution is evaluated against actual observed outcomes in that trajectory

**Comparison with Alternative Formulations:**

- **Trajectory-Average**: $\mathcal{L} = -\bar{A}(\tau) \sum_{t=1}^T \log \pi_\theta(a_t | s_t)$ where $\bar{A}(\tau) = \frac{1}{T}\sum_{t=1}^T A_t(\tau)$
- **Step-Specific** (our approach): $\mathcal{L} = -\sum_{t=1}^T A_t(\tau) \log \pi_\theta(a_t | s_t)$

**Key Difference:** In trajectory-average, all actions in a trajectory get the same gradient weight. In our step-specific approach, early actions naturally receive higher gradients when they lead to better future outcomes in that specific trajectory.

### Implementation Details: Math-Code Correspondence

**Code Implementation Overview:**
```python
# 1. Generate complete trajectories first
trajectory = generate_trajectory(adapter_model, ...)

# 2. Compute trajectory-dependent advantages 
advantages, _ = compute_advantages(trajectory.rewards, gamma=GAMMA)
# advantages[batch_idx, step_t] = A_t(τ^(batch_idx))

# 3. Apply step-specific PPO loss
for t, qkv_step in enumerate(trajectory.qkv_steps):
    step_advantages = advantages[:, t]  # A_t(τ) for each trajectory in batch
    
    # Compute probability ratios
    ratio = π_θ(a_t|s_t) / π_old(a_t|s_t)
    
    # PPO clipped objective with step-specific advantages
    unclipped = ratio * step_advantages  # ρ_t(θ,τ) * A_t(τ)
    clipped = clip(ratio, 1-ε, 1+ε) * step_advantages
    ppo_loss_t = -min(unclipped, clipped)
    
    total_loss += ppo_loss_t  # Accumulate across steps
```

**Mathematical Correspondence:**
- **Line `advantages[:, t]`** ↔ $A_t(\tau^{(i)})$ - trajectory-dependent advantage at step $t$
- **Line `ratio * step_advantages`** ↔ $\rho_t(\theta, \tau) \cdot A_t(\tau)$ - weighted by trajectory-specific advantage
- **Loop accumulation** ↔ $\sum_{t=1}^T$ in the loss function
- **Batch processing** ↔ $\mathbb{E}_{\text{batch}}[\cdot]$ expectation over trajectories

**Key Insight:** The code samples complete trajectories, computes their actual returns, then assigns credit to each action based on what actually happened in that trajectory - not on average expectations.

### KL Divergence Regularization

**Definition 11 (KL Divergence Term):** The KL divergence regularization is computed as:

$$D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \sum_{t=1}^T \sum_{k \in \mathcal{K}_t^{\text{available}}} \pi_\theta(k|s_t) \log \frac{\pi_\theta(k|s_t)}{\pi_{\text{ref}}(k|s_t)}$$

This is implemented using PyTorch's `F.kl_div` with `log_target=True` for numerical stability. The reference policy $\pi_{\text{ref}}$ is the fixed base model without LoRA, providing a stable regularization target.

---

## Training Algorithm

**Algorithm 1: Attention-Guided RL Training**

```
Input: Base model, tokenizer, Wikipedia dataset
Input: Hyperparameters γ, β, ε, learning rate, batch size

1. Initialize LoRA adapter parameters θ
2. Initialize reference model π_ref (base model without LoRA, fixed)
3. Initialize old model π_old ← π_θ
4. Initialize optimizer (AdamW with learning rate 5×10⁻⁴)

5. For episode = 1 to NUM_EPISODES:
   a. Sample Wikipedia article and extract key-value pairs
   b. Generate trajectory τ using current policy π_θ
   c. Compute rewards {r_t}ᵀₜ₌₁ using Equation (7)
   d. Compute returns {R_t}ᵀₜ₌₁ using Equation (9)
   e. Compute advantages {A_t}ᵀₜ₌₁ using GRPO (Equations 10-11)
   f. Compute PPO loss ℒ(θ) using Equation (12) with KL vs π_ref
   g. Update parameters: θ ← θ - ∇_θ ℒ(θ)
   h. If episode mod BASELINE_UPDATE_FREQUENCY == 0:
      Update old model: π_old ← π_θ (for PPO ratios only)
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

**Lemma 1 (Policy Improvement):** Under the PPO clipped objective with appropriate clipping parameter $\epsilon$, the policy improvement is guaranteed to be conservative, preventing destructively large updates.

### Attention Mechanism Properties

**Lemma 2 (Multi-Head Averaging):** The averaging operation over attention heads in Equation (6) preserves the probability distribution property:

$$\sum_{k \in \mathcal{K}_t^{\text{available}}} \pi_\theta(k | s_t) = 1$$

### GRPO Baseline Properties

**Theorem 3 (Unbiased Advantage Estimation):** The GRPO baseline provides unbiased gradient estimates:

$$\mathbb{E}[A_t] = 0 \quad \text{for each timestep } t$$

This property holds by construction since $A_t = R_t - \mathbb{E}[R_t]$ where the expectation is over the batch.

---

## Conclusion

This mathematical formulation describes a sophisticated reinforcement learning system that combines:

1. **Vector-based policy**: Using attention layer embeddings for query generation
2. **Multi-head attention similarity**: Handling both MHA and GQA architectures
3. **Step-level PPO optimization**: With proper temporal credit assignment through step-specific advantages
4. **GRPO advantage estimation**: For variance reduction without additional parameters
5. **Process supervision**: Where each action receives credit proportional to its impact on future outcomes
6. **Self-supervised learning**: Where the model learns to sequence its own training data

The key innovation is the **step-level formulation** that ensures proper credit assignment: early key selections (which affect more future rewards) naturally receive higher gradient magnitudes than later selections. This creates a natural curriculum where the model learns to prioritize the quality of early decisions while still optimizing the entire sequence.

The system represents a novel approach to active learning where the model's attention mechanisms directly drive the selection of training examples, creating a feedback loop between the model's internal representations and its learning curriculum.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms*. arXiv preprint arXiv:1707.06347.

2. Shazeer, N. (2019). *Fast transformer decoding: One write-head is all you need*. arXiv preprint arXiv:1911.02150.

3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv preprint arXiv:2106.09685.

---

*This document was automatically generated from the AttentionGuidedRL repository codebase analysis.* 