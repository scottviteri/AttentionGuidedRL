# Recommended Plotting Improvements for Chain Rule Implementation

## 1. Mathematical Notation Updates

### Current vs. Recommended Plot Labels:

| Current Plot Title | Recommended Title | Mathematical Symbol |
|-------------------|-------------------|-------------------|
| "Policy Loss" | "Policy Gradient Term" | $R_\theta(\tau) \cdot \nabla\log\pi_\theta(\tau)$ |
| "Average Advantage" | "GRPO Advantages (Monitoring Only)" | $A_t = R_t - b_t$ |
| "Loss Components" | "Chain Rule Loss Components" | $-(R_\theta(\tau) \cdot \nabla\log\pi_\theta(\tau) + \nabla R_\theta(\tau))$ |
| "Model Log Probabilities" | "θ-Dependent Rewards per Step" | $r_{\theta,t} = \log p_\theta(v_t \| c_t, k_t)$ |

## 2. New Metrics to Add

### A. Chain Rule Component Variance
**Purpose**: Track empirical variance of your gradient estimator
**Metrics to track**:
- `policy_term_variance`: Variance of $R_\theta(\tau) \cdot \nabla\log\pi_\theta(a_t\|s_t)$ across batch
- `reward_term_variance`: Variance of $\nabla R_\theta(\tau)$ across batch  
- `total_return_variance`: Variance of $R_\theta(\tau)$ across batch

### B. Total Return Statistics
**Purpose**: Better understand trajectory-level performance
**Metrics to track**:
- `total_returns_mean`: Mean of $R_\theta(\tau)$
- `total_returns_std`: Standard deviation of $R_\theta(\tau)$
- `total_returns_percentiles`: [25th, 50th, 75th percentiles]

### C. Reward Gradient Magnitude
**Purpose**: Monitor the second chain rule term
**Metrics to track**:
- `reward_gradient_norm`: $\|\nabla_\theta R_\theta(\tau)\|$
- `policy_reward_ratio`: $\frac{\|R_\theta(\tau) \cdot \nabla\log\pi_\theta(\tau)\|}{\|\nabla R_\theta(\tau)\|}$

## 3. Plot Layout Improvements

### Replace Current Plots:
1. **"Advantage Statistics"** → **"Chain Rule Component Analysis"**
   - Plot both policy term and reward term magnitudes
   - Show their ratio over time
   - Add variance bands

2. **"Model Log Probabilities"** → **"θ-Dependent Reward Evolution"**
   - Show per-step rewards $r_{\theta,t}$
   - Add confidence intervals
   - Highlight step-wise improvement

3. **Add New Plot: "Gradient Estimator Variance"**
   - Policy term variance vs reward term variance
   - Total gradient variance over time
   - Compare to theoretical variance bounds

## 4. Notation Consistency

### Use Mathematical Symbols in Legends:
- $R_\theta(\tau)$ instead of "Total Return"
- $\nabla\log\pi_\theta(a_t\|s_t)$ instead of "Policy Gradient"  
- $r_{\theta,t}$ instead of "Step Reward"
- $A_t$ instead of "Advantage"

## 5. New Analytical Plots

### A. Chain Rule Convergence Analysis
**Track theoretical properties**:
- Policy-reward correlation over time
- Self-consistency metric: $\text{corr}(R_\theta(\tau), \nabla R_\theta(\tau))$
- Gradient alignment: $\cos(\nabla_{\text{policy}}, \nabla_{\text{reward}})$

### B. Variance Decomposition
**Break down total variance**:
- Variance from policy term
- Variance from reward term  
- Cross-covariance term
- Compare to GRPO baseline variance

## 6. Implementation Priority

### High Priority (Immediate):
1. Update plot titles to use mathematical notation
2. Add total return statistics
3. Add chain rule component variance tracking

### Medium Priority:
1. Replace advantage plot with chain rule analysis
2. Add gradient estimator variance plot
3. Implement variance decomposition

### Low Priority (Nice to have):
1. Add self-consistency metrics
2. Implement theoretical variance bounds
3. Add gradient alignment analysis

## 7. Code Changes Needed

### In `src/plotting.py` - Add new fields:
```python
@dataclass(frozen=True)
class PlotData:
    # New chain rule specific metrics
    policy_term_values: List[float] = field(default_factory=list)
    reward_term_values: List[float] = field(default_factory=list)
    total_returns_mean: List[float] = field(default_factory=list)
    total_returns_std: List[float] = field(default_factory=list)
    policy_term_variance: List[float] = field(default_factory=list)
    reward_term_variance: List[float] = field(default_factory=list)
    reward_gradient_norm: List[float] = field(default_factory=list)
    policy_reward_ratio: List[float] = field(default_factory=list)
```

### In `src/main.py` - Track new metrics:
```python
# In policy_gradient_train_step(), compute and track:
total_returns_batch = trajectory.rewards.sum(dim=1)
policy_term_variance = (total_returns_batch * current_action_log_probs).var().item()
reward_term_variance = step_reward.var().item()
reward_gradient_norm = torch.norm(reward_term).item()
```

This will give you much better insight into the empirical properties of your chain rule estimator!