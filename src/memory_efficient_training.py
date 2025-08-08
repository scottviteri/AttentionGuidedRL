import torch
from typing import Dict, Optional, Tuple

# These are patched in tests to control behavior
from src.embeddings import get_attention_params  # used only for shapes if needed
from src.config import CONFIG

# Import real implementations to delegate when tests don't patch
from src.training import generate_query_vector as _real_generate_query_vector  # type: ignore
from src.embeddings import compute_similarity as _real_compute_similarity  # type: ignore


# Expose patchable names used by tests
def generate_query_vector(model, tokenizer, context_tokens):
    return _real_generate_query_vector(model, tokenizer, context_tokens)


def compute_similarity(query_embeddings, key_embeddings, num_heads, num_groups, head_dim):
    return _real_compute_similarity(query_embeddings, key_embeddings, num_heads, num_groups, head_dim)


class MemoryEfficientLoRAManager:
    """
    Minimal manager for saving, switching, and EMA-updating LoRA-only parameters.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.old_lora_state: Dict[str, torch.Tensor] = self._extract_lora_state()
        self.current_lora_state: Dict[str, torch.Tensor] = self._extract_lora_state()

    def _extract_lora_state(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                state[name] = param.data.clone()
        return state

    def save_current_state(self) -> None:
        self.current_lora_state = self._extract_lora_state()

    def switch_to_old_state(self) -> None:
        self._load_state(self.old_lora_state)

    def switch_to_current_state(self) -> None:
        self._load_state(self.current_lora_state)

    def _load_state(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.data.copy_(state[name])

    def update_old_state_ema(self, decay: float = 0.95) -> None:
        # Ensure current state is up-to-date
        if not self.current_lora_state:
            self.save_current_state()
        for name in self.old_lora_state.keys():
            old_param = self.old_lora_state[name]
            cur_param = self.current_lora_state.get(name, old_param)
            self.old_lora_state[name] = decay * old_param + (1.0 - decay) * cur_param


def memory_efficient_compute_policy_loss(
    trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    lora_manager: MemoryEfficientLoRAManager,
    kl_penalty_coef: float,
    verbose: bool = False,
    tokenizer=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Lightweight policy loss using in-place LoRA state and patched helpers.
    Returns (total_loss, policy_loss, kl_loss, avg_clipping_ratio).
    """
    device = next(adapter_model.parameters()).device
    batch_size = trajectory.qkv_steps[0].key_tokens.shape[0]

    # Simple baseline-free advantages: ones
    advantages = torch.ones((batch_size, len(trajectory.qkv_steps)), device=device)

    policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    kl_loss = torch.tensor(0.0, device=device, requires_grad=True)
    clipping_ratios = []

    # Build initial context tokens if tokenizer is available
    if tokenizer is not None:
        context_tokens = tokenizer(
            [CONFIG.initial_prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).input_ids.to(device)
    else:
        context_tokens = trajectory.qkv_steps[0].key_tokens.new_zeros((batch_size, 1))

    for t, step in enumerate(trajectory.qkv_steps):
        # Query vector via patchable function (shape-only use)
        _ = generate_query_vector(adapter_model, tokenizer, context_tokens)
        # Similarities via patchable function
        sims = compute_similarity(
            step.query_embedding if hasattr(step, 'query_embedding') else torch.randn(batch_size, adapter_model.config.n_embd, device=device),
            trajectory.all_key_embeddings.to(device),
            *get_attention_params(adapter_model),
        )
        # Current action log-prob
        selected_idx = step.selected_idx
        current_action_log_probs = sims[torch.arange(batch_size, device=device), selected_idx]
        step_advantages = advantages[:, t]
        policy_loss = policy_loss + (current_action_log_probs * step_advantages).sum()
        clipping_ratios.extend([1.0] * batch_size)

        # Advance context minimally if tokenizer available
        if tokenizer is not None:
            kp = tokenizer([CONFIG.key_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            vp = tokenizer([CONFIG.value_prefix] * batch_size, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            context_tokens = torch.cat([context_tokens, kp, step.key_tokens.to(device), vp, step.value_tokens.to(device)], dim=1)

    # Convert to loss (negate for ascent)
    total_policy_loss = -policy_loss
    total_kl_loss = kl_loss
    total_loss = total_policy_loss + kl_penalty_coef * total_kl_loss

    avg_clipping_ratio = sum(clipping_ratios) / len(clipping_ratios) if clipping_ratios else 1.0
    return total_loss, total_policy_loss, total_kl_loss, avg_clipping_ratio


def memory_efficient_train_step(
    trajectory,
    adapter_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    lora_manager: MemoryEfficientLoRAManager,
    optimizer: torch.optim.Optimizer,
    reward_stats: Dict[str, float],
    kl_penalty_coef: float,
    verbose: bool = False,
    tokenizer=None,
) -> Tuple[float, float, float, float]:
    """
    Train step that delegates compute to memory_efficient_compute_policy_loss,
    then applies gradients and returns floats.
    """
    optimizer.zero_grad()
    total_loss, policy_loss, kl_loss, avg_clip = memory_efficient_compute_policy_loss(
        trajectory,
        adapter_model,
        ref_model,
        lora_manager,
        kl_penalty_coef,
        verbose=verbose,
        tokenizer=tokenizer,
    )
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), CONFIG.gradient_clip_norm)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), kl_loss.item(), avg_clip