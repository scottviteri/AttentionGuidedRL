"""
Embeddings module for the Attention-Guided RL project.

Contains functions for extracting embeddings from the model's last attention layer
and computing similarities between queries and keys using grouped query attention.
"""

import math
from typing import Dict, List, Tuple, Optional, Callable, Union

import torch
import torch.nn.functional as F

from src.config import DEVICE, MODEL_TYPE


def register_embedding_hook(
    model, 
    embed_type: str = "query"
) -> Tuple[Dict, Callable]:
    """
    Register a hook to extract embeddings from the last attention layer.
    
    Args:
        model: The language model to extract embeddings from
        embed_type: Type of embedding to extract, either "query" or "key"
        
    Returns:
        Tuple[Dict, Callable]: A dictionary to store the embeddings and a function to remove the hook
    """
    # Dictionary to store the embeddings
    embeddings_dict = {"embeddings": None}
    
    if MODEL_TYPE == "llama":
        return register_llama_embedding_hook(model, embeddings_dict, embed_type)
    elif MODEL_TYPE == "gpt2":
        # For GPT-2, we'll need to handle this differently as it uses a combined QKV projection
        return register_gpt2_embedding_hook(model, embeddings_dict, embed_type)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def register_llama_embedding_hook(
    model, 
    embeddings_dict: Dict,
    embed_type: str = "query"
) -> Tuple[Dict, Callable]:
    """
    Register a hook for Llama models.
    
    Args:
        model: The Llama language model
        embeddings_dict: Dictionary to store embeddings
        embed_type: Type of embedding to extract, either "query" or "key"
        
    Returns:
        Tuple[Dict, Callable]: The embeddings dictionary and hook removal function
    """
    # Get the target module based on embed_type
    if embed_type.lower() == "query":
        target_module = model.model.model.layers[-1].self_attn.q_proj
    elif embed_type.lower() == "key":
        target_module = model.model.model.layers[-1].self_attn.k_proj
    else:
        raise ValueError(f"Unsupported embed_type: {embed_type}. Must be 'query' or 'key'")
    
    # Define the hook function
    def hook_fn(module, input_tensor, output_tensor):
        # Store the output tensor since we want the embeddings after projection
        embeddings_dict["embeddings"] = output_tensor.detach()
    
    # Register the hook
    hook = target_module.register_forward_hook(hook_fn)
    
    # Return the dictionary and a function to remove the hook
    return embeddings_dict, hook.remove


def register_gpt2_embedding_hook(
    model, 
    embeddings_dict: Dict,
    embed_type: str = "query"
) -> Tuple[Dict, Callable]:
    """
    Register a hook for GPT-2 models.
    
    Args:
        model: The GPT-2 language model
        embeddings_dict: Dictionary to store embeddings
        embed_type: Type of embedding to extract ("query" or "key")
        
    Returns:
        Tuple[Dict, Callable]: The embeddings dictionary and hook removal function
    """
    # Get the target module (last attention layer's combined QKV projection)
    target_module = model.transformer.h[-1].attn.c_attn
    
    # Define the hook function
    def hook_fn(module, input_tensor, output_tensor):
        # Get the hidden size and split size for Q,K,V
        hidden_size = input_tensor[0].shape[-1]
        split_size = hidden_size  # Each of Q,K,V has same size as input
        
        # output_tensor contains concatenated Q,K,V projections
        # Shape: [batch_size, seq_len, 3 * hidden_size]
        if embed_type.lower() == "query":
            # Extract query portion (first third)
            embeddings_dict["embeddings"] = output_tensor[..., :split_size].detach()
        elif embed_type.lower() == "key":
            # Extract key portion (middle third)
            embeddings_dict["embeddings"] = output_tensor[..., split_size:2*split_size].detach()
        else:
            raise ValueError(f"Unsupported embed_type: {embed_type}. Must be 'query' or 'key'")
    
    # Register the hook
    hook = target_module.register_forward_hook(hook_fn)
    
    # Return the dictionary and a function to remove the hook
    return embeddings_dict, hook.remove


def extract_embeddings(
    model: torch.nn.Module,
    tokenized_input: torch.Tensor,  # Shape: [batch, seq_len]
    embeddings_dict: Dict
) -> torch.Tensor:  # Shape: [batch, head_dim*num_heads]
    """
    Extract embeddings from the model's last attention layer and average over the sequence length 
    to produce a single embedding vector per sentence.
    
    Args:
        model: The language model
        tokenized_input: Tokenized input tensor with shape [batch, seq_len]
        embeddings_dict: Dictionary to store the embeddings from the hook
        
    Returns:
        Extracted embeddings with shape [batch, head_dim*num_heads]
    """
    # Ensure input is on the same device as model
    device = next(model.parameters()).device
    tokenized_input = tokenized_input.to(device)
    
    # Run forward pass to trigger the hook
    with torch.no_grad():
        model(tokenized_input)
    
    # Get the embeddings from the hook
    full_embeddings = embeddings_dict["embeddings"]
    
    # Average over the sequence length
    avg_embeddings = torch.mean(full_embeddings, dim=1)  # [batch_size, hidden_size]
    
    return avg_embeddings


def get_attention_params(model) -> Tuple[int, int, int]:
    """
    Get the attention parameters from the model.
    
    Args:
        model: The language model
        
    Returns:
        Tuple[int, int, int]: Number of query heads, number of KV groups, and head dimension
    """
    if MODEL_TYPE == "llama":
        return get_llama_attention_params(model)
    elif MODEL_TYPE == "gpt2":
        return get_gpt2_attention_params(model)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def get_llama_attention_params(model) -> Tuple[int, int, int]:
    """
    Get the grouped query attention parameters from a Llama model.
    
    Args:
        model: The Llama language model
        
    Returns:
        Tuple[int, int, int]: Number of query heads, number of KV groups, and head dimension
    """
    # Get the first layer to extract attention parameters
    first_layer = model.model.model.layers[0].self_attn
    
    # Get the number of heads and groups
    num_query_heads = first_layer.num_heads
    num_kv_groups = first_layer.num_key_value_heads
    
    # Determine the total embedding dimension and head dimension
    embedding_dim = first_layer.hidden_size
    head_dim = embedding_dim // num_query_heads
    
    return num_query_heads, num_kv_groups, head_dim


def get_gpt2_attention_params(model) -> Tuple[int, int, int]:
    """
    Get the attention parameters from a GPT-2 model.
    
    Args:
        model: The GPT-2 language model
        
    Returns:
        Tuple[int, int, int]: Number of heads, number of KV groups, and head dimension
    """
    # Get the configuration to extract attention parameters
    config = model.config
    
    # GPT-2 doesn't use grouped query attention
    num_heads = config.n_head
    num_kv_groups = num_heads  # In GPT-2, every head has its own key/value
    
    # Determine the embedding dimension and head dimension
    embedding_dim = config.n_embd
    head_dim = embedding_dim // num_heads
    
    return num_heads, num_kv_groups, head_dim


def compute_similarity(
    query_embeddings: torch.Tensor,  # Shape: [batch, hidden_size]
    key_embeddings: torch.Tensor,  # Shape: [batch, num_keys, hidden_size]
    model,
    temperature: float = 1.0
) -> torch.Tensor:  # Shape: [batch, num_keys]
    """
    Compute similarity between query and key embeddings using a fully batched approach
    with zero explicit loops or list comprehensions.
    
    This implementation correctly handles both standard Multi-Head Attention (MHA)
    and Grouped Query Attention (GQA) by applying softmax per head:
    
    - In MHA: Each query head attends to its corresponding key head (1:1 mapping)
    - In GQA: Multiple query heads attend to the same key head (N:1 mapping)
    
    Args:
        query_embeddings: Query embeddings [batch, hidden_size]
        key_embeddings: Key embeddings [batch, num_keys, hidden_size]
        model: The language model (used to get attention parameters)
        temperature: Temperature parameter for softmax scaling. Higher temperature makes
                    the distribution more uniform, lower temperature makes it more peaked.
        
    Returns:
        Similarity scores as probabilities [batch, num_keys]
    """
    batch_size = query_embeddings.shape[0]
    num_keys = key_embeddings.shape[1]
    
    # Get attention parameters
    num_heads, num_groups, head_dim = get_attention_params(model)
    
    # Reshape query embeddings to separate head dimensions
    # [batch, hidden_size] -> [batch, num_heads, head_dim]
    query_reshaped = query_embeddings.view(batch_size, num_heads, head_dim)
    
    # For key embeddings, the provided tensor has dimensions [batch, num_keys, hidden_size]
    # where hidden_size = num_heads * head_dim
    
    # In GQA, the model only produces num_groups key vectors (where num_groups < num_heads)
    # So we need to reshape the key embeddings to use only the first num_groups*head_dim dimensions
    key_group_dim = num_groups * head_dim
    
    # Only use the first key_group_dim dimensions of the hidden_size dimension
    # in case num_groups < num_heads (as in GQA)
    key_embeddings_truncated = key_embeddings[:, :, :key_group_dim]
    
    # Reshape to [batch, num_keys, num_groups, head_dim]
    key_reshaped = key_embeddings_truncated.view(batch_size, num_keys, num_groups, head_dim)
    
    # Create a mapping from heads to groups
    # For each head h, we get its corresponding group: h // (num_heads // num_groups)
    head_to_group = torch.div(
        torch.arange(num_heads, device=query_embeddings.device),
        num_heads // num_groups,
        rounding_mode='floor'
    )
    
    # Using torch.index_select for a fully vectorized implementation with no loops or comprehensions
    
    # First, reshape keys to prepare for index_select
    # [batch, num_keys, num_groups, head_dim] -> [batch*num_keys, num_groups, head_dim]
    key_reshaped_flat = key_reshaped.reshape(-1, num_groups, head_dim)
    
    # Use index_select to gather the right groups for all heads at once
    # [batch*num_keys, num_groups, head_dim] + head_to_group -> [batch*num_keys, num_heads, head_dim]
    # Here we select from the group dimension (dim=1) using head_to_group indices
    # For each head index, we select its corresponding group index from head_to_group
    # This eliminates the need for any loops or list comprehensions
    key_groups_selected = torch.index_select(
        key_reshaped_flat, 
        dim=1,
        index=head_to_group
    )
    
    # Reshape back to include batch and num_keys dimensions
    # [batch*num_keys, num_heads, head_dim] -> [batch, num_keys, num_heads, head_dim]
    key_groups_batched = key_groups_selected.view(batch_size, num_keys, num_heads, head_dim)
    
    # Transpose to get [batch, num_heads, num_keys, head_dim]
    all_key_groups = key_groups_batched.permute(0, 2, 1, 3)
    
    # Reshape query for batched computation: [batch, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
    query_expanded = query_reshaped.unsqueeze(2)
    
    # Now compute dot products for all heads in parallel
    # [batch, num_heads, 1, head_dim] × [batch, num_heads, num_keys, head_dim] -> [batch, num_heads, 1, num_keys]
    # Using einsum for clearer semantics and batch matrix multiplication
    similarities = torch.einsum(
        'bhad,bhkd->bhak', 
        query_expanded, 
        all_key_groups
    ) / math.sqrt(head_dim)
    
    # Remove the extra dimension [batch, num_heads, 1, num_keys] -> [batch, num_heads, num_keys]
    similarities = similarities.squeeze(2)
    
    # Apply temperature scaling
    scaled_similarities = similarities / temperature
    
    # Apply softmax per head [batch, num_heads, num_keys]
    head_probabilities = F.softmax(scaled_similarities, dim=2)
    
    # Average over heads to get final probabilities [batch, num_keys]
    probabilities = torch.mean(head_probabilities, dim=1)
    
    return probabilities


def sample_key_value(
    similarity_scores: torch.Tensor,  # Shape: [batch, num_keys]
    available_keys: List[List[int]],
    batch_size: int
) -> Tuple[List[int], torch.Tensor]:  # Returns: Tuple[List[int], Tensor[batch]]
    """
    Sample key indices based on similarity scores using batched operations.
    
    Args:
        similarity_scores: Similarity scores [batch, num_keys]
        available_keys: List of available key indices for each batch item
        batch_size: Batch size
        
    Returns:
        Tuple[List[int], torch.Tensor]: Sampled key indices and their probabilities
        
    Raises:
        ValueError: If any batch item has no available keys
    """
    # Check for empty available keys
    for b, keys in enumerate(available_keys):
        if not keys:
            raise ValueError(f"Batch item {b} has no available keys to sample from")
    
    # Create a mask of available keys for each batch (1 for available, -inf for unavailable)
    num_keys = similarity_scores.shape[1]
    device = similarity_scores.device
    
    # Initialize mask with -inf everywhere
    key_mask = torch.full((batch_size, num_keys), float('-inf'), device=device)
    
    # Set available keys to 0 in the mask (will be added to similarity scores)
    for b in range(batch_size):
        key_mask[b, available_keys[b]] = 0.0
    
    # Apply mask to similarity scores (adds 0 to available keys, -inf to unavailable)
    masked_scores = similarity_scores + key_mask
    
    # Create categorical distributions for each batch element
    distributions = torch.distributions.Categorical(logits=masked_scores)
    
    # Sample from all distributions at once
    sampled_indices_tensor = distributions.sample()
    
    # Convert to Python list for compatibility with previous version
    sampled_indices = sampled_indices_tensor.tolist()
    
    # Get probabilities for the sampled indices
    sampled_probs = torch.zeros(batch_size, device=device)
    for b in range(batch_size):
        sampled_probs[b] = similarity_scores[b, sampled_indices[b]]
    
    return sampled_indices, sampled_probs 