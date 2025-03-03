"""
Data handling module for the Attention-Guided RL project.

Contains data structures and utilities for loading, processing, and batching data.
"""
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Callable, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import random

from src.config import (
    TOKENIZER_NAME,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS,
    KV_SUBSET_FRACTION,
    QUERY_PREFIX,
    RESPONSE_PREFIX,
    DEVICE
)


@dataclass
class KeyValuePair:
    """
    Dataclass for a key-value pair optimized for batched processing.
    
    Attributes:
        key_tokens: Tokenized keys [batch_size, TOKENS_PER_KEY]
        value_tokens: Tokenized values [batch_size, TOKENS_PER_VALUE]
        key_embedding: Precomputed embeddings for keys [batch_size, embedding_dim]
        key_text: Original text of keys (for logging/debugging)
        value_text: Original text of values (for logging/debugging)
    """
    key_tokens: torch.Tensor  # Shape: [batch_size, TOKENS_PER_KEY]
    value_tokens: torch.Tensor  # Shape: [batch_size, TOKENS_PER_VALUE]
    key_embedding: torch.Tensor  # Shape: [batch_size, embedding_dim]
    key_text: List[str]  # For logging and debugging
    value_text: List[str]  # For logging and debugging
    
    def __post_init__(self):
        """Validate tensor shapes and types."""
        batch_size = self.key_tokens.shape[0]
        
        assert isinstance(self.key_tokens, torch.Tensor), "key_tokens must be a tensor"
        assert isinstance(self.value_tokens, torch.Tensor), "value_tokens must be a tensor"
        assert isinstance(self.key_embedding, torch.Tensor), "key_embedding must be a tensor"
        assert isinstance(self.key_text, list), "key_text must be a list"
        assert isinstance(self.value_text, list), "value_text must be a list"
        
        assert self.key_tokens.shape == (batch_size, TOKENS_PER_KEY), f"key_tokens shape should be ({batch_size}, {TOKENS_PER_KEY})"
        assert self.value_tokens.shape == (batch_size, TOKENS_PER_VALUE), f"value_tokens shape should be ({batch_size}, {TOKENS_PER_VALUE})"
        assert self.key_embedding.shape[0] == batch_size, f"key_embedding first dimension should be {batch_size}"
        assert len(self.key_text) == batch_size, f"key_text length should be {batch_size}"
        assert len(self.value_text) == batch_size, f"value_text length should be {batch_size}"


def get_tokenizer() -> PreTrainedTokenizer:
    """
    Get the tokenizer for the model.
    
    Returns:
        PreTrainedTokenizer: The tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer


def tokenize_text(text: str or List[str], tokenizer: PreTrainedTokenizer) -> List[int] or List[List[int]]:
    """
    Tokenize text into token IDs.
    
    Args:
        text: The text to tokenize or a list of texts
        tokenizer: The tokenizer to use
        
    Returns:
        List[int] or List[List[int]]: The token IDs
    """
    # Handle both single strings and lists of strings
    if isinstance(text, str):
        return tokenizer.encode(text, add_special_tokens=False)
    else:
        # Batch tokenization
        encoding = tokenizer(text, add_special_tokens=False, padding=False, truncation=True)
        return encoding['input_ids']


def format_prompt_with_kv_pairs(pairs: List[Tuple[str, str]]) -> str:
    """
    Format key-value pairs into a prompt.
    
    Args:
        pairs: The key-value pairs to format
        
    Returns:
        str: The formatted prompt
    """
    prompt = ""
    for key, value in pairs:
        prompt += f"{QUERY_PREFIX}{key}{RESPONSE_PREFIX}{value}"
    
    return prompt


def iter_wikipedia_articles() -> Iterator[Dict]:
    """
    Create an iterator that yields Wikipedia articles.
    
    Returns:
        Iterator[Dict]: Iterator yielding article dictionaries
    """
    # Use streaming mode to avoid loading the entire dataset into memory
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    for article in dataset:
        yield article


def filter_articles_by_length(tokenizer: PreTrainedTokenizer) -> Iterator[Dict]:
    """
    Filter Wikipedia articles by length.
    
    Args:
        tokenizer: The tokenizer to use for length calculation
        
    Returns:
        Iterator[Dict]: Iterator yielding articles that meet the length requirement
    """
    min_tokens = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS
    
    for article in iter_wikipedia_articles():
        text = article["text"]
        tokens = tokenize_text(text, tokenizer)
        
        if len(tokens) >= min_tokens:
            yield article


def iter_key_value_pairs(batch_size: int = 1, embedding_fn=None) -> Iterator[Tuple[KeyValuePair, List[Dict]]]:
    """
    Create an iterator that yields batches of key-value pairs.
    
    Args:
        batch_size: Number of articles to process in each batch
        embedding_fn: Optional function to compute embeddings
        
    Returns:
        Iterator[Tuple[KeyValuePair, List[Dict]]]: Iterator yielding a batched KeyValuePair object and the source articles
    """
    tokenizer = get_tokenizer()
    
    while True:
        # Collect batch_size number of suitable articles
        article_batch = []
        all_tokens = []
        
        for article in filter_articles_by_length(tokenizer):
            # Tokenize article text in place
            tokens = tokenize_text(article["text"], tokenizer)
            
            # Only include articles with enough tokens
            tokens_needed = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS
            if len(tokens) >= tokens_needed:
                article_batch.append(article)
                all_tokens.append(tokens)
                
                if len(article_batch) >= batch_size:
                    break
        
        # If we couldn't collect enough articles, exit
        if not article_batch:
            break
            
        # Process the batch of articles
        all_key_tokens_list = []
        all_value_tokens_list = []
        all_key_text_list = []
        all_value_text_list = []
        all_article_indices = []  # To track which article each pair belongs to
        
        for article_idx, (article, tokens) in enumerate(zip(article_batch, all_tokens)):
            # Create pairs using index arithmetic - divide tokens into equal chunks
            chunk_size = TOKENS_PER_KEY + TOKENS_PER_VALUE
            
            # Only take complete key-value pairs
            num_complete_pairs = min(NUM_KV_PAIRS, len(tokens) // chunk_size)
            
            # Extract key and value tokens
            for j in range(num_complete_pairs):
                start_idx = j * chunk_size
                key_end_idx = start_idx + TOKENS_PER_KEY
                value_end_idx = key_end_idx + TOKENS_PER_VALUE
                
                # Extract key and value tokens
                key_tokens_seq = tokens[start_idx:key_end_idx]
                value_tokens_seq = tokens[key_end_idx:value_end_idx]
                
                # Decode to get original text (for logging)
                key_text = tokenizer.decode(key_tokens_seq)
                value_text = tokenizer.decode(value_tokens_seq)
                
                all_key_text_list.append(key_text)
                all_value_text_list.append(value_text)
                all_key_tokens_list.append(key_tokens_seq)
                all_value_tokens_list.append(value_tokens_seq)
                all_article_indices.append(article_idx)
            
            # Sample a subset if needed
            if KV_SUBSET_FRACTION < 1.0:
                # Identify pairs belonging to this article
                article_pair_indices = [i for i, idx in enumerate(all_article_indices) if idx == article_idx]
                subset_size = max(1, int(len(article_pair_indices) * KV_SUBSET_FRACTION))
                
                # If we need to sample, choose indices to keep
                if subset_size < len(article_pair_indices):
                    # Randomly select indices to keep
                    keep_indices = set(random.sample(article_pair_indices, subset_size))
                    
                    # Filter pairs not in the keep set
                    all_key_text_list = [x for i, x in enumerate(all_key_text_list) if i in keep_indices or all_article_indices[i] != article_idx]
                    all_value_text_list = [x for i, x in enumerate(all_value_text_list) if i in keep_indices or all_article_indices[i] != article_idx]
                    all_key_tokens_list = [x for i, x in enumerate(all_key_tokens_list) if i in keep_indices or all_article_indices[i] != article_idx]
                    all_value_tokens_list = [x for i, x in enumerate(all_value_tokens_list) if i in keep_indices or all_article_indices[i] != article_idx]
                    all_article_indices = [x for i, x in enumerate(all_article_indices) if i in keep_indices or all_article_indices[i] != article_idx]
        
        # Convert lists to tensors
        total_pairs = len(all_key_tokens_list)
        key_tokens_tensor = torch.zeros((total_pairs, TOKENS_PER_KEY), dtype=torch.long, device=DEVICE)
        value_tokens_tensor = torch.zeros((total_pairs, TOKENS_PER_VALUE), dtype=torch.long, device=DEVICE)
        
        for k, (key_seq, value_seq) in enumerate(zip(all_key_tokens_list, all_value_tokens_list)):
            key_tokens_tensor[k, :len(key_seq)] = torch.tensor(key_seq, dtype=torch.long, device=DEVICE)
            value_tokens_tensor[k, :len(value_seq)] = torch.tensor(value_seq, dtype=torch.long, device=DEVICE)
        
        # Generate embeddings if a function is provided
        if embedding_fn is not None:
            key_embedding = embedding_fn(key_tokens_tensor)
        else:
            # Create dummy embeddings
            embedding_dim = 768  # Default embedding dimension
            key_embedding = torch.zeros((total_pairs, embedding_dim), device=DEVICE)
        
        # Create a single KeyValuePair containing all pairs
        kv_pair = KeyValuePair(
            key_tokens=key_tokens_tensor,
            value_tokens=value_tokens_tensor,
            key_embedding=key_embedding,
            key_text=all_key_text_list,
            value_text=all_value_text_list
        )
        
        # Include the article_indices for downstream processing
        yield kv_pair, article_batch 