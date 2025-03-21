"""
Data handling module for the Attention-Guided RL project.

Contains data structures and utilities for loading, processing, and batching data.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import (
    TOKENIZER_NAME,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS,
    QUERY_PREFIX,
    RESPONSE_PREFIX,
    DEVICE,
    KV_EVERY_N,
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
        assert isinstance(
            self.value_tokens, torch.Tensor
        ), "value_tokens must be a tensor"
        assert isinstance(
            self.key_embedding, torch.Tensor
        ), "key_embedding must be a tensor"
        assert isinstance(self.key_text, list), "key_text must be a list"
        assert isinstance(self.value_text, list), "value_text must be a list"

        assert self.key_tokens.shape == (
            batch_size,
            TOKENS_PER_KEY,
        ), f"key_tokens shape should be ({batch_size}, {TOKENS_PER_KEY})"
        assert self.value_tokens.shape == (
            batch_size,
            TOKENS_PER_VALUE,
        ), f"value_tokens shape should be ({batch_size}, {TOKENS_PER_VALUE})"
        assert (
            self.key_embedding.shape[0] == batch_size
        ), f"key_embedding first dimension should be {batch_size}"
        assert (
            len(self.key_text) == batch_size
        ), f"key_text length should be {batch_size}"
        assert (
            len(self.value_text) == batch_size
        ), f"value_text length should be {batch_size}"


def get_tokenizer() -> PreTrainedTokenizer:
    """
    Get the tokenizer for the model.

    Returns:
        PreTrainedTokenizer: The tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    return tokenizer


def tokenize_text(
    text: Union[str, List[str]], tokenizer: PreTrainedTokenizer
) -> Union[List[int], List[List[int]]]:
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
        encoding = tokenizer(
            text, add_special_tokens=False, padding=False, truncation=True
        )
        return encoding["input_ids"]


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
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    for article in dataset:
        yield article


def filter_articles_by_length(tokenizer: PreTrainedTokenizer) -> Iterator[Dict]:
    """
    Filter Wikipedia articles by length.

    Args:
        tokenizer: The tokenizer to use for length calculation

    Returns:
        Iterator[Dict]: Iterator yielding articles that meet the minimum token requirement.
    """
    # Compute the required minimum length based on the desired number of key-value pairs
    # and the stride KV_EVERY_N
    max_len = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS * KV_EVERY_N

    for article in iter_wikipedia_articles():
        text = article["text"]
        tokens = tokenize_text(text, tokenizer)

        if len(tokens) >= max_len:
            yield article


def iter_key_value_pairs(
    batch_size: int = 1, embedding_fn=None
) -> Iterator[KeyValuePair]:
    """
    Create an iterator that yields batches of key-value pairs.

    Args:
        batch_size: Number of articles to process in each batch
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KeyValuePair]: Iterator yielding a batched KeyValuePair object
    """
    tokenizer = get_tokenizer()

    while True:
        # Collect batch_size number of suitable articles
        article_batch = []
        for article in filter_articles_by_length(tokenizer):
            article_batch.append(article)
            if len(article_batch) >= batch_size:
                break

        # Only yield full batches
        if len(article_batch) < batch_size:
            break

        # Ensure we have exactly batch_size articles
        assert len(article_batch) == batch_size, f"Expected batch size {batch_size}, got {len(article_batch)}"
        # Determine the fixed token length we require for each article
        # (we only need the first max_len tokens)
        max_len = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS * KV_EVERY_N
        chunk_size = TOKENS_PER_KEY + TOKENS_PER_VALUE

        # Batch tokenize the article texts to a fixed length tensor (truncating if necessary)
        article_texts = [article["text"] for article in article_batch]
        batch_tokens = get_tokenizer()(
            article_texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]
        assert (
            batch_tokens.size(0) == batch_size
        ), f"Expected batch size {batch_size}, got {batch_tokens.size(0)}"

        # Instead of aggregating all key-value pairs, yield a KeyValuePair for each chunk
        for i in range(NUM_KV_PAIRS):
            j = i * KV_EVERY_N  # compute the starting index multiplier
            start_idx = j * chunk_size
            key_end_idx = start_idx + TOKENS_PER_KEY
            value_end_idx = key_end_idx + TOKENS_PER_VALUE

            # Batched slicing: extract pair keys and values with shape (batch_size, TOKENS_PER_KEY) and (batch_size, TOKENS_PER_VALUE)
            pair_keys = batch_tokens[:, start_idx:key_end_idx]
            pair_values = batch_tokens[:, key_end_idx:value_end_idx]

            # For logging, decode each row in the batch
            key_text_list = tokenizer.batch_decode(pair_keys.tolist(), clean_up_tokenization_spaces=False)
            value_text_list = tokenizer.batch_decode(pair_values.tolist(), clean_up_tokenization_spaces=False)

            # Compute embeddings for the key tokens if embedding_fn is provided
            if embedding_fn is not None:
                key_embedding = embedding_fn(pair_keys)
            else:
                embedding_dim = 768  # Default embedding dimension
                key_embedding = torch.zeros((batch_size, embedding_dim), device=DEVICE)

            yield KeyValuePair(
                key_tokens=pair_keys,
                value_tokens=pair_values,
                key_embedding=key_embedding,
                key_text=key_text_list,
                value_text=value_text_list,
            )
