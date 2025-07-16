"""
Data handling module for the Attention-Guided RL project.

Contains data structures and utilities for loading, processing, and batching data.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union, Optional
import json
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import (
    TOKENIZER_NAME,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS,
    VALUE_PREFIX,
    DEVICE,
    KV_EVERY_N,
    QUERY_VEC_TOKEN,
    USE_STANDARD_QUERY_TOKEN,
    KEY_EMBEDDING_BATCH_SIZE,
    KEY_PREFIX,
)


@dataclass(frozen=True)
class KVPair:
    """
    Immutable representation of key-value data before any query processing or selection.
    
    This represents the raw, initial state of a key-value pair extracted from data.
    All fields are required and the object is immutable after creation.
    
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

        assert self.key_tokens.shape == (batch_size, TOKENS_PER_KEY), \
            f"key_tokens shape should be ({batch_size}, {TOKENS_PER_KEY})"
        assert self.value_tokens.shape == (batch_size, TOKENS_PER_VALUE), \
            f"value_tokens shape should be ({batch_size}, {TOKENS_PER_VALUE})"
        assert self.key_embedding.shape[0] == batch_size, \
            f"key_embedding first dimension should be {batch_size}"
        assert len(self.key_text) == batch_size, \
            f"key_text length should be {batch_size}"
        assert len(self.value_text) == batch_size, \
            f"value_text length should be {batch_size}"


@dataclass(frozen=True)
class QKVSelection:
    """
    Immutable representation of a complete query-key-value selection.
    
    This represents the final state after a query has been generated, similarities computed,
    and a key-value pair has been selected. All data is immutable after creation.
    
    Attributes:
        data: The selected KVPair that was chosen
        query_text: Query text that led to selecting this key-value pair
        query_tokens: Tokenized query that led to selecting this pair
        query_embedding: Query embeddings [batch_size, embedding_dim]
        similarity_scores: Similarity scores between query and all available keys [batch_size, num_keys]
        selected_idx: Index of the selected key in the original pool
    """
    # Core selected data
    data: KVPair
    
    # Query information
    query_text: List[str]
    query_tokens: torch.Tensor
    query_embedding: torch.Tensor  # Shape: [batch_size, embedding_dim]
    
    # Selection metadata
    similarity_scores: torch.Tensor  # Shape: [batch_size, num_keys]
    selected_idx: torch.Tensor  # Index of the selected key in the original pool (or tensor of indices)
    available_mask: torch.Tensor  # Mask of available keys at selection time [batch_size, num_keys] (0 for available, -inf for masked)

    def __post_init__(self):
        """Validate tensor shapes and types."""
        batch_size = self.data.key_tokens.shape[0]
        
        # Validate query data
        assert isinstance(self.query_embedding, torch.Tensor), "query_embedding must be a tensor"
        assert self.query_embedding.shape[0] == batch_size, \
            f"query_embedding first dimension should be {batch_size}"
            
        # Validate selection metadata
        assert isinstance(self.similarity_scores, torch.Tensor), "similarity_scores must be a tensor"
        assert self.similarity_scores.shape[0] == batch_size, \
            f"similarity_scores first dimension should be {batch_size}"
            
        # Validate selected_idx - must be a tensor
        assert isinstance(self.selected_idx, torch.Tensor), "selected_idx must be a torch.Tensor"
        assert self.selected_idx.shape == (batch_size,) or self.selected_idx.shape == torch.Size([batch_size]), \
            f"selected_idx tensor should have shape ({batch_size},), got {self.selected_idx.shape}"

        # Validate available_mask
        assert isinstance(self.available_mask, torch.Tensor), "available_mask must be a torch.Tensor"
        assert self.available_mask.shape == self.similarity_scores.shape, \
            "available_mask must have same shape as similarity_scores"

    # Convenience properties to access the underlying data
    @property
    def key_tokens(self) -> torch.Tensor:
        return self.data.key_tokens
    
    @property
    def value_tokens(self) -> torch.Tensor:
        return self.data.value_tokens
    
    @property
    def key_embedding(self) -> torch.Tensor:
        return self.data.key_embedding
    
    @property
    def key_text(self) -> List[str]:
        return self.data.key_text
    
    @property
    def value_text(self) -> List[str]:
        return self.data.value_text


# Maintain backward compatibility
QKVStep = QKVSelection  # Alias for existing code
QKVStepWithSelection = QKVSelection  # Transition alias
QKVData = KVPair  # Transition alias  
KeyValuePair = KVPair  # Update the old alias


def get_tokenizer() -> PreTrainedTokenizer:
    """
    Get the tokenizer for the model.

    Returns:
        PreTrainedTokenizer: The tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    
    # Conditionally add special token only if not using standard tokens
    if not USE_STANDARD_QUERY_TOKEN:
        # Add the special query vector token only when using special token mode
        special_tokens_dict = {'additional_special_tokens': [QUERY_VEC_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
    
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
    # KEY_PREFIX is already imported at module level; removed redundant import
    prompt = ""
    for key, value in pairs:
        prompt += f"{KEY_PREFIX}{key} {VALUE_PREFIX}{value} "

    return prompt.strip()  # Remove trailing space


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
) -> Iterator[KVPair]:
    """
    Create an iterator that yields batches of query-key-value data.

    Args:
        batch_size: Number of articles to process in each batch
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding a batched KVPair object
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
        )["input_ids"].to(DEVICE)  # Ensure tokens are on the correct device from the start
        
        assert (
            batch_tokens.size(0) == batch_size
        ), f"Expected batch size {batch_size}, got {batch_tokens.size(0)}"

        # Extract all key-value pairs first
        all_keys = []
        all_values = []
        all_key_texts = []
        all_value_texts = []
        
        for i in range(NUM_KV_PAIRS):
            j = i * KV_EVERY_N  # compute the starting index multiplier
            start_idx = j * chunk_size
            key_end_idx = start_idx + TOKENS_PER_KEY
            value_end_idx = key_end_idx + TOKENS_PER_VALUE

            # Batched slicing: extract pair keys and values with shape (batch_size, TOKENS_PER_KEY) and (batch_size, TOKENS_PER_VALUE)
            pair_keys = batch_tokens[:, start_idx:key_end_idx]  # Already on the device
            pair_values = batch_tokens[:, key_end_idx:value_end_idx]  # Already on the device

            # For logging, decode each row in the batch
            key_text_list = tokenizer.batch_decode(pair_keys.tolist(), clean_up_tokenization_spaces=False)
            value_text_list = tokenizer.batch_decode(pair_values.tolist(), clean_up_tokenization_spaces=False)

            all_keys.append(pair_keys)
            all_values.append(pair_values)
            all_key_texts.append(key_text_list)
            all_value_texts.append(value_text_list)

        # Batch process key embeddings if embedding_fn is provided
        if embedding_fn is not None:
            all_key_embeddings = []
            
            # Process keys in batches of KEY_EMBEDDING_BATCH_SIZE
            for start_idx in range(0, NUM_KV_PAIRS, KEY_EMBEDDING_BATCH_SIZE):
                end_idx = min(start_idx + KEY_EMBEDDING_BATCH_SIZE, NUM_KV_PAIRS)
                
                # Stack keys for this batch: [KEY_EMBEDDING_BATCH_SIZE, batch_size, TOKENS_PER_KEY]
                key_batch = torch.stack(all_keys[start_idx:end_idx], dim=0)
                
                # Reshape to [KEY_EMBEDDING_BATCH_SIZE * batch_size, TOKENS_PER_KEY] for processing
                key_batch_flat = key_batch.view(-1, TOKENS_PER_KEY)
                
                # Process this batch of keys
                embeddings_batch_flat = embedding_fn(key_batch_flat)
                
                # Reshape back to [KEY_EMBEDDING_BATCH_SIZE, batch_size, embedding_dim]
                embeddings_batch = embeddings_batch_flat.view(
                    end_idx - start_idx, batch_size, -1
                )
                
                # Add embeddings for each key in this batch
                for i in range(embeddings_batch.shape[0]):
                    all_key_embeddings.append(embeddings_batch[i])
        else:
            # Default embeddings if no embedding function
            embedding_dim = 768  # Default embedding dimension
            all_key_embeddings = [
                torch.zeros((batch_size, embedding_dim), device=DEVICE) 
                for _ in range(NUM_KV_PAIRS)
            ]

        # Yield KVPair objects one by one
        for i in range(NUM_KV_PAIRS):
            yield KVPair(
                key_tokens=all_keys[i],
                value_tokens=all_values[i],
                key_embedding=all_key_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )


def load_twenty_questions_dataset(dataset_path: str = None) -> Dict:
    """
    Load the twenty questions dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file

    Returns:
        Dict: The dataset as a dictionary with keys 'questions', 'all_objects', and 'data'
    """
    if dataset_path is None:
        # Use default path relative to the project root
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "20q_dataset.json"
        )

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    return dataset


def iter_twenty_questions() -> Iterator[Dict]:
    """
    Create an iterator that yields twenty questions games.

    Returns:
        Iterator[Dict]: Iterator yielding game dictionaries with object and answers
    """
    dataset = load_twenty_questions_dataset()
    
    # The dataset has a 'data' key that contains the games
    for game in dataset['data']:
        yield game


def iter_twenty_questions_pairs(
    batch_size: int = 1, embedding_fn=None
) -> Iterator[KVPair]:
    """
    Create an iterator that yields batches of query-key-value data from the twenty questions dataset.

    Args:
        batch_size: Number of games to process in each batch
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding a batched KVPair object
    """
    tokenizer = get_tokenizer()
    dataset = load_twenty_questions_dataset()
    questions = dataset['questions']
    games = dataset['data']
    
    # Process games in chunks without wrapping
    game_idx = 0
    
    while game_idx < len(games):
        # Collect batch_size number of games
        game_batch = []
        for _ in range(batch_size):
            if game_idx >= len(games):
                # Not enough games to fill the batch, stop here
                break
            game_batch.append(games[game_idx])
            game_idx += 1
        
        # Only process if we have a full batch or it's the last batch
        if len(game_batch) == 0:
            break
            
        # If it's not a full batch and we need exactly batch_size, skip this batch
        if len(game_batch) < batch_size:
            # For the last batch, we can either skip it or pad it
            # Here we choose to skip incomplete batches
            break
        
        # Extract all question-answer pairs first
        all_key_tokens = []
        all_value_tokens = []
        all_key_texts = []
        all_value_texts = []
        
        for q_idx in range(min(len(questions), NUM_KV_PAIRS)):
            # Prepare batch data
            key_texts = []
            value_texts = []
            
            for game in game_batch:
                # The key is the question
                key_texts.append(questions[q_idx])
                # The value is the answer (YES/NO)
                value_texts.append(game['answers'][q_idx])
            
            # Tokenize in batch
            key_tokens = tokenizer(
                key_texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=TOKENS_PER_KEY,
                return_tensors="pt",
            )["input_ids"].to(DEVICE)
            
            value_tokens = tokenizer(
                value_texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=TOKENS_PER_VALUE,
                return_tensors="pt",
            )["input_ids"].to(DEVICE)
            
            all_key_tokens.append(key_tokens)
            all_value_tokens.append(value_tokens)
            all_key_texts.append(key_texts)
            all_value_texts.append(value_texts)

        # Batch process key embeddings if embedding_fn is provided
        if embedding_fn is not None:
            all_key_embeddings = []
            num_questions = len(all_key_tokens)
            
            # Process keys in batches of KEY_EMBEDDING_BATCH_SIZE
            for start_idx in range(0, num_questions, KEY_EMBEDDING_BATCH_SIZE):
                end_idx = min(start_idx + KEY_EMBEDDING_BATCH_SIZE, num_questions)
                
                # Stack keys for this batch: [KEY_EMBEDDING_BATCH_SIZE, batch_size, TOKENS_PER_KEY]
                key_batch = torch.stack(all_key_tokens[start_idx:end_idx], dim=0)
                
                # Reshape to [KEY_EMBEDDING_BATCH_SIZE * batch_size, TOKENS_PER_KEY] for processing
                key_batch_flat = key_batch.view(-1, TOKENS_PER_KEY)
                
                # Process this batch of keys
                embeddings_batch_flat = embedding_fn(key_batch_flat)
                
                # Reshape back to [KEY_EMBEDDING_BATCH_SIZE, batch_size, embedding_dim]
                embeddings_batch = embeddings_batch_flat.view(
                    end_idx - start_idx, batch_size, -1
                )
                
                # Add embeddings for each key in this batch
                for i in range(embeddings_batch.shape[0]):
                    all_key_embeddings.append(embeddings_batch[i])
        else:
            # Default embeddings if no embedding function
            embedding_dim = 768  # Default embedding dimension
            all_key_embeddings = [
                torch.zeros((batch_size, embedding_dim), device=DEVICE) 
                for _ in range(len(all_key_tokens))
            ]

        # Yield KVPair objects one by one
        for i in range(len(all_key_tokens)):
            yield KVPair(
                key_tokens=all_key_tokens[i],
                value_tokens=all_value_tokens[i],
                key_embedding=all_key_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )


def iter_key_value_pairs_unified(
    dataset_name: str = "wikipedia",
    batch_size: int = 1,
    embedding_fn=None
) -> Iterator[KVPair]:
    """
    Unified iterator for different datasets.

    Args:
        dataset_name: Name of the dataset ('wikipedia' or 'twenty_questions')
        batch_size: Number of items to process in each batch
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding batched KVPair objects
    """
    if dataset_name == "wikipedia":
        return iter_key_value_pairs(batch_size, embedding_fn)
    elif dataset_name == "twenty_questions":
        return iter_twenty_questions_pairs(batch_size, embedding_fn)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def iter_key_value_pairs_unified_with_tokenizer(
    dataset_name: str = "wikipedia",
    batch_size: int = 1,
    tokenizer: PreTrainedTokenizer = None,
    embedding_fn=None
) -> Iterator[KVPair]:
    """
    Unified iterator for different datasets that accepts a specific tokenizer instance.

    Args:
        dataset_name: Name of the dataset ('wikipedia' or 'twenty_questions')
        batch_size: Number of items to process in each batch
        tokenizer: Specific tokenizer instance to use (to ensure vocabulary consistency)
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding batched KVPair objects
    """
    if dataset_name == "wikipedia":
        return iter_key_value_pairs_with_tokenizer(batch_size, tokenizer, embedding_fn)
    elif dataset_name == "twenty_questions":
        return iter_twenty_questions_pairs_with_tokenizer(batch_size, tokenizer, embedding_fn)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def iter_key_value_pairs_with_tokenizer(
    batch_size: int = 1, 
    tokenizer: PreTrainedTokenizer = None, 
    embedding_fn=None
) -> Iterator[KVPair]:
    """
    Create an iterator that yields batches of query-key-value data with a specific tokenizer.

    Args:
        batch_size: Number of articles to process in each batch
        tokenizer: Specific tokenizer instance to use
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding a batched KVPair object
    """
    if tokenizer is None:
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
        batch_tokens = tokenizer(
            article_texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"].to(DEVICE)  # Ensure tokens are on the correct device from the start
        
        assert (
            batch_tokens.size(0) == batch_size
        ), f"Expected batch size {batch_size}, got {batch_tokens.size(0)}"

        # Extract all key-value pairs first
        all_keys = []
        all_values = []
        all_key_texts = []
        all_value_texts = []
        
        for i in range(NUM_KV_PAIRS):
            j = i * KV_EVERY_N  # compute the starting index multiplier
            start_idx = j * chunk_size
            key_end_idx = start_idx + TOKENS_PER_KEY
            value_end_idx = key_end_idx + TOKENS_PER_VALUE

            # Batched slicing: extract pair keys and values with shape (batch_size, TOKENS_PER_KEY) and (batch_size, TOKENS_PER_VALUE)
            pair_keys = batch_tokens[:, start_idx:key_end_idx]  # Already on the device
            pair_values = batch_tokens[:, key_end_idx:value_end_idx]  # Already on the device

            # For logging, decode each row in the batch
            key_text_list = tokenizer.batch_decode(pair_keys.tolist(), clean_up_tokenization_spaces=False)
            value_text_list = tokenizer.batch_decode(pair_values.tolist(), clean_up_tokenization_spaces=False)

            all_keys.append(pair_keys)
            all_values.append(pair_values)
            all_key_texts.append(key_text_list)
            all_value_texts.append(value_text_list)

        # Batch process key embeddings if embedding_fn is provided
        if embedding_fn is not None:
            all_key_embeddings = []
            
            # Process keys in batches of KEY_EMBEDDING_BATCH_SIZE
            for start_idx in range(0, NUM_KV_PAIRS, KEY_EMBEDDING_BATCH_SIZE):
                end_idx = min(start_idx + KEY_EMBEDDING_BATCH_SIZE, NUM_KV_PAIRS)
                
                # Stack keys for this batch: [KEY_EMBEDDING_BATCH_SIZE, batch_size, TOKENS_PER_KEY]
                key_batch = torch.stack(all_keys[start_idx:end_idx], dim=0)
                
                # Reshape to [KEY_EMBEDDING_BATCH_SIZE * batch_size, TOKENS_PER_KEY] for processing
                key_batch_flat = key_batch.view(-1, TOKENS_PER_KEY)
                
                # Process this batch of keys
                embeddings_batch_flat = embedding_fn(key_batch_flat)
                
                # Reshape back to [KEY_EMBEDDING_BATCH_SIZE, batch_size, embedding_dim]
                embeddings_batch = embeddings_batch_flat.view(
                    end_idx - start_idx, batch_size, -1
                )
                
                # Add embeddings for each key in this batch
                for i in range(embeddings_batch.shape[0]):
                    all_key_embeddings.append(embeddings_batch[i])
        else:
            # Default embeddings if no embedding function
            embedding_dim = 768  # Default embedding dimension
            all_key_embeddings = [
                torch.zeros((batch_size, embedding_dim), device=DEVICE) 
                for _ in range(NUM_KV_PAIRS)
            ]

        # Yield KVPair objects one by one
        for i in range(NUM_KV_PAIRS):
            yield KVPair(
                key_tokens=all_keys[i],
                value_tokens=all_values[i],
                key_embedding=all_key_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )


def iter_twenty_questions_pairs_with_tokenizer(
    batch_size: int = 1, 
    tokenizer: Optional[PreTrainedTokenizer] = None, 
    embedding_fn=None
) -> Iterator[KVPair]:
    """
    Create an iterator that yields batches of query-key-value data from the twenty questions dataset with a specific tokenizer.

    Args:
        batch_size: Number of games to process in each batch
        tokenizer: Specific tokenizer instance to use
        embedding_fn: Optional function to compute embeddings

    Returns:
        Iterator[KVPair]: Iterator yielding a batched KVPair object
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    dataset = load_twenty_questions_dataset()
    questions = dataset['questions']
    games = dataset['data']
    
    # Process games in chunks without wrapping
    game_idx = 0
    
    while game_idx < len(games):
        # Collect batch_size number of games
        game_batch = []
        for _ in range(batch_size):
            if game_idx >= len(games):
                # Not enough games to fill the batch, stop here
                break
            game_batch.append(games[game_idx])
            game_idx += 1
        
        # Only process if we have a full batch or it's the last batch
        if len(game_batch) == 0:
            break
            
        # If it's not a full batch and we need exactly batch_size, skip this batch
        if len(game_batch) < batch_size:
            # For the last batch, we can either skip it or pad it
            # Here we choose to skip incomplete batches
            break
        
        # Extract all question-answer pairs first
        all_key_tokens = []
        all_value_tokens = []
        all_key_texts = []
        all_value_texts = []
        
        for q_idx in range(min(len(questions), NUM_KV_PAIRS)):
            # Prepare batch data
            key_texts = []
            value_texts = []
            
            for game in game_batch:
                # The key is the question
                key_texts.append(questions[q_idx])
                # The value is the answer (YES/NO)
                value_texts.append(game['answers'][q_idx])
            
            # Tokenize in batch
            key_tokens = tokenizer(
                key_texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=TOKENS_PER_KEY,
                return_tensors="pt",
            )["input_ids"].to(DEVICE)
            
            value_tokens = tokenizer(
                value_texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=TOKENS_PER_VALUE,
                return_tensors="pt",
            )["input_ids"].to(DEVICE)
            
            all_key_tokens.append(key_tokens)
            all_value_tokens.append(value_tokens)
            all_key_texts.append(key_texts)
            all_value_texts.append(value_texts)

        # Batch process key embeddings if embedding_fn is provided
        if embedding_fn is not None:
            all_key_embeddings = []
            num_questions = len(all_key_tokens)
            
            # Process keys in batches of KEY_EMBEDDING_BATCH_SIZE
            for start_idx in range(0, num_questions, KEY_EMBEDDING_BATCH_SIZE):
                end_idx = min(start_idx + KEY_EMBEDDING_BATCH_SIZE, num_questions)
                
                # Stack keys for this batch: [KEY_EMBEDDING_BATCH_SIZE, batch_size, TOKENS_PER_KEY]
                key_batch = torch.stack(all_key_tokens[start_idx:end_idx], dim=0)
                
                # Reshape to [KEY_EMBEDDING_BATCH_SIZE * batch_size, TOKENS_PER_KEY] for processing
                key_batch_flat = key_batch.view(-1, TOKENS_PER_KEY)
                
                # Process this batch of keys
                embeddings_batch_flat = embedding_fn(key_batch_flat)
                
                # Reshape back to [KEY_EMBEDDING_BATCH_SIZE, batch_size, embedding_dim]
                embeddings_batch = embeddings_batch_flat.view(
                    end_idx - start_idx, batch_size, -1
                )
                
                # Add embeddings for each key in this batch
                for i in range(embeddings_batch.shape[0]):
                    all_key_embeddings.append(embeddings_batch[i])
        else:
            # Default embeddings if no embedding function
            embedding_dim = 768  # Default embedding dimension
            all_key_embeddings = [
                torch.zeros((batch_size, embedding_dim), device=DEVICE) 
                for _ in range(len(all_key_tokens))
            ]

        # Yield KVPair objects one by one
        for i in range(len(all_key_tokens)):
            yield KVPair(
                key_tokens=all_key_tokens[i],
                value_tokens=all_value_tokens[i],
                key_embedding=all_key_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )


def repeat_n_times(n: int, stream: Iterator) -> Iterator:
    """
    Stream operator that repeats each item from the input stream n times.
    
    This is useful for GRPO-style batching where we want multiple copies
    of the same data point in our batch.
    
    Args:
        n: Number of times to repeat each item
        stream: Input iterator/generator
        
    Yields:
        Each item from the stream, repeated n times
    """
    for item in stream:
        for _ in range(n):
            yield item


def iter_key_value_pairs_unified_with_repeat(
    dataset_name: str = "wikipedia",
    batch_size: int = 1,
    repeat_count: int = 1,
    tokenizer: PreTrainedTokenizer = None,
    embedding_fn=None
) -> Iterator[KVPair]:
    """
    Unified iterator with optional repetition for GRPO-style batching.
    
    Args:
        dataset_name: Name of the dataset ('wikipedia' or 'twenty_questions')
        batch_size: Number of items to process in each batch
        repeat_count: Number of times to repeat each item (for GRPO)
        tokenizer: Specific tokenizer instance to use
        embedding_fn: Optional function to compute embeddings
        
    Returns:
        Iterator[KVPair]: Iterator yielding batched KVPair objects
    """
    # Get the base iterator
    base_iterator = iter_key_value_pairs_unified_with_tokenizer(
        dataset_name=dataset_name,
        batch_size=1,  # Generate single items first
        tokenizer=tokenizer,
        embedding_fn=embedding_fn
    )
    
    # Apply repetition if requested
    if repeat_count > 1:
        repeated_iterator = repeat_n_times(repeat_count, base_iterator)
        
        # Now batch the repeated items
        batched_iterator = batch_iterator(repeated_iterator, batch_size)
        
        # Process batches to create KVPair objects
        for batch in batched_iterator:
            if len(batch) == batch_size:
                # Stack the batch elements into single KVPair
                yield stack_qkv_batch(batch)
    else:
        # Use the original iterator with its own batching
        return iter_key_value_pairs_unified_with_tokenizer(
            dataset_name=dataset_name,
            batch_size=batch_size,
            tokenizer=tokenizer,
            embedding_fn=embedding_fn
        )


def batch_iterator(stream: Iterator, batch_size: int) -> Iterator[List]:
    """
    Batch items from a stream into groups of batch_size.
    
    Args:
        stream: Input iterator
        batch_size: Size of each batch
        
    Yields:
        Lists of items with size batch_size
    """
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # Don't yield incomplete batches


def stack_qkv_batch(batch: List[KVPair]) -> KVPair:
    """
    Stack a list of KVPair objects into a single batched KVPair.
    
    Args:
        batch: List of KVPair objects to stack
        
    Returns:
        Single KVPair with batched tensors
    """
    if not batch:
        raise ValueError("Cannot stack empty batch")
    
    # Get device from first item
    device = batch[0].key_tokens.device
    
    # Stack all tensor fields
    key_tokens = torch.stack([item.key_tokens.squeeze(0) for item in batch], dim=0)
    value_tokens = torch.stack([item.value_tokens.squeeze(0) for item in batch], dim=0)
    key_embedding = torch.stack([item.key_embedding.squeeze(0) for item in batch], dim=0)
    
    # Combine text fields
    key_text = []
    value_text = []
    for item in batch:
        key_text.extend(item.key_text)
        value_text.extend(item.value_text)
    
    # Create batched KVPair
    batched_data = KVPair(
        key_tokens=key_tokens,
        value_tokens=value_tokens,
        key_embedding=key_embedding,
        key_text=key_text,
        value_text=value_text
    )
    
    return batched_data
