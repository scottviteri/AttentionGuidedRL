
"""
Data handling module for the Attention-Guided RL project.

Clean, functional data pipeline using toolz for stream processing.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union, Callable
import json
import os
import itertools

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

# Toolz for functional programming
from toolz import (
    compose, partition_all, concat, take, peek, 
    pipe, curry, identity, do
)
from toolz.curried import map as cmap, filter as cfilter

from src.config import CONFIG

import time
import torch
import logging # Ensure logging is imported here


# === Core Data Structures ===

@dataclass(frozen=True)
class KVPair:
    """Immutable key-value pair representation."""
    key_tokens: torch.Tensor       # [batch_size, CONFIG.tokens_per_key]
    value_tokens: torch.Tensor     # [batch_size, CONFIG.tokens_per_value]
    key_embedding: torch.Tensor    # [batch_size, embedding_dim]
    key_text: List[str]           # For debugging
    value_text: List[str]         # For debugging

    def __post_init__(self):
        """Validate tensor shapes and types."""
        batch_size = self.key_tokens.shape[0]
        assert isinstance(self.key_tokens, torch.Tensor)
        assert isinstance(self.value_tokens, torch.Tensor)
        assert isinstance(self.key_embedding, torch.Tensor)
        assert self.key_tokens.shape == (batch_size, CONFIG.tokens_per_key)
        assert self.value_tokens.shape == (batch_size, CONFIG.tokens_per_value)
        assert self.key_embedding.shape[0] == batch_size
        assert len(self.key_text) == batch_size
        assert len(self.value_text) == batch_size


@dataclass(frozen=True)
class QKVSelection:
    """Complete query-key-value selection with metadata."""
    data: KVPair
    query_embedding: torch.Tensor     # [batch_size, embedding_dim]
    similarity_scores: torch.Tensor   # [batch_size, num_keys]
    selected_idx: torch.Tensor        # [batch_size]
    available_mask: torch.Tensor      # [batch_size, num_keys]

    # Convenience properties
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


@dataclass(frozen=True)
class RawTrajectory:
    """Trajectory without rewards."""
    qkv_steps: List[QKVSelection]
    all_key_embeddings: torch.Tensor  # [batch, num_keys, hidden]


@dataclass(frozen=True)
class Trajectory(RawTrajectory):
    """Complete trajectory with rewards."""
    rewards: torch.Tensor      # [batch, num_steps]
    avg_reward: torch.Tensor   # [batch]


# === Core Functions ===

def get_tokenizer() -> PreTrainedTokenizer:
    """Get configured tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def tokenize_text(text: Union[str, List[str]], tokenizer: PreTrainedTokenizer) -> Union[List[int], List[List[int]]]:
    """Tokenize text into token IDs."""
    if isinstance(text, str):
        return tokenizer.encode(text, add_special_tokens=False)
    else:
        encoding = tokenizer(text, add_special_tokens=False, padding=False, truncation=True)
        return encoding["input_ids"]


# === Stream Utilities ===

def complete_batches_only(batch_size: int):
    """Return only complete batches from a stream."""
    def batches(stream):
        for batch in partition_all(batch_size, stream):
            if len(batch) == batch_size:
                yield batch
    return batches


@curry
def repeat_each(n: int, stream: Iterator):
    """Repeat each item n times (for GRPO batching)."""
    for item in stream:
        for _ in range(n):
            yield item


def repeat_n_times(n: int, stream: Iterator):
    """Repeat each item n times (alias for repeat_each for backward compatibility)."""
    return repeat_each(n, stream)


def debug_stream(stream: Iterator, name: str, max_items: int = 5):
    """Debug stream by printing items."""
    import logging
    count = 0
    for item in stream:
        if count < max_items:
            logging.info(f"[{name}] Item {count}: {type(item).__name__}")
            count += 1
        yield item


def count_stream(stream: Iterator, name: str = "stream"):
    """Count items in stream and log periodically."""
    import logging
    count = 0
    for item in stream:
        count += 1
        if count % 100 == 0:
            logging.info(f"[{name}] Processed {count} items")
        yield item


def time_stream(stream: Iterator, name: str = "stream"):
    """Time stream processing."""
    import logging
    import time
    start_time = time.time()
    count = 0
    for item in stream:
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            logging.info(f"[{name}] {count} items in {elapsed:.2f}s ({count/elapsed:.2f} items/s)")
        yield item


def peek_stream(stream: Iterator, peek_count: int = 1):
    """Peek at first few items without consuming them."""
    import logging
    from itertools import chain
    items = []
    for i, item in enumerate(stream):
        if i < peek_count:
            items.append(item)
            logging.info(f"[peek] Item {i}: {type(item).__name__}")
        # Chain the peeked items back with the rest of the stream
        return chain(items, [item] if i >= peek_count else [], stream)
    # If stream was shorter than peek_count
    return iter(items)


# === Wikipedia Data Pipeline ===

def wikipedia_articles() -> Iterator[Dict]:
    """Stream Wikipedia articles."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    return iter(dataset)


def articles_with_sufficient_length(tokenizer: PreTrainedTokenizer) -> Iterator[Dict]:
    """Filter articles by minimum token length."""
    min_length = (CONFIG.tokens_per_key + CONFIG.tokens_per_value) * CONFIG.num_kv_pairs * CONFIG.kv_every_n
    
    def has_sufficient_tokens(article: Dict) -> bool:
        tokens = tokenize_text(article["text"], tokenizer)
        return len(tokens) >= min_length
    
    return filter(has_sufficient_tokens, wikipedia_articles())


def tokenize_articles(tokenizer: PreTrainedTokenizer, max_len: int):
    """Tokenize a batch of articles."""
    def tokenize_batch(articles: List[Dict]) -> torch.Tensor:
        texts = [article["text"] for article in articles]
        tokens = tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return tokens.input_ids.to(CONFIG.device)
    return tokenize_batch


def extract_kv_pairs(tokenizer: PreTrainedTokenizer):
    """Extract key-value pairs from tokenized articles."""
    def extract_from_tokens(batch_tokens: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], List[List[str]]]:
        chunk_size = CONFIG.tokens_per_key + CONFIG.tokens_per_value
        all_keys, all_values, all_key_texts, all_value_texts = [], [], [], []
        
        for i in range(CONFIG.num_kv_pairs):
            j = i * CONFIG.kv_every_n
            start_idx = j * chunk_size
            key_end_idx = start_idx + CONFIG.tokens_per_key
            value_end_idx = key_end_idx + CONFIG.tokens_per_value

            pair_keys = batch_tokens[:, start_idx:key_end_idx]
            pair_values = batch_tokens[:, key_end_idx:value_end_idx]

            key_texts = tokenizer.batch_decode(pair_keys.tolist(), clean_up_tokenization_spaces=False)
            value_texts = tokenizer.batch_decode(pair_values.tolist(), clean_up_tokenization_spaces=False)

            all_keys.append(pair_keys)
            all_values.append(pair_values)
            all_key_texts.append(key_texts)
            all_value_texts.append(value_texts)

        return all_keys, all_values, all_key_texts, all_value_texts
    return extract_from_tokens


def compute_embeddings(embedding_fn: Callable[[torch.Tensor], torch.Tensor], batch_size: int):
    """Compute embeddings for keys in batches."""
    def compute_for_keys(all_keys: List[torch.Tensor]) -> List[torch.Tensor]:
        # Start timing and memory tracking for key embedding computation
        start_time = time.time()
        initial_mem_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        initial_mem_cached = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

        all_embeddings = []
        for start_idx in range(0, CONFIG.num_kv_pairs, 12): # Use a reasonable default batch size
            end_idx = min(start_idx + 12, CONFIG.num_kv_pairs)
            key_batch = torch.stack(all_keys[start_idx:end_idx], dim=0)
            key_batch_flat = key_batch.view(-1, CONFIG.tokens_per_key)
            
            embeddings_flat = embedding_fn(key_batch_flat)
            
            embeddings = embeddings_flat.view(end_idx - start_idx, batch_size, -1)
            all_embeddings.extend(embeddings)
        
        end_time = time.time()
        final_mem_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        final_mem_cached = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0

        total_time_ms = (end_time - start_time) * 1000
        
        # Only log memory details for first few episodes or every 100 episodes to reduce verbosity
        episode_num = getattr(compute_for_keys, '_episode_counter', 0)
        compute_for_keys._episode_counter = episode_num + 1
        
        if episode_num < 5 or episode_num % 100 == 0:
            logging.info(f"Key embedding computation took: {total_time_ms:.2f} ms")
            if torch.cuda.is_available():
                # Report increase in allocated memory
                allocated_increase_mb = (final_mem_allocated - initial_mem_allocated) / (1024 * 1024)
                # Report total cached/reserved memory
                cached_memory_mb = final_mem_cached / (1024 * 1024)
                logging.info(f"  CUDA Allocated Memory Increase: {allocated_increase_mb:.2f} MB")
                logging.info(f"  CUDA Total Cached Memory: {cached_memory_mb:.2f} MB")

        return all_embeddings
    return compute_for_keys


def articles_to_kv_pairs(tokenizer: PreTrainedTokenizer, embedding_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Convert article batches to KV pairs."""
    max_len = (CONFIG.tokens_per_key + CONFIG.tokens_per_value) * CONFIG.num_kv_pairs * CONFIG.kv_every_n
    
    def process_article_batch(articles: List[Dict]) -> Iterator[KVPair]:
        # Pipeline: articles -> tokens -> kv_data -> embeddings -> KVPairs
        batch_tokens = tokenize_articles(tokenizer, max_len)(articles)
        all_keys, all_values, all_key_texts, all_value_texts = extract_kv_pairs(tokenizer)(batch_tokens)
        all_embeddings = compute_embeddings(embedding_fn, len(articles))(all_keys)
        
        for i in range(CONFIG.num_kv_pairs):
            yield KVPair(
                key_tokens=all_keys[i],
                value_tokens=all_values[i],
                key_embedding=all_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )
    return process_article_batch


# === Main Data Pipeline ===

def wikipedia_kv_stream(batch_size: int, tokenizer: PreTrainedTokenizer, embedding_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[KVPair]:
    """Clean Wikipedia KV stream using functional composition."""
    return pipe(
        articles_with_sufficient_length(tokenizer),
        complete_batches_only(batch_size),
        cmap(articles_to_kv_pairs(tokenizer, embedding_fn)),
        concat
    )


# === Twenty Questions Data Pipeline ===

def load_twenty_questions(dataset_path: str) -> Dict:
    """Load twenty questions dataset."""
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback tiny dataset for tests if file is missing
    return {
        "questions": [
            "Is it an animal?",
            "Is it bigger than a breadbox?",
            "Is it man-made?",
            "Is it used daily?",
            "Is it electronic?",
        ],
        "data": [
            {"answers": ["Yes", "No", "Yes", "Yes", "No"]},
            {"answers": ["No", "Yes", "No", "No", "Yes"]},
        ],
    }


def get_twenty_questions_path() -> str:
    """Get default twenty questions dataset path."""
    return os.path.join(os.path.dirname(__file__), "..", "data", "twenty_questions.json")


def twenty_questions_kv_stream(batch_size: int, tokenizer: PreTrainedTokenizer, embedding_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[KVPair]:
    """Twenty questions KV stream."""
    dataset = load_twenty_questions(get_twenty_questions_path())
    questions = dataset['questions']
    games = dataset['data']
    
    def process_game_batch(game_batch: List[Dict]) -> Iterator[KVPair]:
        batch_size_actual = len(game_batch)
        all_key_tokens, all_value_tokens, all_key_texts, all_value_texts = [], [], [], []
        
        for q_idx in range(min(len(questions), CONFIG.num_kv_pairs)):
            key_texts = [questions[q_idx] for _ in game_batch]
            value_texts = [game['answers'][q_idx] for game in game_batch]
            
            key_tokens = tokenizer(key_texts, add_special_tokens=False, padding="max_length", 
                                 truncation=True, max_length=CONFIG.tokens_per_key, return_tensors="pt")["input_ids"].to(CONFIG.device)
            value_tokens = tokenizer(value_texts, add_special_tokens=False, padding="max_length",
                                   truncation=True, max_length=CONFIG.tokens_per_value, return_tensors="pt")["input_ids"].to(CONFIG.device)
            
            all_key_tokens.append(key_tokens)
            all_value_tokens.append(value_tokens)
            all_key_texts.append(key_texts)
            all_value_texts.append(value_texts)

        all_embeddings = compute_embeddings(embedding_fn, batch_size_actual)(all_key_tokens)
        
        for i in range(len(all_key_tokens)):
            yield KVPair(
                key_tokens=all_key_tokens[i],
                value_tokens=all_value_tokens[i],
                key_embedding=all_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )
    
    return pipe(
        iter(games),
        complete_batches_only(batch_size),
        cmap(process_game_batch),
        concat
    )


# === Unified Interface ===

def create_kv_stream(dataset_name: str, batch_size: int, tokenizer: PreTrainedTokenizer, embedding_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[KVPair]:
    """Create a KV stream for any dataset."""
    if dataset_name == "wikipedia":
        return wikipedia_kv_stream(batch_size, tokenizer, embedding_fn)
    elif dataset_name == "twenty_questions":
        return twenty_questions_kv_stream(batch_size, tokenizer, embedding_fn)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Backward compatibility alias
def iter_key_value_pairs_unified_with_tokenizer(dataset_name: str, batch_size: int, tokenizer: PreTrainedTokenizer, embedding_fn: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[KVPair]:
    """Backward compatibility alias for create_kv_stream."""
    return create_kv_stream(dataset_name, batch_size, tokenizer, embedding_fn)



