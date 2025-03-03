"""
Test script to verify the optimized data processing pipeline.
"""
import torch
import time
from typing import Tuple, List, Dict
import pytest

from src.data import (
    iter_key_value_pairs,
    KeyValuePair,
    get_tokenizer,
    format_prompt_with_kv_pairs
)
from src.config import (
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS,
    KV_SUBSET_FRACTION,
    DEVICE
)


def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")


class TestKeyValuePair:
    """Test the KeyValuePair data structure."""
    
    def test_init_valid(self):
        """Test initializing a KeyValuePair with valid inputs."""
        batch_size = 2
        
        # Create valid tensors
        key_tokens = torch.ones((batch_size, TOKENS_PER_KEY), dtype=torch.long)
        value_tokens = torch.ones((batch_size, TOKENS_PER_VALUE), dtype=torch.long)
        key_embedding = torch.ones((batch_size, 768))
        key_text = ["key1", "key2"]
        value_text = ["value1", "value2"]
        
        # Create a KeyValuePair object
        kv_pair = KeyValuePair(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text
        )
        
        # Verify the object was created correctly
        assert kv_pair.key_tokens.shape == (batch_size, TOKENS_PER_KEY)
        assert kv_pair.value_tokens.shape == (batch_size, TOKENS_PER_VALUE)
        assert kv_pair.key_embedding.shape == (batch_size, 768)
        assert len(kv_pair.key_text) == batch_size
        assert len(kv_pair.value_text) == batch_size


def test_tokenize_text():
    """Test the tokenize_text function."""
    tokenizer = get_tokenizer()
    
    # Test with a single string
    text = "This is a test text."
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    
    # Verify the output
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    # Test with a list of strings
    texts = ["Text one.", "Text two."]
    batch_tokens = tokenizer(texts, add_special_tokens=False, return_tensors="pt").input_ids
    
    # Verify the output
    assert isinstance(batch_tokens, torch.Tensor)
    assert batch_tokens.ndim == 2
    assert batch_tokens.shape[0] == len(texts)


def test_data_iterator():
    """Test the optimized data iterator."""
    print("Testing optimized data processing pipeline...")
    
    # Create a dummy embedding function
    def dummy_embedding_fn(tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        return torch.ones((batch_size, 768), device=DEVICE)

    # Test with small batch size
    batch_size = 1
    
    # Get the iterator
    iterator = iter_key_value_pairs(batch_size=batch_size, embedding_fn=dummy_embedding_fn)
    
    # Process one batch
    kv_pair, articles = next(iterator)
    
    # Verify KeyValuePair object
    assert isinstance(kv_pair, KeyValuePair)
    assert kv_pair.key_tokens.ndim == 2
    assert kv_pair.key_tokens.shape[1] == TOKENS_PER_KEY
    assert kv_pair.value_tokens.shape[1] == TOKENS_PER_VALUE
    
    # Verify article batch
    assert len(articles) == batch_size
    
    # Test with larger batch size
    batch_size = 2
    
    # Get the iterator
    iterator = iter_key_value_pairs(batch_size=batch_size, embedding_fn=dummy_embedding_fn)
    
    # Process one batch
    kv_pair, articles = next(iterator)
    
    # Verify the batch size
    assert len(articles) == batch_size
    
    # Verify key-value pair scaling
    # The number of pairs should scale with batch size (approximately)
    assert kv_pair.key_tokens.shape[0] > 1


def test_format_prompt_with_kv_pairs():
    """Test the format_prompt_with_kv_pairs function."""
    # Create sample key-value pairs
    pairs = [
        ("This is a key", "This is a value"),
        ("Another key", "Another value")
    ]
    
    # Format the prompt
    prompt = format_prompt_with_kv_pairs(pairs)
    
    # Verify the prompt contains the expected content
    assert " Query: " in prompt
    assert " Response: " in prompt
    assert "This is a key" in prompt
    assert "This is a value" in prompt
    
    # Verify the format is correct
    assert prompt.startswith(" Query: ")


if __name__ == "__main__":
    # Run the tests manually
    test_data_iterator() 