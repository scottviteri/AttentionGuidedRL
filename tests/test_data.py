"""
Test script to verify the optimized data processing pipeline.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from src.data import (
    iter_key_value_pairs,
    KeyValuePair,
    get_tokenizer,
    format_prompt_with_kv_pairs,
    tokenize_text,
    filter_articles_by_length,
)
from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE


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
            value_text=value_text,
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
    batch_tokens = tokenizer(
        texts, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # Verify the output
    assert isinstance(batch_tokens, torch.Tensor)
    assert batch_tokens.ndim == 2
    assert batch_tokens.shape[0] == len(texts)


def test_data_iterator():
    """Test the optimized data iterator."""
    print("Testing optimized data processing pipeline...")

    # Test with small batch size
    batch_size = 1

    # Get the iterator
    iterator = iter_key_value_pairs(batch_size=batch_size)

    # Process one batch
    kv_pair = next(iterator)

    # Verify KeyValuePair object
    assert isinstance(kv_pair, KeyValuePair)
    assert kv_pair.key_tokens.ndim == 2
    assert kv_pair.key_tokens.shape[1] == TOKENS_PER_KEY
    assert kv_pair.value_tokens.shape[1] == TOKENS_PER_VALUE

    # Verify that first dimension matches batch_size exactly
    assert kv_pair.key_tokens.shape[0] == batch_size
    assert kv_pair.value_tokens.shape[0] == batch_size
    assert kv_pair.key_embedding.shape[0] == batch_size
    assert len(kv_pair.key_text) == batch_size
    assert len(kv_pair.value_text) == batch_size

    # Test with larger batch size
    batch_size = 2

    # Get the iterator
    iterator = iter_key_value_pairs(batch_size=batch_size)

    # Process one batch
    kv_pair = next(iterator)

    # Verify that first dimension matches batch_size exactly
    assert kv_pair.key_tokens.shape[0] == batch_size
    assert kv_pair.value_tokens.shape[0] == batch_size
    assert kv_pair.key_embedding.shape[0] == batch_size
    assert len(kv_pair.key_text) == batch_size
    assert len(kv_pair.value_text) == batch_size


def test_format_prompt_with_kv_pairs():
    """Test formatting key-value pairs into a prompt."""
    # Create sample key-value pairs
    pairs = [
        ("key1", "value1"),
        ("key2", "value2"),
    ]
    
    # Call the function
    prompt = format_prompt_with_kv_pairs(pairs)
    
    # Check the result
    expected_prompt = " Query: key1 Value: value1 Query: key2 Value: value2"
    assert prompt == expected_prompt


def test_keyvaluepair_validation():
    """Test KeyValuePair post-initialization validation."""
    # Set up test data
    batch_size = 2
    embedding_dim = 768
    
    # Valid inputs
    key_tokens = torch.randint(0, 1000, (batch_size, 10))
    value_tokens = torch.randint(0, 1000, (batch_size, 10))
    key_embedding = torch.randn(batch_size, embedding_dim)
    key_text = ["key1", "key2"]
    value_text = ["value1", "value2"]
    
    # Create KeyValuePair with valid inputs
    kv_pair = KeyValuePair(
        key_tokens=key_tokens,
        value_tokens=value_tokens,
        key_embedding=key_embedding,
        key_text=key_text,
        value_text=value_text,
    )
    
    # Test that no exception is raised
    assert kv_pair.key_tokens.shape == (batch_size, 10)
    assert kv_pair.value_tokens.shape == (batch_size, 10)
    assert kv_pair.key_embedding.shape == (batch_size, embedding_dim)
    
    # Test with invalid inputs (e.g., wrong shape)
    with pytest.raises(AssertionError):
        invalid_kv_pair = KeyValuePair(
            key_tokens=torch.randint(0, 1000, (batch_size, 15)),  # Wrong shape (15 instead of 10)
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text,
        )


@patch("src.data.AutoTokenizer.from_pretrained")
def test_get_tokenizer(mock_from_pretrained):
    """Test getting the tokenizer."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_from_pretrained.return_value = mock_tokenizer
    
    # Call the function
    tokenizer = get_tokenizer()
    
    # Check that from_pretrained was called
    mock_from_pretrained.assert_called_once()
    
    # Check that pad_token was set to eos_token
    assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
    
    # Check that the function returned the tokenizer
    assert tokenizer == mock_tokenizer


@patch("src.data.AutoTokenizer.from_pretrained")
def test_tokenize_text(mock_from_pretrained):
    """Test tokenizing text."""
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3], [4, 5, 6]]}
    mock_from_pretrained.return_value = mock_tokenizer
    
    # Single string
    single_text = "Test text"
    result = tokenize_text(single_text, mock_tokenizer)
    
    # Check that encode was called with the right parameters
    mock_tokenizer.encode.assert_called_with(single_text, add_special_tokens=False)
    
    # Check that the function returned the tokenized text
    assert result == [1, 2, 3, 4]
    
    # List of strings
    texts = ["Text 1", "Text 2"]
    result = tokenize_text(texts, mock_tokenizer)
    
    # Check that the tokenizer was called with the right parameters
    mock_tokenizer.assert_called_with(
        texts, add_special_tokens=False, padding=False, truncation=True
    )
    
    # Check that the function returned the tokenized texts
    assert result == [[1, 2, 3], [4, 5, 6]]


@patch("src.data.iter_wikipedia_articles")
@patch("src.data.tokenize_text")
def test_filter_articles_by_length(mock_tokenize_text, mock_iter_wikipedia_articles):
    """Test filtering articles by length."""
    # Mock article iterator
    articles = [
        {"text": "Short article"},
        {"text": "This is a long enough article to pass the filter"},
        {"text": "Another short one"},
        {"text": "This article should also pass the length filter"},
    ]
    mock_iter_wikipedia_articles.return_value = iter(articles)
    
    # Mock tokenize_text to return different lengths for different articles
    def mock_tokenize_side_effect(text, _):
        if text == "Short article" or text == "Another short one":
            return [0] * 10  # Too short
        else:
            return [0] * 2000  # Long enough
    
    mock_tokenize_text.side_effect = mock_tokenize_side_effect
    
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    
    # Call the function and convert the iterator to a list
    filtered_articles = list(filter_articles_by_length(mock_tokenizer))
    
    # Check that only the long articles were returned
    assert len(filtered_articles) == 2
    assert filtered_articles[0]["text"] == "This is a long enough article to pass the filter"
    assert filtered_articles[1]["text"] == "This article should also pass the length filter"


@patch("src.data.filter_articles_by_length")
@patch("src.data.get_tokenizer")
def test_iter_key_value_pairs(mock_get_tokenizer, mock_filter_articles_by_length):
    """Test iterating over key-value pairs."""
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.batch_decode.return_value = ["key text", "value text"]
    mock_tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (2, 2000))  # 2 articles, 2000 tokens each
    }
    mock_get_tokenizer.return_value = mock_tokenizer
    
    # Mock filter_articles_by_length
    articles = [
        {"text": "Article 1"},
        {"text": "Article 2"},
    ]
    mock_filter_articles_by_length.return_value = iter(articles)
    
    # Mock embedding function
    embedding_fn = MagicMock()
    embedding_dim = 768
    embedding_fn.return_value = torch.randn(2, embedding_dim)  # 2 embeddings (batch_size)
    
    # Call the function and get the first result
    kv_pair_iterator = iter_key_value_pairs(batch_size=2, embedding_fn=embedding_fn)
    first_kv_pair = next(kv_pair_iterator)
    
    # Check that the result is a KeyValuePair
    assert isinstance(first_kv_pair, KeyValuePair)
    
    # Check that the shapes are correct
    assert first_kv_pair.key_tokens.shape == (2, 10)  # batch_size, TOKENS_PER_KEY
    assert first_kv_pair.value_tokens.shape == (2, 10)  # batch_size, TOKENS_PER_VALUE
    assert first_kv_pair.key_embedding.shape == (2, embedding_dim)  # batch_size, embedding_dim
    
    # Check that tokenizer's batch_decode was called
    assert mock_tokenizer.batch_decode.call_count >= 2


def test_create_key_value_pair_with_real_model(gpt2_model, gpt2_tokenizer):
    """Test creating key-value pairs with a real GPT-2 model."""
    from src.data import KeyValuePair
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    
    # Setup test data
    key_text = ["This is a test key"]
    value_text = ["This is a test value"]
    batch_size = 1
    
    # Tokenize the text
    key_tokens = gpt2_tokenizer(
        key_text, 
        return_tensors="pt",
        padding="max_length",
        max_length=10,
        truncation=True
    ).input_ids.to(gpt2_model.device)
    
    value_tokens = gpt2_tokenizer(
        value_text,
        return_tensors="pt",
        padding="max_length",
        max_length=10,
        truncation=True
    ).input_ids.to(gpt2_model.device)
    
    # Create key embedding
    key_embedding = torch.randn(batch_size, gpt2_model.config.n_embd, device=gpt2_model.device)
    
    # Create key-value pair directly
    kv_pair = KeyValuePair(
        key_tokens=key_tokens,
        value_tokens=value_tokens,
        key_embedding=key_embedding,
        key_text=key_text,
        value_text=value_text
    )
    
    # Verify the pair was created correctly
    assert kv_pair is not None
    assert kv_pair.key_text == key_text
    assert kv_pair.value_text == value_text
    assert kv_pair.key_tokens.device == gpt2_model.device
    assert kv_pair.value_tokens.device == gpt2_model.device
    assert kv_pair.key_embedding.device == gpt2_model.device
    assert kv_pair.key_embedding.shape == (batch_size, gpt2_model.config.n_embd)
    assert kv_pair.key_tokens.shape == (batch_size, 10)
    assert kv_pair.value_tokens.shape == (batch_size, 10)


def test_real_model_data_pipeline(gpt2_model, gpt2_tokenizer):
    """Test the full data pipeline with a real GPT-2 model."""
    from src.data import KeyValuePair, tokenize_text
    from src.embeddings import register_embedding_hook, extract_embeddings
    from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE
    import torch
    
    # Setup embedding extraction with a real model
    with patch('src.embeddings.MODEL_TYPE', 'gpt2'):
        embeddings_dict, hook_remover = register_embedding_hook(gpt2_model)
    
    try:
        # Create sample texts
        key_text = ["What is the capital of France?", "How does photosynthesis work?"]
        value_text = ["The capital of France is Paris.", "Photosynthesis is the process by which plants convert sunlight into energy."]
        
        # Define an embedding function using the real model
        def embedding_fn(tokens):
            return extract_embeddings(gpt2_model, tokens, embeddings_dict)
        
        # Tokenize texts with a real tokenizer
        key_tokens_list = tokenize_text(key_text, gpt2_tokenizer)
        value_tokens_list = tokenize_text(value_text, gpt2_tokenizer)
        
        # Convert to tensors and pad to required lengths
        # Add batch dimension if needed
        device = gpt2_model.device
        
        # Pad or truncate key tokens
        padded_key_tokens = []
        for tokens in key_tokens_list:
            if len(tokens) > 10:  # Using 10 instead of TOKENS_PER_KEY
                padded_key_tokens.append(tokens[:10])
            else:
                padded_key_tokens.append(tokens + [gpt2_tokenizer.pad_token_id] * (10 - len(tokens)))
        
        # Pad or truncate value tokens
        padded_value_tokens = []
        for tokens in value_tokens_list:
            if len(tokens) > 10:  # Using 10 instead of TOKENS_PER_VALUE
                padded_value_tokens.append(tokens[:10])
            else:
                padded_value_tokens.append(tokens + [gpt2_tokenizer.pad_token_id] * (10 - len(tokens)))
        
        # Convert to tensors
        key_tokens = torch.tensor(padded_key_tokens, device=device)
        value_tokens = torch.tensor(padded_value_tokens, device=device)
        
        # Get key embeddings
        key_embedding = embedding_fn(key_tokens)
        
        # Create a KeyValuePair directly
        kv_pair = KeyValuePair(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text,
        )
        
        # Verify the structure
        assert hasattr(kv_pair, 'key_tokens')
        assert hasattr(kv_pair, 'value_tokens')
        assert hasattr(kv_pair, 'key_embedding')
        assert hasattr(kv_pair, 'key_text')
        assert hasattr(kv_pair, 'value_text')
        
        # Check shapes
        batch_size = len(key_text)
        embedding_dim = gpt2_model.config.n_embd
        
        assert kv_pair.key_embedding.shape[0] == batch_size
        assert kv_pair.key_embedding.shape[1] == embedding_dim
        
        # Check that embeddings are on the right device
        assert kv_pair.key_embedding.device == gpt2_model.device
        
        # Test with mismatched batch sizes (should raise an error)
        mixed_key_text = ["Single question"]
        mixed_value_text = ["Answer one", "Answer two"]
        
        # Tokenize and pad the texts
        mixed_key_tokens_list = tokenize_text(mixed_key_text, gpt2_tokenizer)
        mixed_value_tokens_list = tokenize_text(mixed_value_text, gpt2_tokenizer)
        
        # Pad key tokens
        padded_mixed_key_tokens = []
        for tokens in mixed_key_tokens_list:
            if len(tokens) > 10:  # Using 10 instead of TOKENS_PER_KEY
                padded_mixed_key_tokens.append(tokens[:10])
            else:
                padded_mixed_key_tokens.append(tokens + [gpt2_tokenizer.pad_token_id] * (10 - len(tokens)))
        
        # Pad value tokens
        padded_mixed_value_tokens = []
        for tokens in mixed_value_tokens_list:
            if len(tokens) > 10:  # Using 10 instead of TOKENS_PER_VALUE
                padded_mixed_value_tokens.append(tokens[:10])
            else:
                padded_mixed_value_tokens.append(tokens + [gpt2_tokenizer.pad_token_id] * (10 - len(tokens)))
        
        # Convert to tensors
        mixed_key_tokens = torch.tensor(padded_mixed_key_tokens, device=device)
        mixed_value_tokens = torch.tensor(padded_mixed_value_tokens, device=device)
        
        # Get embeddings for the single key
        mixed_key_embedding = embedding_fn(mixed_key_tokens)
        
        # Should raise an assertion error due to batch size mismatch
        with pytest.raises(AssertionError):
            KeyValuePair(
                key_tokens=mixed_key_tokens,
                value_tokens=mixed_value_tokens, 
                key_embedding=mixed_key_embedding,
                key_text=mixed_key_text,
                value_text=mixed_value_text,
            )
    finally:
        # Clean up hook
        hook_remover()


if __name__ == "__main__":
    # Run the tests manually
    test_data_iterator()
