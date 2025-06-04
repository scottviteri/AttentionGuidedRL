"""
Test script to verify the optimized data processing pipeline.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch
import itertools

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
    """Test the data iterator."""
    print("Testing data processing pipeline...")

    # Set up mock articles
    batch_size = 2

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


def test_get_tokenizer_real():
    """Test ``get_tokenizer`` without mocks using the actual implementation."""
    tokenizer = get_tokenizer()

    # The tokenizer should have its ``pad_token`` set to the EOS token by ``get_tokenizer``.
    assert tokenizer.pad_token == tokenizer.eos_token

    # Sanity-check that the tokenizer can encode some text.
    sample_ids = tokenizer.encode("Hello world", add_special_tokens=False)
    assert isinstance(sample_ids, list) and len(sample_ids) > 0


def test_filter_articles_by_length():
    """Test filtering articles by length."""
    # Use the real tokenizer
    tokenizer = get_tokenizer()
    
    # Get the filtered articles using the actual function, limit to first 10 for testing
    filtered_articles = list(itertools.islice(filter_articles_by_length(tokenizer), 10))
    
    # Basic assertion to check if articles are returned
    assert len(filtered_articles) > 0, "No articles were returned after filtering."
    
    # Additional checks can be added based on expected dataset properties
    for article in filtered_articles:
        assert "text" in article, "Article does not contain 'text' key."
        # Optionally, check if the article text meets a minimum length requirement
        tokenized_text = tokenize_text([article["text"]], tokenizer)
        # tokenize_text returns List[List[int]], so we check len(tokenized_text[0])
        assert len(tokenized_text[0]) >= 10, f"Article text too short: {article['text'][:50]}..."


def test_iter_key_value_pairs():
    """Test iterating over key-value pairs."""
    # Use the real tokenizer
    tokenizer = get_tokenizer()

    # Define a real embedding function (assuming a model is available for testing)
    # For this test, we'll use a dummy embedding function if a model isn't set up
    def dummy_embedding_fn(tokens):
        batch_size = tokens.shape[0]
        embedding_dim = 768  # Typical embedding dimension
        return torch.randn(batch_size, embedding_dim, device=tokens.device)
    
    # Call the function and get the first result using the actual data pipeline
    kv_pair_iterator = iter_key_value_pairs(batch_size=2, embedding_fn=dummy_embedding_fn)
    first_kv_pair = next(kv_pair_iterator)
    
    # Check that the result is a KeyValuePair
    assert isinstance(first_kv_pair, KeyValuePair)
    
    # Check that the shapes are correct
    assert first_kv_pair.key_tokens.shape == (2, TOKENS_PER_KEY)  # batch_size, TOKENS_PER_KEY
    assert first_kv_pair.value_tokens.shape == (2, TOKENS_PER_VALUE)  # batch_size, TOKENS_PER_VALUE
    assert first_kv_pair.key_embedding.shape[0] == 2  # batch_size
    assert first_kv_pair.key_embedding.shape[1] > 0  # embedding_dim should be positive
    
    # Ensure token text lists match batch size
    assert len(first_kv_pair.key_text) == 2
    assert len(first_kv_pair.value_text) == 2


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


def test_load_twenty_questions_dataset():
    """Test loading the twenty questions dataset."""
    from src.data import load_twenty_questions_dataset
    
    # Load the dataset
    dataset = load_twenty_questions_dataset()
    
    # Check the structure
    assert isinstance(dataset, dict), "Dataset should be a dictionary"
    assert 'questions' in dataset, "Dataset should have 'questions' key"
    assert 'all_objects' in dataset, "Dataset should have 'all_objects' key"
    assert 'data' in dataset, "Dataset should have 'data' key"
    
    # Check data types
    assert isinstance(dataset['questions'], list), "Questions should be a list"
    assert isinstance(dataset['all_objects'], list), "Objects should be a list"
    assert isinstance(dataset['data'], list), "Data should be a list"
    
    # Check that we have data
    assert len(dataset['questions']) > 0, "Should have at least one question"
    assert len(dataset['all_objects']) > 0, "Should have at least one object"
    assert len(dataset['data']) > 0, "Should have at least one game"
    
    # Check the structure of a game
    first_game = dataset['data'][0]
    assert 'object' in first_game, "Game should have 'object' key"
    assert 'answers' in first_game, "Game should have 'answers' key"
    assert isinstance(first_game['answers'], list), "Answers should be a list"
    assert len(first_game['answers']) == len(dataset['questions']), "Should have answers for all questions"


def test_iter_twenty_questions():
    """Test iterating over twenty questions games."""
    from src.data import iter_twenty_questions
    
    # Get a few games
    games = list(itertools.islice(iter_twenty_questions(), 5))
    
    # Check that we got games
    assert len(games) > 0, "Should have at least one game"
    
    # Check game structure
    for game in games:
        assert isinstance(game, dict), "Game should be a dictionary"
        assert 'object' in game, "Game should have 'object' key"
        assert 'answers' in game, "Game should have 'answers' key"
        assert all(answer in ['YES', 'NO'] for answer in game['answers']), "Answers should be YES or NO"


def test_iter_twenty_questions_pairs():
    """Test iterating over twenty questions key-value pairs."""
    from src.data import iter_twenty_questions_pairs, KeyValuePair
    
    batch_size = 2
    
    # Define a dummy embedding function
    def dummy_embedding_fn(tokens):
        batch_size = tokens.shape[0]
        embedding_dim = 768
        return torch.randn(batch_size, embedding_dim, device=tokens.device)
    
    # Get the iterator
    iterator = iter_twenty_questions_pairs(batch_size=batch_size, embedding_fn=dummy_embedding_fn)
    
    # Get a few pairs
    pairs = list(itertools.islice(iterator, 5))
    
    # Check that we got pairs
    assert len(pairs) > 0, "Should have at least one pair"
    
    # Check pair structure
    for pair in pairs:
        assert isinstance(pair, KeyValuePair), "Should be a KeyValuePair instance"
        assert pair.key_tokens.shape == (batch_size, TOKENS_PER_KEY)
        assert pair.value_tokens.shape == (batch_size, TOKENS_PER_VALUE)
        assert pair.key_embedding.shape[0] == batch_size
        assert len(pair.key_text) == batch_size
        assert len(pair.value_text) == batch_size
        
        # Check that values are YES or NO
        for value in pair.value_text:
            assert value in ['YES', 'NO'], f"Value should be YES or NO, got {value}"


def test_iter_key_value_pairs_unified():
    """Test the unified iterator for different datasets."""
    from src.data import iter_key_value_pairs_unified, KeyValuePair
    
    batch_size = 2
    
    # Define a dummy embedding function
    def dummy_embedding_fn(tokens):
        batch_size = tokens.shape[0]
        embedding_dim = 768
        return torch.randn(batch_size, embedding_dim, device=tokens.device)
    
    # Test with Wikipedia dataset
    wiki_iterator = iter_key_value_pairs_unified(
        dataset_name="wikipedia",
        batch_size=batch_size,
        embedding_fn=dummy_embedding_fn
    )
    wiki_pair = next(wiki_iterator)
    assert isinstance(wiki_pair, KeyValuePair), "Should return KeyValuePair for Wikipedia"
    assert wiki_pair.key_tokens.shape == (batch_size, TOKENS_PER_KEY)
    
    # Test with Twenty Questions dataset  
    tq_iterator = iter_key_value_pairs_unified(
        dataset_name="twenty_questions", 
        batch_size=batch_size,
        embedding_fn=dummy_embedding_fn
    )
    tq_pair = next(tq_iterator)
    assert isinstance(tq_pair, KeyValuePair), "Should return KeyValuePair for Twenty Questions"
    assert tq_pair.key_tokens.shape == (batch_size, TOKENS_PER_KEY)
    
    # Test with invalid dataset name
    with pytest.raises(ValueError) as excinfo:
        iter_key_value_pairs_unified(
            dataset_name="invalid_dataset",
            batch_size=batch_size,
            embedding_fn=dummy_embedding_fn
        )
    assert "Unknown dataset" in str(excinfo.value)
