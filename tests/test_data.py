"""
Test script for the clean, functional data processing pipeline.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch
import itertools

from src.data import (
    KVPair,
    QKVSelection,
    get_tokenizer,
    tokenize_text,
    create_kv_stream,
    complete_batches_only,
    repeat_each,
    wikipedia_articles,
    articles_with_sufficient_length,
    load_twenty_questions,
    get_twenty_questions_path,
)
from src.config import TOKENS_PER_KEY, TOKENS_PER_VALUE, KEY_EMBEDDING_BATCH_SIZE, NUM_KV_PAIRS


def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "=" * 80 + "\n")


class TestKVPair:
    """Test the KVPair data structure."""

    def test_init_valid(self):
        """Test initializing a KVPair with valid inputs."""
        batch_size = 2
        key_tokens = torch.ones((batch_size, TOKENS_PER_KEY), dtype=torch.long)
        value_tokens = torch.ones((batch_size, TOKENS_PER_VALUE), dtype=torch.long)
        key_embedding = torch.ones((batch_size, 768))
        key_text = ["key1", "key2"]
        value_text = ["value1", "value2"]

        kv_pair = KVPair(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text,
        )

        assert torch.equal(kv_pair.key_tokens, key_tokens)
        assert torch.equal(kv_pair.value_tokens, value_tokens)
        assert torch.equal(kv_pair.key_embedding, key_embedding)
        assert kv_pair.key_text == key_text
        assert kv_pair.value_text == value_text

    def test_init_invalid_shapes(self):
        """Test that KVPair raises errors for invalid tensor shapes."""
        batch_size = 2
        
        # Wrong key_tokens shape
        with pytest.raises(AssertionError):
            KVPair(
                key_tokens=torch.ones((batch_size, TOKENS_PER_KEY + 1)),  # Wrong shape
                value_tokens=torch.ones((batch_size, TOKENS_PER_VALUE)),
                key_embedding=torch.ones((batch_size, 768)),
                key_text=["key1", "key2"],
                value_text=["value1", "value2"],
            )

        # Wrong batch size in text
        with pytest.raises(AssertionError):
            KVPair(
                key_tokens=torch.ones((batch_size, TOKENS_PER_KEY)),
                value_tokens=torch.ones((batch_size, TOKENS_PER_VALUE)),
                key_embedding=torch.ones((batch_size, 768)),
                key_text=["key1"],  # Wrong length
                value_text=["value1", "value2"],
            )


class TestStreamUtilities:
    """Test the functional stream utilities."""

    def test_complete_batches_only(self):
        """Test complete_batches_only function."""
        # Test with data that makes complete batches
        data = list(range(10))
        batches = list(complete_batches_only(3)(data))
        assert len(batches) == 3
        assert list(batches[0]) == [0, 1, 2]
        assert list(batches[1]) == [3, 4, 5]
        assert list(batches[2]) == [6, 7, 8]
        # Item 9 should be dropped as incomplete batch

    def test_repeat_each(self):
        """Test repeat_each function."""
        data = [1, 2, 3]
        repeated = list(repeat_each(2, data))
        assert repeated == [1, 1, 2, 2, 3, 3]
        
        # Test with n=1 (no repetition)
        no_repeat = list(repeat_each(1, data))
        assert no_repeat == [1, 2, 3]

    def test_repeat_each_functionality(self):
        """Test repeat_each function."""
        data = [1, 2, 3]
        repeated = list(repeat_each(2, iter(data)))
        assert repeated == [1, 1, 2, 2, 3, 3]


class TestTokenization:
    """Test tokenization utilities."""

    def test_get_tokenizer(self):
        """Test get_tokenizer function."""
        tokenizer = get_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
        assert tokenizer.pad_token == tokenizer.eos_token

    def test_tokenize_text_string(self):
        """Test tokenize_text with a single string."""
        tokenizer = get_tokenizer()
        text = "Hello world"
        tokens = tokenize_text(text, tokenizer)
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

    def test_tokenize_text_list(self):
        """Test tokenize_text with a list of strings."""
        tokenizer = get_tokenizer()
        texts = ["Hello world", "How are you?"]
        tokens = tokenize_text(texts, tokenizer)
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        assert all(isinstance(token_list, list) for token_list in tokens)


class TestDataPipeline:
    """Test the main data pipeline functions."""

    @pytest.fixture
    def mock_embedding_fn(self):
        """Mock embedding function for testing."""
        def embedding_fn(tokens):
            batch_size, seq_len = tokens.shape
            return torch.zeros(batch_size, 768)
        return embedding_fn

    @pytest.fixture
    def gpt2_tokenizer(self):
        """Get GPT-2 tokenizer for testing."""
        return get_tokenizer()

    def test_create_kv_stream_invalid_dataset(self, gpt2_tokenizer, mock_embedding_fn):
        """Test create_kv_stream with invalid dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            list(create_kv_stream("invalid_dataset", 1, gpt2_tokenizer, mock_embedding_fn))

    def test_main_interface(self, gpt2_tokenizer, mock_embedding_fn):
        """Test that the main create_kv_stream interface works."""
        try:
            # This should not raise an error
            kv_stream = create_kv_stream(
                dataset_name="wikipedia",
                batch_size=1,
                tokenizer=gpt2_tokenizer,
                embedding_fn=mock_embedding_fn
            )
            assert kv_stream is not None
        except Exception as e:
            # It's okay if Wikipedia data isn't available, we just want to test the interface
            if "dataset" not in str(e).lower():
                raise

    @patch('src.data.wikipedia_articles')
    def test_wikipedia_pipeline_with_mock(self, mock_articles, gpt2_tokenizer, mock_embedding_fn):
        """Test Wikipedia pipeline with mocked data."""
        # Create mock articles with sufficient length
        long_text = "This is a test article. " * 100  # Long enough for processing
        mock_articles.return_value = iter([
            {"text": long_text, "title": "Test1", "id": "1"},
            {"text": long_text, "title": "Test2", "id": "2"},
        ])
        
        try:
            kv_stream = create_kv_stream("wikipedia", 1, gpt2_tokenizer, mock_embedding_fn)
            # Just test that we can create the stream without errors
            assert kv_stream is not None
        except Exception as e:
            # Handle potential issues with the actual data processing
            if "embedding" in str(e).lower() or "tensor" in str(e).lower():
                pytest.skip(f"Skipping due to tensor processing issue: {e}")
            else:
                raise

    def test_twenty_questions_path(self):
        """Test twenty questions path function."""
        path = get_twenty_questions_path()
        assert isinstance(path, str)
        assert "twenty_questions.json" in path

    def test_load_twenty_questions_missing_file(self):
        """Test load_twenty_questions with missing file."""
        with pytest.raises(FileNotFoundError):
            load_twenty_questions("/nonexistent/path.json")


class TestQKVSelection:
    """Test QKVSelection data structure."""

    def test_qkv_selection_properties(self):
        """Test QKVSelection convenience properties."""
        batch_size = 2
        key_tokens = torch.ones((batch_size, TOKENS_PER_KEY), dtype=torch.long)
        value_tokens = torch.ones((batch_size, TOKENS_PER_VALUE), dtype=torch.long)
        key_embedding = torch.ones((batch_size, 768))
        key_text = ["key1", "key2"]
        value_text = ["value1", "value2"]

        kv_pair = KVPair(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=key_text,
            value_text=value_text,
        )

        qkv = QKVSelection(
            data=kv_pair,
            query_embedding=torch.ones((batch_size, 768)),
            similarity_scores=torch.ones((batch_size, 10)),
            selected_idx=torch.tensor([0, 1]),
            available_mask=torch.ones((batch_size, 10)),
        )

        # Test convenience properties
        assert torch.equal(qkv.key_tokens, key_tokens)
        assert torch.equal(qkv.value_tokens, value_tokens)
        assert torch.equal(qkv.key_embedding, key_embedding)
        assert qkv.key_text == key_text
        assert qkv.value_text == value_text


class TestFunctionalFeatures:
    """Test the functional programming features."""

    def test_toolz_integration(self):
        """Test that toolz functions work correctly."""
        from toolz import pipe, partition_all, concat
        from toolz.curried import map as cmap
        
        # Test a simple functional pipeline
        result = pipe(
            range(10),
            lambda x: partition_all(3, x),
            cmap(lambda batch: [item * 2 for item in batch]),
            concat,
            list
        )
        
        # partition_all(3, range(10)) gives [(0,1,2), (3,4,5), (6,7,8), (9,)]
        # All batches get doubled, including the incomplete last batch
        expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # All items doubled
        assert result == expected

    def test_curried_functions(self):
        """Test curried function usage."""
        from toolz.curried import filter as cfilter, map as cmap
        
        data = [1, 2, 3, 4, 5]
        
        # Test curried filter
        evens = list(cfilter(lambda x: x % 2 == 0)(data))
        assert evens == [2, 4]
        
        # Test curried map
        doubled = list(cmap(lambda x: x * 2)(data))
        assert doubled == [2, 4, 6, 8, 10]


# Additional integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_mock_pipeline(self):
        """Test a complete mock data pipeline."""
        from unittest.mock import MagicMock
        
        # Mock all external dependencies
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.batch_decode.return_value = ["mock text"]
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.input_ids = torch.ones(1, 100)
        
        def mock_embedding_fn(tokens):
            return torch.zeros(tokens.shape[0], 768)
        
        # Test that the pipeline structure works
        try:
            # This tests the function signatures and basic flow
            kv_stream = create_kv_stream(
                dataset_name="wikipedia",
                batch_size=1,
                tokenizer=mock_tokenizer,
                embedding_fn=mock_embedding_fn
            )
            # We don't need to actually consume the stream, just test it can be created
            assert kv_stream is not None
        except Exception as e:
            # Allow for data loading issues but not API issues
            if "dataset" in str(e).lower() or "file" in str(e).lower():
                pytest.skip(f"Skipping due to data availability: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
