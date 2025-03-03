"""
Test script for the data module.

This script tests the functionality of the data module with real data.
"""
import torch
from transformers import AutoModel

from src.data import (
    get_tokenizer,
    iter_key_value_pairs,
    format_prompt_with_kv_pairs
)
from src.config import (
    MODEL_NAME,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    NUM_KV_PAIRS, 
    KV_SUBSET_FRACTION,
    DEVICE
)


def test_data_loading():
    """Test loading and processing data."""
    print("Testing data loading...")
    
    # Get the tokenizer
    tokenizer = get_tokenizer()
    print(f"Loaded tokenizer: {tokenizer}")
    
    # Create a simple embedding function for testing
    def dummy_embedding_fn(tokens):
        batch_size = tokens.shape[0]
        return torch.ones((batch_size, 768), device=DEVICE)
    
    # Process articles and print statistics
    print(f"Processing articles with parameters:")
    print(f"- TOKENS_PER_KEY: {TOKENS_PER_KEY}")
    print(f"- TOKENS_PER_VALUE: {TOKENS_PER_VALUE}")
    print(f"- NUM_KV_PAIRS: {NUM_KV_PAIRS}")
    print(f"- KV_SUBSET_FRACTION: {KV_SUBSET_FRACTION}")
    
    # Iterate through key-value pairs
    iterator = iter_key_value_pairs(batch_size=2, embedding_fn=dummy_embedding_fn)
    
    # Process a few batches
    num_batches = 3
    for i in range(num_batches):
        try:
            print(f"\nProcessing batch {i+1}/{num_batches}...")
            kv_pairs, articles = next(iterator)
            
            print(f"Batch contains {len(kv_pairs)} articles with key-value pairs:")
            
            for j, kv_pair in enumerate(kv_pairs):
                article = articles[j]
                print(f"\nArticle {j+1}: {article.get('title', 'No title')}")
                print(f"- Number of key-value pairs: {len(kv_pair.key_text)}")
                print(f"- Key tokens shape: {kv_pair.key_tokens.shape}")
                print(f"- Value tokens shape: {kv_pair.value_tokens.shape}")
                print(f"- Key embedding shape: {kv_pair.key_embedding.shape}")
                
                # Print sample key-value pairs
                num_samples = min(2, len(kv_pair.key_text))
                print(f"\nSample key-value pairs:")
                for k in range(num_samples):
                    print(f"Key {k+1}: {kv_pair.key_text[k]}")
                    print(f"Value {k+1}: {kv_pair.value_text[k]}")
                
                # Test formatting a prompt
                sample_pairs = [(kv_pair.key_text[k], kv_pair.value_text[k]) for k in range(num_samples)]
                prompt = format_prompt_with_kv_pairs(sample_pairs)
                print(f"\nSample prompt (truncated to 100 chars):")
                print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
                
        except StopIteration:
            print("No more data available.")
            break
        except Exception as e:
            print(f"Error processing batch: {e}")
            break


def test_with_real_model_embeddings():
    """Test data processing with real model embeddings."""
    print("\nTesting with real model embeddings...")
    
    try:
        # Load the model for embeddings
        print(f"Loading model: {MODEL_NAME}")
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
        
        # Create an embedding function using the real model
        def real_embedding_fn(tokens):
            with torch.no_grad():
                outputs = model(tokens)
                # Use the last hidden state, averaged across sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings
        
        # Process one batch with real embeddings
        print("Processing a batch with real embeddings...")
        iterator = iter_key_value_pairs(batch_size=1, embedding_fn=real_embedding_fn)
        kv_pairs, articles = next(iterator)
        
        for j, kv_pair in enumerate(kv_pairs):
            print(f"\nKey embedding statistics:")
            print(f"- Shape: {kv_pair.key_embedding.shape}")
            print(f"- Mean: {kv_pair.key_embedding.mean().item():.4f}")
            print(f"- Std: {kv_pair.key_embedding.std().item():.4f}")
            print(f"- Min: {kv_pair.key_embedding.min().item():.4f}")
            print(f"- Max: {kv_pair.key_embedding.max().item():.4f}")
    
    except Exception as e:
        print(f"Error testing with real model: {e}")
        print("This is expected if the model is not available or too large for the system.")


if __name__ == "__main__":
    # Run the tests
    test_data_loading()
    
    # Only run the model test if GPU is available
    if torch.cuda.is_available():
        test_with_real_model_embeddings()
    else:
        print("\nSkipping real model test as GPU is not available.") 