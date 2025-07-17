#!/usr/bin/env python3
"""
Standalone program to demonstrate the Wikipedia data loading pipeline.
Shows how elements flow through each stage of the transformation.
"""

import sys
import os
import torch
from typing import Iterator, List, Dict, Callable
import itertools

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import (
    get_tokenizer, 
    wikipedia_articles, 
    complete_batches_only,
    tokenize_articles,
    extract_kv_pairs,
    compute_embeddings,
    KVPair
)
from config import TOKENS_PER_KEY, TOKENS_PER_VALUE, NUM_KV_PAIRS, KV_EVERY_N, DEVICE


def mock_embedding_fn(tokens: torch.Tensor) -> torch.Tensor:
    """Mock embedding function that returns random embeddings."""
    batch_size = tokens.shape[0]
    return torch.randn(batch_size, 768, device=tokens.device)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def print_stage_header(stage_name: str, stage_num: int):
    """Print a formatted header for each pipeline stage."""
    print(f"\n{'='*80}")
    print(f"STAGE {stage_num}: {stage_name}")
    print(f"{'='*80}")


def print_article_info(article: Dict, index: int):
    """Print information about a Wikipedia article."""
    title = article.get('title', 'Unknown')
    text = article.get('text', '')
    url = article.get('url', 'No URL')
    
    print(f"Article #{index}:")
    print(f"  Title: {title}")
    print(f"  URL: {url}")
    print(f"  Text length: {len(text)} characters")
    print(f"  Text preview: {truncate_text(text, 150)}")
    print()


def demonstrate_pipeline():
    """Demonstrate the Wikipedia data loading pipeline step by step."""
    print("WIKIPEDIA DATA LOADING PIPELINE DEMONSTRATION")
    print("=" * 80)
    print("This program shows how data flows through each stage of the pipeline.")
    print("We'll follow the same elements through their transformations.")
    
    # Setup
    tokenizer = get_tokenizer()
    embedding_fn = mock_embedding_fn
    batch_size = 2
    num_elements_to_show = 3  # Reduced to avoid threading issues
    
    # Stage 1: Raw Wikipedia articles
    print_stage_header("Raw Wikipedia Articles", 1)
    print("Loading raw articles from Wikipedia dataset...")
    
    try:
        raw_articles = wikipedia_articles()
        sample_articles = list(itertools.islice(raw_articles, num_elements_to_show))
        
        for i, article in enumerate(sample_articles):
            print_article_info(article, i)
            
    except Exception as e:
        print(f"Error loading Wikipedia articles: {e}")
        print("Using mock data for demonstration...")
        
        # Create mock articles for demonstration
        sample_articles = [
            {
                'title': f'Mock Article {i}', 
                'text': f'This is a very long mock Wikipedia article number {i}. ' * 100,
                'url': f'https://en.wikipedia.org/wiki/Mock_Article_{i}'
            }
            for i in range(num_elements_to_show)
        ]
        
        for i, article in enumerate(sample_articles):
            print_article_info(article, i)
    
    # Stage 2: Filter articles with sufficient length
    print_stage_header("Filtered Articles (Sufficient Length)", 2)
    min_length = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS * KV_EVERY_N
    print(f"Filtering articles with minimum length of {min_length} tokens...")
    
    filtered_articles = []
    for article in sample_articles:
        tokens = tokenizer.encode(article["text"], add_special_tokens=False)
        if len(tokens) >= min_length:
            filtered_articles.append(article)
    
    print(f"Articles after filtering: {len(filtered_articles)}")
    for i, article in enumerate(filtered_articles):
        title = article.get('title', 'Unknown')
        text_len = len(article.get('text', ''))
        tokens = tokenizer.encode(article["text"], add_special_tokens=False)
        print(f"  [{i}] {title} ({text_len} chars, {len(tokens)} tokens)")
    print()
    
    # Stage 3: Complete batches
    print_stage_header("Batched Articles", 3)
    print(f"Grouping articles into complete batches of size {batch_size}...")
    
    batches = list(complete_batches_only(batch_size)(filtered_articles))
    
    for i, batch in enumerate(batches):
        print(f"Batch #{i} (size: {len(batch)}):")
        for j, article in enumerate(batch):
            title = article.get('title', 'Unknown')
            text_len = len(article.get('text', ''))
            print(f"  [{j}] {title} ({text_len} chars)")
        print()
    
    if not batches:
        print("No complete batches available. Need more articles or smaller batch size.")
        return
    
    # Stage 4: Tokenized batches
    print_stage_header("Tokenized Article Batches", 4)
    print("Converting article text to token tensors...")
    
    max_len = (TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS * KV_EVERY_N
    tokenize_fn = tokenize_articles(tokenizer, max_len)
    
    # Process first batch only to avoid complexity
    batch = batches[0]
    try:
        tokens = tokenize_fn(batch)
        print(f"Tokenized Batch #0:")
        print(f"  Shape: {tokens.shape}")
        print(f"  Device: {tokens.device}")
        print(f"  First few tokens of first article: {tokens[0, :20].tolist()}")
        print()
    except Exception as e:
        print(f"Error tokenizing batch: {e}")
        return
    
    # Stage 5: Extracted key-value pairs
    print_stage_header("Extracted Key-Value Pairs", 5)
    print("Extracting key-value pairs from tokenized articles...")
    
    extract_fn = extract_kv_pairs(tokenizer)
    
    try:
        all_keys, all_values, all_key_texts, all_value_texts = extract_fn(tokens)
        print(f"Extracted KV Pairs:")
        print(f"  Number of KV pairs: {len(all_keys)}")
        print(f"  Key tensor shape: {all_keys[0].shape}")
        print(f"  Value tensor shape: {all_values[0].shape}")
        
        # Show first few KV pairs
        for i in range(min(3, len(all_keys))):
            print(f"  KV Pair #{i}:")
            print(f"    Key text: {truncate_text(all_key_texts[i][0], 100)}")
            print(f"    Value text: {truncate_text(all_value_texts[i][0], 100)}")
        print()
    except Exception as e:
        print(f"Error extracting KV pairs: {e}")
        return
    
    # Stage 6: Computed embeddings
    print_stage_header("Computed Key Embeddings", 6)
    print("Computing embeddings for extracted keys...")
    
    try:
        compute_fn = compute_embeddings(embedding_fn, batch_size)
        all_embeddings = compute_fn(all_keys)
        print(f"Computed Embeddings:")
        print(f"  Number of embedding tensors: {len(all_embeddings)}")
        print(f"  First embedding shape: {all_embeddings[0].shape}")
        print(f"  First embedding norm: {torch.norm(all_embeddings[0][0]).item():.4f}")
        print()
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return
    
    # Stage 7: Final KVPair objects
    print_stage_header("Final KVPair Objects", 7)
    print("Creating final KVPair dataclass instances...")
    
    try:
        kv_pairs = []
        for i in range(min(num_elements_to_show, len(all_keys))):
            kv_pair = KVPair(
                key_tokens=all_keys[i],
                value_tokens=all_values[i],
                key_embedding=all_embeddings[i],
                key_text=all_key_texts[i],
                value_text=all_value_texts[i],
            )
            kv_pairs.append(kv_pair)
            
            print(f"KVPair #{i}:")
            print(f"  Key tokens shape: {kv_pair.key_tokens.shape}")
            print(f"  Value tokens shape: {kv_pair.value_tokens.shape}")
            print(f"  Key embedding shape: {kv_pair.key_embedding.shape}")
            print(f"  Key text: {truncate_text(kv_pair.key_text[0], 80)}")
            print(f"  Value text: {truncate_text(kv_pair.value_text[0], 80)}")
            print()
    except Exception as e:
        print(f"Error creating KVPair objects: {e}")
        return
    
    # Summary
    print_stage_header("Pipeline Summary", 8)
    print(f"Successfully processed {len(kv_pairs)} KVPair objects through the complete pipeline.")
    print(f"Each KVPair contains:")
    print(f"  - Key tokens: {TOKENS_PER_KEY} tokens")
    print(f"  - Value tokens: {TOKENS_PER_VALUE} tokens") 
    print(f"  - Key embedding: 768-dimensional vector")
    print(f"  - Human-readable key and value text")
    print("\nThis demonstrates the complete Wikipedia data loading pipeline!")
    print("\nPipeline Flow Summary:")
    print("Raw Articles → Filter by Length → Batch → Tokenize → Extract KV → Embed → KVPair")


if __name__ == "__main__":
    try:
        # Set torch to avoid threading issues
        torch.set_num_threads(1)
        demonstrate_pipeline()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean exit
        print("\nDemo completed.") 