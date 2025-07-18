#!/usr/bin/env python3
"""
Debug script to track which articles are used in each episode
to understand the gradient spike pattern.
"""

from src.data import iter_key_value_pairs_unified_with_tokenizer
from src.model import setup_model_and_tokenizer
from src.config import NUM_KV_PAIRS
import torch

def identify_article(key_text):
    """Extract a short identifier from a key text to identify the source article."""
    if not key_text:
        return "Unknown"
    
    # Take first 30 characters and clean up
    identifier = key_text[:30].strip()
    
    # Common patterns to identify articles
    if "Anarchism" in identifier or "anarchist" in identifier:
        return "Anarchism"
    elif "political spectrum" in identifier or "libertarian" in identifier:
        return "Political_Spectrum"  
    elif "Albedo" in identifier or "albedo" in identifier:
        return "Albedo"
    elif identifier.startswith(" "):
        # If it starts with space, it's likely a continuation
        return f"Continuation: {identifier[:20]}"
    else:
        return f"Other: {identifier[:20]}"

def main():
    print("=== Debugging Data Pattern ===")
    
    # Set up tokenizer
    try:
        _, _, tokenizer = setup_model_and_tokenizer()
    except Exception as e:
        print(f"Error setting up models: {e}")
        return
    
    # Mock embedding function
    def mock_embedding_fn(tokens):
        return torch.randn(tokens.shape[0], 768).detach()
    
    # Create the data iterator (single items to see source)
    kv_iterator = iter_key_value_pairs_unified_with_tokenizer(
        dataset_name='wikipedia',
        batch_size=1,
        tokenizer=tokenizer,
        embedding_fn=mock_embedding_fn,
    )
    
    print(f"Tracking data across episodes (NUM_KV_PAIRS = {NUM_KV_PAIRS})")
    print("Episode | Key Index | Article Identifier")
    print("--------|-----------|------------------")
    
    episode = 0
    total_items = 0
    
    try:
        for episode in range(10):  # Check first 10 episodes
            print(f"\n=== Episode {episode} ===")
            
            episode_articles = set()
            
            # Get NUM_KV_PAIRS items for this episode
            for key_idx in range(NUM_KV_PAIRS):
                try:
                    kv_pair = next(kv_iterator)
                    key_text = kv_pair.key_text[0] if kv_pair.key_text else None
                    article_id = identify_article(key_text)
                    episode_articles.add(article_id)
                    
                    print(f"{episode:7d} | {key_idx:9d} | {article_id}")
                    total_items += 1
                    
                except StopIteration:
                    print(f"Iterator exhausted at episode {episode}, key {key_idx}")
                    return
                except Exception as e:
                    print(f"Error at episode {episode}, key {key_idx}: {e}")
                    return
            
            print(f"Episode {episode} summary: {len(episode_articles)} unique articles: {episode_articles}")
            
            # Check if this is a "mixed" episode (could correlate with spikes)
            if len(episode_articles) > 1:
                print(f"*** MIXED EPISODE {episode} - POTENTIAL SPIKE EPISODE ***")
    
    except Exception as e:
        print(f"Error during iteration: {e}")
        print(f"Processed {total_items} total items across {episode+1} episodes")

if __name__ == "__main__":
    main() 